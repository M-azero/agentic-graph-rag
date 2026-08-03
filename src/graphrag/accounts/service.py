"""Accounts: signup, email verification, login, sessions.

The flow is deliberately ordinary — email + password, then a six-digit code to
prove the address is real, then a server-side session in an httpOnly cookie.
Two decisions worth stating:

**Codes only at signup, not every login.** Emailing a code per login would burn
a free-tier sending quota (Resend allows 100/day) and add friction to the most
common action, for little gain over a password the user chose.

**Server-side sessions, not JWTs.** The cookie carries an opaque token; the
database holds only its hash. That makes revocation instant — suspending or
deleting an account cuts every session immediately — which a stateless token
can only match by adding a denylist lookup, i.e. the same round trip with worse
failure modes.

Responses never reveal whether an address is registered. Signup, resend and
password login all answer the same way for known and unknown addresses;
otherwise the endpoints become an account-enumeration oracle.
"""

from __future__ import annotations

import contextlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from graphrag.accounts.emails import EmailSender
from graphrag.accounts.passwords import hash_password, validate_password, verify_password
from graphrag.config.settings import Settings
from graphrag.container import sanitize_user
from graphrag.core.logging import get_logger
from graphrag.db.engine import session_scope
from graphrag.db.models import PURPOSE_RESET, PURPOSE_VERIFY, EmailOTP, User
from graphrag.db.models import Session as SessionRow

log = get_logger(__name__)

_SESSION_CACHE = "graphrag:session:"
_SESSION_CACHE_TTL = 60
_TENANT_SUFFIX_BYTES = 4


class AccountError(Exception):
    """A problem worth showing the user (bad code, weak password, ...)."""

    def __init__(self, message: str, code: str = "invalid") -> None:
        super().__init__(message)
        self.code = code
        # Seconds until the caller may retry. Only meaningful for
        # `too_many_attempts`; 0 everywhere else, so routers can pass it
        # through unconditionally.
        self.retry_after = 0


@dataclass(frozen=True)
class Principal:
    """The authenticated identity attached to a request."""

    user_id: str
    tenant_id: str
    role: str
    email: str = ""

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


@dataclass(frozen=True)
class SessionSummary:
    """One live session, for the "signed-in devices" list.

    Carries `token_hash` so the router can mark which row is the caller's own
    without the service needing to know about cookies.
    """

    id: str
    token_hash: str
    created_at: datetime | None
    last_seen_at: datetime | None
    ip: str | None
    user_agent: str | None


def _now() -> datetime:
    return datetime.now(UTC)


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def _generate_code() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


def _hash_token(token: str) -> str:
    import hashlib

    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def hash_session_token(token: str) -> str:
    """The stored form of a session cookie, for callers that need to compare
    one against a `SessionSummary` without holding the plaintext twice."""
    return _hash_token(token)


def _maybe_uuid(value) -> uuid.UUID | None:
    """A user id as a UUID, or None when it isn't one.

    With auth disabled the identity is whatever `X-User-Id` said, so it is a
    sanitized name rather than a key. Endpoints that reach the accounts tables
    have to answer "no such user" for those rather than raising ValueError out
    of a query and turning a 404 into a 500.
    """
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return None


def lockout_delay(failures: int, threshold: int, base: int, maximum: int) -> int:
    """How long to lock an account after `failures` consecutive bad passwords.

    Zero below the threshold, then doubling from `base` to a ceiling: the first
    lock past the line is short enough that a real user who mistyped waits
    seconds, while a script that keeps going is quickly spending an hour per
    attempt. Pure and parameterised so the policy can be tested without a
    database or a clock.
    """
    if threshold <= 0 or failures < threshold:
        return 0
    # 2**over can be enormous once an attacker keeps hammering a locked
    # account; cap the exponent before computing it rather than after.
    over = min(failures - threshold, 32)
    return min(base * (2**over), maximum)


def _tenant_for(email: str) -> str:
    """A storage namespace derived from the address but not equal to it.

    The tenant id names a Neo4j corpus and a DuckDB filename, so it has to be a
    safe token; the random suffix keeps two similar addresses
    ("a.b@x" / "a-b@x") from colliding after sanitizing, and keeps the
    filesystem from spelling out who owns what.
    """
    local = sanitize_user(email.split("@", 1)[0])[:24].strip("-") or "user"
    return f"{local}-{secrets.token_hex(_TENANT_SUFFIX_BYTES)}"


class AccountService:
    def __init__(
        self,
        factory: async_sessionmaker[AsyncSession] | None,
        settings: Settings,
        email_sender: EmailSender,
        redis_client=None,
    ) -> None:
        self._factory = factory
        self._settings = settings
        self._email = email_sender
        self._redis = redis_client

    @property
    def _auth(self):
        return self._settings.auth

    # -- registration ---------------------------------------------------------
    async def signup(self, email: str, password: str) -> None:
        """Create a pending account and email a verification code.

        Returns nothing on purpose: the caller says the same thing whether or
        not the address was already taken.
        """
        email = normalize_email(email)
        if not email or "@" not in email:
            raise AccountError("Enter a valid email address.", "invalid_email")
        problem = validate_password(password)
        if problem:
            raise AccountError(problem, "weak_password")

        async with session_scope(self._factory) as s:
            existing = await self._by_email(s, email)
            if existing is not None:
                if existing.status == "pending":
                    # Signing up twice before verifying just resends the code.
                    code = await self._issue_otp(s, existing)
                else:
                    code = None
            else:
                user = User(
                    email=email,
                    password_hash=hash_password(password),
                    tenant_id=_tenant_for(email),
                    status="pending",
                    role="user",
                )
                s.add(user)
                await s.flush()
                code = await self._issue_otp(s, user)
                log.info("account_created", user=str(user.id))

        if code is not None:
            await self._send_code(email, code)

    async def resend_code(self, email: str) -> None:
        email = normalize_email(email)
        async with session_scope(self._factory) as s:
            user = await self._by_email(s, email)
            if user is None or user.status != "pending":
                return  # same silence as an unknown address
            code = await self._issue_otp(s, user)
        await self._send_code(email, code)

    async def verify(self, email: str, code: str) -> tuple[Principal, str]:
        """Confirm ownership of the address and open a session.

        Deliberately two transactions. The attempt counter has to be *committed*
        before the code is compared: doing both in one transaction means a
        rejected guess rolls the increment back with the error, the cap never
        rises, and a six-digit code is brute-forceable at request speed.
        """
        email = normalize_email(email)

        # 1. Charge the attempt. Commits even though the guess may be wrong.
        async with session_scope(self._factory) as s:
            user = await self._by_email(s, email)
            if user is None:
                raise AccountError("That code is not valid.", "invalid_code")
            if user.status == "active":
                raise AccountError("This account is already verified.", "already_verified")
            if user.status != "pending":
                raise AccountError("This account cannot be verified.", "not_verifiable")

            otp = await self._latest_otp(s, user, PURPOSE_VERIFY)
            if otp is None:
                raise AccountError("Request a new code.", "no_code")
            if otp.expires_at <= _now():
                raise AccountError("That code has expired. Request a new one.", "expired_code")
            if otp.attempts >= self._auth.otp_max_attempts:
                raise AccountError(
                    "Too many attempts. Request a new code.", "too_many_attempts"
                )

            otp.attempts += 1
            otp_id = otp.id
            matched = secrets.compare_digest(otp.code_hash, _hash_token(code.strip()))

        if not matched:
            raise AccountError("That code is not valid.", "invalid_code")

        # 2. The code was right: consume it and activate the account.
        async with session_scope(self._factory) as s:
            user = await self._by_email(s, email)
            if user is None or user.status != "pending":
                raise AccountError("That code is not valid.", "invalid_code")
            otp = (
                await s.execute(
                    select(EmailOTP).where(
                        EmailOTP.id == otp_id, EmailOTP.consumed_at.is_(None)
                    )
                )
            ).scalar_one_or_none()
            if otp is None:  # raced with another verify of the same code
                raise AccountError("That code is not valid.", "invalid_code")

            otp.consumed_at = _now()
            user.status = "active"
            user.email_verified_at = _now()
            user.last_login_at = _now()
            principal = self._principal(user)
            token = await self._open_session(s, user)
            log.info("account_verified", user=str(user.id))
        return principal, token

    # -- sessions -------------------------------------------------------------
    async def login(
        self, email: str, password: str, ip: str | None = None, user_agent: str | None = None
    ) -> tuple[Principal, str]:
        """Exchange a password for a session, counting failures against a lock.

        The structure is unusual on purpose. Every rejection *records* something
        (a failure count, possibly a lock), and that record has to survive the
        rejection — so the transaction computes a `failure` and exits normally,
        committing, and the error is raised afterwards. Raising from inside
        `session_scope` would roll the increment back with the exception: the
        counter would never rise and the lock would never engage. This is the
        same trap `verify()` documents above, arrived at from the other side.

        `verify_password` runs before any branch, including for a missing or
        locked account, so every outcome costs one Argon2 hash. Skipping it for
        a locked account would make lock state observable by timing, and
        skipping it for an unknown address would make registration observable —
        which is what `_DUMMY_HASH` exists to prevent.

        Accepted trade-off: `too_many_attempts` is only ever returned for an
        address that exists, so it is an enumeration oracle. It is kept because
        the per-IP limit on this endpoint caps how fast that can be farmed, and
        because a locked-out user told nothing concludes the product is broken.
        The same trade is already made by `email_unverified` (403 vs 401), for
        the same reason.
        """
        email = normalize_email(email)
        threshold = self._auth.lockout_threshold
        failure: tuple[str, str, int] | None = None
        principal: Principal | None = None
        token = ""

        async with session_scope(self._factory) as s:
            user = await self._by_email(s, email)
            stored = user.password_hash if user else _DUMMY_HASH
            ok = verify_password(stored, password)
            now = _now()

            locked_for = 0
            if user is not None and user.locked_until is not None:
                locked_for = int((user.locked_until - now).total_seconds())

            if locked_for > 0:
                failure = (
                    "too_many_attempts",
                    "Too many failed attempts. Try again later.",
                    locked_for,
                )
            elif user is None or not ok:
                if user is not None and threshold > 0:
                    user.failed_logins += 1
                    if user.failed_logins >= threshold:
                        wait = lockout_delay(
                            user.failed_logins,
                            threshold,
                            self._auth.lockout_base_seconds,
                            self._auth.lockout_max_seconds,
                        )
                        user.locked_until = now + timedelta(seconds=wait)
                        log.warning(
                            "account_locked",
                            user=str(user.id),
                            failures=user.failed_logins,
                            seconds=wait,
                        )
                failure = ("invalid_credentials", "Email or password is incorrect.", 0)
            elif user.status == "pending":
                failure = ("email_unverified", "Verify your email address first.", 0)
            elif user.status != "active":
                failure = ("account_inactive", "This account is not active.", 0)
            else:
                # A correct password clears the count: the lock is there to slow
                # guessing, not to punish someone who eventually remembered.
                user.failed_logins = 0
                user.locked_until = None
                user.last_login_at = now
                principal = self._principal(user)
                token = await self._open_session(s, user, ip=ip, user_agent=user_agent)

        if failure is not None:
            code, message, retry_after = failure
            error = AccountError(message, code)
            error.retry_after = retry_after
            raise error
        assert principal is not None  # narrowing: set on the success branch
        return principal, token

    async def unlock(self, user_id: str) -> bool:
        """Clear a lockout. The admin action, and the CLI break-glass behind
        `graphrag unlock` for when the only admin is the one locked out."""
        async with session_scope(self._factory) as s:
            user = await self._by_id(s, user_id)
            if user is None:
                return False
            user.failed_logins = 0
            user.locked_until = None
        log.info("account_unlocked", user=str(user_id))
        return True

    async def resolve_session(self, token: str) -> Principal | None:
        """Identify the holder of a session cookie, or None."""
        if not token:
            return None
        token_hash = _hash_token(token)
        cached = self._cache_get(token_hash)
        if cached is not None:
            return cached

        async with session_scope(self._factory) as s:
            row = (
                await s.execute(
                    select(SessionRow, User)
                    .join(User, User.id == SessionRow.user_id)
                    .where(
                        SessionRow.token_hash == token_hash,
                        SessionRow.revoked_at.is_(None),
                        SessionRow.expires_at > _now(),
                    )
                )
            ).first()
            if row is None:
                return None
            session_row, user = row
            if user.status != "active":
                return None  # suspension takes effect on the next request
            session_row.last_seen_at = _now()
            principal = self._principal(user)

        self._cache_put(token_hash, principal)
        return principal

    async def logout(self, token: str) -> None:
        if not token:
            return
        token_hash = _hash_token(token)
        async with session_scope(self._factory) as s:
            await s.execute(
                update(SessionRow)
                .where(SessionRow.token_hash == token_hash)
                .values(revoked_at=_now())
            )
        self._cache_drop(token_hash)

    async def revoke_sessions(self, user_id: str, except_token: str | None = None) -> int:
        """Drop every session a user holds (suspension, password change, or an
        admin cutting someone off).

        `except_token` spares the caller's own session, which is what
        "sign out everywhere else" needs — without it the user would have to
        sign back in on the device they just used to secure the account.
        """
        owner = _maybe_uuid(user_id)
        if owner is None:
            return 0
        keep = _hash_token(except_token) if except_token else None
        where = [SessionRow.user_id == owner, SessionRow.revoked_at.is_(None)]
        if keep is not None:
            where.append(SessionRow.token_hash != keep)
        async with session_scope(self._factory) as s:
            hashes = (
                await s.execute(select(SessionRow.token_hash).where(*where))
            ).scalars().all()
            if hashes:
                await s.execute(
                    update(SessionRow).where(*where).values(revoked_at=_now())
                )
        for h in hashes:
            self._cache_drop(h)
        return len(hashes)

    async def list_sessions(self, user_id: str) -> list[SessionSummary]:
        """The user's live sessions, newest activity first.

        `ip` and `user_agent` have been recorded on every login since the first
        migration and read by nothing until now; showing them is what lets
        someone notice a session they don't recognise.
        """
        owner = _maybe_uuid(user_id)
        if owner is None:
            return []
        async with session_scope(self._factory) as s:
            rows = (
                await s.execute(
                    select(SessionRow)
                    .where(
                        SessionRow.user_id == owner,
                        SessionRow.revoked_at.is_(None),
                        SessionRow.expires_at > _now(),
                    )
                    .order_by(
                        SessionRow.last_seen_at.desc().nullslast(),
                        SessionRow.created_at.desc(),
                    )
                )
            ).scalars().all()
            return [
                SessionSummary(
                    id=str(r.id),
                    token_hash=r.token_hash,
                    created_at=r.created_at,
                    last_seen_at=r.last_seen_at,
                    ip=str(r.ip) if r.ip else None,
                    user_agent=r.user_agent,
                )
                for r in rows
            ]

    async def revoke_session(self, user_id: str, session_id: str) -> str | None:
        """Revoke one of *this user's* sessions; return its token hash, or None.

        Scoped by owner in the WHERE clause rather than checked afterwards, so
        another user's session id is indistinguishable from one that never
        existed — both return None and the router answers 404, matching how
        threads, jobs and files already behave.
        """
        sid = _maybe_uuid(session_id)
        owner = _maybe_uuid(user_id)
        if sid is None or owner is None:
            return None
        async with session_scope(self._factory) as s:
            row = (
                await s.execute(
                    select(SessionRow).where(
                        SessionRow.id == sid,
                        SessionRow.user_id == owner,
                        SessionRow.revoked_at.is_(None),
                    )
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            row.revoked_at = _now()
            token_hash = row.token_hash
        self._cache_drop(token_hash)
        return token_hash

    # -- passwords ------------------------------------------------------------
    async def request_password_reset(self, email: str) -> None:
        """Email a reset code, or do nothing — indistinguishably.

        Silent for unknown addresses and for accounts that cannot use a
        password anyway (pending, suspended, deleted). The caller says the same
        thing either way, so this cannot be used to test whether an address is
        registered.
        """
        email = normalize_email(email)
        code = None
        async with session_scope(self._factory) as s:
            user = await self._by_email(s, email)
            if user is not None and user.status == "active":
                code = await self._issue_otp(s, user, PURPOSE_RESET)
        if code is not None:
            await self._send_reset_code(email, code)

    async def reset_password(self, email: str, code: str, new_password: str) -> None:
        """Set a new password using an emailed code.

        Two transactions for the same reason `verify()` uses two: the attempt
        counter must be committed before the guess is judged, or a wrong code
        rolls the increment back and the six-digit space is brute-forceable at
        request speed.

        Deliberately does not open a session. Signing the user in here would
        mean a stolen code is a login; making them enter the new password once
        proves they know it and puts them through the normal, rate-limited path.
        """
        email = normalize_email(email)
        problem = validate_password(new_password)
        if problem:
            raise AccountError(problem, "weak_password")

        # 1. Charge the attempt. Commits whether or not the guess is right.
        async with session_scope(self._factory) as s:
            user = await self._by_email(s, email)
            if user is None or user.status != "active":
                raise AccountError("That code is not valid.", "invalid_code")
            otp = await self._latest_otp(s, user, PURPOSE_RESET)
            if otp is None:
                raise AccountError("Request a new code.", "no_code")
            if otp.expires_at <= _now():
                raise AccountError("That code has expired. Request a new one.", "expired_code")
            if otp.attempts >= self._auth.otp_max_attempts:
                raise AccountError(
                    "Too many attempts. Request a new code.", "too_many_attempts"
                )
            otp.attempts += 1
            otp_id = otp.id
            matched = secrets.compare_digest(otp.code_hash, _hash_token(code.strip()))

        if not matched:
            raise AccountError("That code is not valid.", "invalid_code")

        # 2. The code was right: consume it and set the password.
        async with session_scope(self._factory) as s:
            user = await self._by_email(s, email)
            if user is None or user.status != "active":
                raise AccountError("That code is not valid.", "invalid_code")
            otp = (
                await s.execute(
                    select(EmailOTP).where(
                        EmailOTP.id == otp_id, EmailOTP.consumed_at.is_(None)
                    )
                )
            ).scalar_one_or_none()
            if otp is None:  # raced with another reset using the same code
                raise AccountError("That code is not valid.", "invalid_code")
            otp.consumed_at = _now()
            user.password_hash = hash_password(new_password)
            user.password_changed_at = _now()
            # Whoever forced the reset may have been in the account already.
            user.failed_logins = 0
            user.locked_until = None
            user_id = str(user.id)

        # Outside the transaction: a reset is also the remedy for "someone else
        # is signed in as me", so every existing session has to go.
        await self.revoke_sessions(user_id)
        log.info("password_reset", user=user_id)

    async def change_password(
        self, user_id: str, current_password: str, new_password: str
    ) -> str:
        """Change a signed-in user's password; return a fresh session token.

        Every existing session is revoked, including the caller's, and a new one
        is opened. Rotating rather than sparing the current token means a
        session captured before the change cannot outlive it — the tab the user
        is looking at stays signed in only because it is handed a new cookie.
        """
        problem = validate_password(new_password)
        if problem:
            raise AccountError(problem, "weak_password")

        async with session_scope(self._factory) as s:
            user = await self._by_id(s, user_id)
            if user is None:
                raise AccountError("This account no longer exists.", "no_account")
            if not verify_password(user.password_hash, current_password):
                raise AccountError(
                    "That is not your current password.", "current_password_invalid"
                )
            if verify_password(user.password_hash, new_password):
                raise AccountError(
                    "The new password must be different from the current one.",
                    "password_unchanged",
                )
            user.password_hash = hash_password(new_password)
            user.password_changed_at = _now()

        await self.revoke_sessions(user_id)
        async with session_scope(self._factory) as s:
            user = await self._by_id(s, user_id)
            if user is None:  # pragma: no cover - deleted mid-change
                raise AccountError("This account no longer exists.", "no_account")
            token = await self._open_session(s, user)
        log.info("password_changed", user=str(user_id))
        return token

    # -- admin-driven account management --------------------------------------
    async def admin_create_user(self, email: str, role: str = "user") -> User:
        """Create an active account and email a code to set its password.

        The password is random and never shown to anyone: the invitee sets
        their own through the ordinary reset flow, so there is no shared
        secret in an inbox and no code path here that a normal reset does not
        already exercise. The account is created verified — an admin typing
        the address is the verification.
        """
        email = normalize_email(email)
        if not email or "@" not in email:
            raise AccountError("Enter a valid email address.", "invalid_email")
        if role not in ("user", "admin"):
            raise AccountError("Role must be 'user' or 'admin'.", "invalid_role")

        async with session_scope(self._factory) as s:
            if await self._by_email(s, email) is not None:
                raise AccountError("That address already has an account.", "already_exists")
            user = User(
                email=email,
                password_hash=hash_password(secrets.token_urlsafe(32)),
                tenant_id=_tenant_for(email),
                status="active",
                role=role,
                email_verified_at=_now(),
            )
            s.add(user)
            await s.flush()
            code = await self._issue_otp(s, user, PURPOSE_RESET)
            await s.refresh(user)
            s.expunge(user)
            log.info("account_invited", user=str(user.id), role=role)

        await self._send_invite(email, code)
        return user

    async def admin_force_reset(self, user_id: str) -> bool:
        """Cut a user off and email them a code to set a new password.

        Sessions and the reset code are handled in that order on purpose: the
        account is locked out first, so a compromised session cannot be used in
        the window between the two.
        """
        async with session_scope(self._factory) as s:
            user = await self._by_id(s, user_id)
            if user is None or user.status == "deleted":
                return False
            email = user.email

        await self.revoke_sessions(user_id)
        async with session_scope(self._factory) as s:
            user = await self._by_id(s, user_id)
            if user is None:  # pragma: no cover - deleted mid-reset
                return False
            code = await self._issue_otp(s, user, PURPOSE_RESET)
        await self._send_reset_code(email, code)
        log.info("password_reset_forced", user=str(user_id))
        return True

    # -- admin bootstrap ------------------------------------------------------
    async def promote_admin(self, email: str) -> bool:
        email = normalize_email(email)
        async with session_scope(self._factory) as s:
            user = await self._by_email(s, email)
            if user is None:
                return False
            if user.role != "admin":
                user.role = "admin"
                log.info("admin_promoted", email=email)
            # An admin bootstrapped from configuration should be usable without
            # a round trip through the inbox.
            if user.status == "pending":
                user.status = "active"
                user.email_verified_at = _now()
        await self._drop_user_session_cache(email)
        return True

    async def get_by_id(self, user_id: str) -> User | None:
        async with session_scope(self._factory) as s:
            return await self._by_id(s, user_id)

    async def get_by_email(self, email: str) -> User | None:
        async with session_scope(self._factory) as s:
            return await self._by_email(s, normalize_email(email))

    # -- internals ------------------------------------------------------------
    @staticmethod
    def _principal(user: User) -> Principal:
        return Principal(str(user.id), user.tenant_id, user.role, user.email)

    @staticmethod
    async def _by_id(s: AsyncSession, user_id: str) -> User | None:
        """Load a user by primary key, or None — including when the id is not a
        UUID at all, which is what a dev-mode identity looks like."""
        key = _maybe_uuid(user_id)
        if key is None:
            return None
        return (
            await s.execute(select(User).where(User.id == key))
        ).scalar_one_or_none()

    @staticmethod
    async def _by_email(s: AsyncSession, email: str) -> User | None:
        return (
            await s.execute(select(User).where(func.lower(User.email) == email))
        ).scalar_one_or_none()

    async def _issue_otp(
        self, s: AsyncSession, user: User, purpose: str = PURPOSE_VERIFY
    ) -> str:
        """Mint a code, replacing any outstanding one *of the same purpose*.

        Scoped by purpose in both directions. Without the filter, asking for a
        password reset would silently consume a pending signup code — the user
        would be told to check their inbox for a verification code that had
        just been invalidated by the reset they also asked for. Within one
        purpose the invalidation is deliberate: two live codes double the guess
        space for the same attempt budget.
        """
        await s.execute(
            update(EmailOTP)
            .where(
                EmailOTP.user_id == user.id,
                EmailOTP.purpose == purpose,
                EmailOTP.consumed_at.is_(None),
            )
            .values(consumed_at=_now())
        )
        code = _generate_code()
        s.add(
            EmailOTP(
                user_id=user.id,
                code_hash=_hash_token(code),
                purpose=purpose,
                expires_at=_now() + timedelta(minutes=self._auth.otp_ttl_minutes),
            )
        )
        return code

    @staticmethod
    async def _latest_otp(s: AsyncSession, user: User, purpose: str) -> EmailOTP | None:
        """The one live code for this purpose, locked for update.

        The `purpose` filter is load-bearing: without it a password-reset code
        would be accepted at /auth/verify and vice versa, which turns two
        separate proofs into one.
        """
        return (
            await s.execute(
                select(EmailOTP)
                .where(
                    EmailOTP.user_id == user.id,
                    EmailOTP.purpose == purpose,
                    EmailOTP.consumed_at.is_(None),
                )
                .order_by(EmailOTP.id.desc())
                .limit(1)
                .with_for_update()
            )
        ).scalar_one_or_none()

    async def _send_code(self, email: str, code: str) -> None:
        minutes = self._auth.otp_ttl_minutes
        await self._email.send(
            email,
            "Your verification code",
            f"Your verification code is {code}\n\n"
            f"It expires in {minutes} minutes. If you didn't request it, ignore this email.",
        )

    async def _send_reset_code(self, email: str, code: str) -> None:
        minutes = self._auth.otp_ttl_minutes
        await self._email.send(
            email,
            "Reset your password",
            f"Your password reset code is {code}\n\n"
            f"It expires in {minutes} minutes. If you didn't request it, ignore this "
            "email — your password has not changed.",
        )

    async def _send_invite(self, email: str, code: str) -> None:
        minutes = self._auth.otp_ttl_minutes
        await self._email.send(
            email,
            "Set your password",
            f"An account has been created for you.\n\n"
            f"Your setup code is {code}\n\n"
            f"It expires in {minutes} minutes. Use it on the 'forgot password' page "
            "to choose a password and sign in.",
        )

    async def _open_session(
        self, s: AsyncSession, user: User, ip: str | None = None, user_agent: str | None = None
    ) -> str:
        token = secrets.token_urlsafe(32)
        s.add(
            SessionRow(
                user_id=user.id,
                token_hash=_hash_token(token),
                expires_at=_now() + timedelta(days=self._auth.session_ttl_days),
                ip=ip,
                user_agent=(user_agent or "")[:400] or None,
            )
        )
        return token

    async def _drop_user_session_cache(self, email: str) -> None:
        """Role changes must not wait out the session cache."""
        if self._redis is None:
            return
        with contextlib.suppress(Exception):
            for key in self._redis.scan_iter(match=_SESSION_CACHE + "*", count=500):
                self._redis.delete(key)

    def _cache_get(self, token_hash: str) -> Principal | None:
        if self._redis is None:
            return None
        with contextlib.suppress(Exception):
            raw = self._redis.get(_SESSION_CACHE + token_hash)
            if raw:
                user_id, tenant_id, role, email = raw.split("|", 3)
                return Principal(user_id, tenant_id, role, email)
        return None

    def _cache_put(self, token_hash: str, principal: Principal) -> None:
        if self._redis is None:
            return
        with contextlib.suppress(Exception):
            self._redis.setex(
                _SESSION_CACHE + token_hash,
                _SESSION_CACHE_TTL,
                f"{principal.user_id}|{principal.tenant_id}|{principal.role}|{principal.email}",
            )

    def _cache_drop(self, token_hash: str) -> None:
        if self._redis is None:
            return
        with contextlib.suppress(Exception):
            self._redis.delete(_SESSION_CACHE + token_hash)


# A real Argon2 hash of a random string. Verifying against it makes a login for
# an unknown address cost the same as one for a known address.
_DUMMY_HASH = hash_password(secrets.token_urlsafe(16))
