"""Sign-up, email verification, login, passwords, sessions and API keys.

The session cookie is httpOnly (JavaScript cannot read it, so an XSS bug cannot
exfiltrate it) and SameSite=Lax (the browser withholds it from cross-site POSTs,
which is the CSRF defence for the mutating endpoints here).

Every endpoint reachable without credentials carries an `auth_throttle`
dependency, keyed on the caller's address. Without that, the six-digit codes
and the password field are brute-forceable at network speed whatever the
per-attempt caps say — the caps bound guesses per *code*, not per second.
`tests/unit/test_api_exposure.py` asserts the dependency is present on each of
them, so a new endpoint here cannot quietly ship without one.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from graphrag.accounts import AccountError, AccountService, hash_session_token
from graphrag.agent.presets import preset_options
from graphrag.api.deps import (
    SESSION_COOKIE,
    AuthUser,
    get_accounts,
    get_container,
    get_current_user,
    get_db,
    get_key_store,
)
from graphrag.api.ratelimit import auth_throttle, client_ip, too_many_requests
from graphrag.api.schemas import (
    Acknowledged,
    APIKeyCreate,
    APIKeyCreated,
    APIKeyInfo,
    APIKeyList,
    ChangePasswordRequest,
    EmailRequest,
    ForgotPasswordRequest,
    LimitsInfo,
    LoginRequest,
    Me,
    ModelOption,
    PresetOption,
    ResetPasswordRequest,
    SessionInfo,
    SessionList,
    SignupRequest,
    VerifyRequest,
)
from graphrag.container import Container
from graphrag.core.logging import get_logger
from graphrag.limits import get_limits
from graphrag.llm.registry import allowed_models, resolve_model

router = APIRouter(prefix="/auth", tags=["auth"])
log = get_logger(__name__)

# Deliberately identical for known and unknown addresses — anything more
# specific turns these endpoints into an account-enumeration oracle.
_SENT = "If that address can be registered, we've sent a code to it."
_RESET_SENT = "If that address has an account, we've sent a reset code to it."


def _is_secure(request: Request) -> bool:
    """Did this request arrive over TLS? Trusts X-Forwarded-Proto, which the
    bundled Caddy sets — without it every request looks like plain HTTP from
    behind a proxy."""
    forwarded = request.headers.get("X-Forwarded-Proto", "").split(",")[0].strip()
    return (forwarded or request.url.scheme) == "https"


def _set_session_cookie(
    response: Response, token: str, container: Container, request: Request
) -> None:
    auth = container.settings.auth
    configured = str(auth.cookie_secure).lower()
    secure = _is_secure(request) if configured == "auto" else configured in ("1", "true", "yes")
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=auth.session_ttl_days * 86400,
        httponly=True,
        secure=secure,
        samesite="lax",
        path="/",
    )


def _require_accounts(accounts: AccountService | None) -> AccountService:
    if accounts is None or not accounts.available:
        raise HTTPException(
            status_code=503,
            detail="Accounts are unavailable: the server has no database configured.",
        )
    return accounts


def _account_error(exc: AccountError, status: int) -> HTTPException:
    """Turn a service error into an HTTP one, carrying its code through.

    `too_many_attempts` from a lockout also carries `retry_after`, in the same
    shape as the rate-limit and quota 429s, so the UI has one branch for
    "come back later" rather than three.
    """
    detail: dict[str, object] = {"code": exc.code, "message": str(exc)}
    if status == 429:
        return too_many_requests(exc.retry_after or 60, str(exc))
    return HTTPException(status_code=status, detail=detail)


@router.post(
    "/signup",
    response_model=Acknowledged,
    dependencies=[Depends(auth_throttle("signup"))],
)
async def signup(
    payload: SignupRequest,
    request: Request,
    accounts: AccountService = Depends(get_accounts),
    container: Container = Depends(get_container),
) -> Acknowledged:
    accounts = _require_accounts(accounts)
    if not container.settings.auth.open_registration:
        raise HTTPException(status_code=403, detail="Registration is by invitation only.")
    try:
        await accounts.signup(payload.email, payload.password)
    except AccountError as exc:
        # Validation problems are the user's to fix, so they are reported
        # plainly; "already registered" is never among them.
        raise HTTPException(
            status_code=400, detail={"code": exc.code, "message": str(exc)}
        ) from None
    return Acknowledged(message=_SENT)


@router.post(
    "/resend",
    response_model=Acknowledged,
    dependencies=[Depends(auth_throttle("resend"))],
)
async def resend(
    payload: EmailRequest,
    request: Request,
    accounts: AccountService = Depends(get_accounts),
) -> Acknowledged:
    await _require_accounts(accounts).resend_code(payload.email)
    return Acknowledged(message=_SENT)


@router.post(
    "/verify",
    response_model=Me,
    dependencies=[Depends(auth_throttle("verify"))],
)
async def verify(
    payload: VerifyRequest,
    request: Request,
    response: Response,
    accounts: AccountService = Depends(get_accounts),
    container: Container = Depends(get_container),
) -> Me:
    try:
        principal, token = await _require_accounts(accounts).verify(payload.email, payload.code)
    except AccountError as exc:
        raise HTTPException(
            status_code=400, detail={"code": exc.code, "message": str(exc)}
        ) from None
    _set_session_cookie(response, token, container, request)
    return _me(principal, container, _enabled_models(request))


@router.post(
    "/login",
    response_model=Me,
    dependencies=[Depends(auth_throttle("login"))],
)
async def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    accounts: AccountService = Depends(get_accounts),
    container: Container = Depends(get_container),
) -> Me:
    try:
        principal, token = await _require_accounts(accounts).login(
            payload.email,
            payload.password,
            ip=client_ip(request),
            user_agent=request.headers.get("User-Agent"),
        )
    except AccountError as exc:
        # 403 for an unverified account so the UI can route to /verify;
        # 429 when the account is locked, so the UI can say how long;
        # 401 for bad credentials.
        if exc.code == "too_many_attempts":
            status = 429
        elif exc.code in ("email_unverified", "account_inactive"):
            status = 403
        else:
            status = 401
        raise _account_error(exc, status) from None
    _set_session_cookie(response, token, container, request)
    return _me(principal, container, _enabled_models(request))


# -- passwords ----------------------------------------------------------------

@router.post(
    "/forgot-password",
    response_model=Acknowledged,
    dependencies=[Depends(auth_throttle("reset"))],
)
async def forgot_password(
    payload: ForgotPasswordRequest,
    request: Request,
    accounts: AccountService = Depends(get_accounts),
) -> Acknowledged:
    """Start a password reset. Answers identically for every address."""
    await _require_accounts(accounts).request_password_reset(payload.email)
    return Acknowledged(message=_RESET_SENT)


@router.post(
    "/reset-password",
    response_model=Acknowledged,
    dependencies=[Depends(auth_throttle("reset"))],
)
async def reset_password(
    payload: ResetPasswordRequest,
    request: Request,
    accounts: AccountService = Depends(get_accounts),
) -> Acknowledged:
    """Finish a password reset.

    No session is opened: the user signs in with the new password through the
    ordinary rate-limited path, so a stolen code alone is not a login.
    """
    try:
        await _require_accounts(accounts).reset_password(
            payload.email, payload.code, payload.password
        )
    except AccountError as exc:
        raise _account_error(exc, 400) from None
    return Acknowledged(message="Password updated. You can sign in now.")


@router.post("/change-password", response_model=Acknowledged)
async def change_password(
    payload: ChangePasswordRequest,
    request: Request,
    response: Response,
    user: AuthUser = Depends(get_current_user),
    accounts: AccountService = Depends(get_accounts),
    container: Container = Depends(get_container),
) -> Acknowledged:
    """Change your own password, signing out every device including this one.

    The caller is handed a fresh cookie, so the tab they are looking at stays
    usable — but the token it held before the change is dead, which is the
    point: a session captured earlier cannot outlive the password it was
    obtained under.
    """
    try:
        token = await _require_accounts(accounts).change_password(
            user.user_id, payload.current_password, payload.new_password
        )
    except AccountError as exc:
        raise _account_error(exc, 400) from None
    _set_session_cookie(response, token, container, request)
    return Acknowledged(message="Password changed. Signed out on every other device.")


# -- sessions -----------------------------------------------------------------

@router.get("/sessions", response_model=SessionList)
async def list_sessions(
    request: Request,
    user: AuthUser = Depends(get_current_user),
    accounts: AccountService = Depends(get_accounts),
) -> SessionList:
    """Signed-in devices, newest activity first."""
    cookie = request.cookies.get(SESSION_COOKIE)
    current_hash = hash_session_token(cookie) if cookie else None
    rows = await _require_accounts(accounts).list_sessions(user.user_id)
    return SessionList(
        sessions=[
            SessionInfo(
                id=r.id,
                created_at=r.created_at.isoformat() if r.created_at else "",
                last_seen_at=r.last_seen_at.isoformat() if r.last_seen_at else None,
                ip=r.ip,
                user_agent=r.user_agent,
                current=r.token_hash == current_hash,
            )
            for r in rows
        ]
    )


@router.delete("/sessions/{session_id}", response_model=Acknowledged)
async def revoke_session(
    session_id: str,
    request: Request,
    response: Response,
    user: AuthUser = Depends(get_current_user),
    accounts: AccountService = Depends(get_accounts),
) -> Acknowledged:
    """Sign one device out. 404 for a session that isn't yours — the same
    answer as one that does not exist, so this cannot enumerate sessions."""
    revoked = await _require_accounts(accounts).revoke_session(user.user_id, session_id)
    if revoked is None:
        raise HTTPException(status_code=404, detail="No such session.")
    cookie = request.cookies.get(SESSION_COOKIE)
    if cookie and hash_session_token(cookie) == revoked:
        response.delete_cookie(SESSION_COOKIE, path="/")
    return Acknowledged(message="Signed out.")


@router.post("/sessions/revoke-all", response_model=Acknowledged)
async def revoke_other_sessions(
    request: Request,
    user: AuthUser = Depends(get_current_user),
    accounts: AccountService = Depends(get_accounts),
) -> Acknowledged:
    """Sign out everywhere else, keeping this device signed in."""
    cookie = request.cookies.get(SESSION_COOKIE)
    count = await _require_accounts(accounts).revoke_sessions(
        user.user_id, except_token=cookie
    )
    return Acknowledged(message=f"Signed out {count} other device(s).")


@router.post("/logout", response_model=Acknowledged)
async def logout(
    request: Request,
    response: Response,
    accounts: AccountService = Depends(get_accounts),
) -> Acknowledged:
    token = request.cookies.get(SESSION_COOKIE)
    if token and accounts is not None:
        await accounts.logout(token)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return Acknowledged(message="Signed out.")


@router.get("/me", response_model=Me)
async def me(
    request: Request,
    user: AuthUser = Depends(get_current_user),
    container: Container = Depends(get_container),
) -> Me:
    """Who am I, and what can I choose? The UI calls this on load."""
    enabled = _enabled_models(request)
    return Me(
        user_id=user.user_id,
        email=user.email,
        role=user.role,
        tenant_id=user.tenant_id,
        models=_model_options(container, enabled),
        default_model=resolve_model(None, container.settings, enabled).model,
        presets=_preset_options(),
    )


@router.get("/limits", response_model=LimitsInfo)
async def my_limits(
    user: AuthUser = Depends(get_current_user),
    limits=Depends(get_limits),
    db=Depends(get_db),
) -> LimitsInfo:
    """Allowances and consumption, for the account page's meters."""
    effective = await limits.effective(user.user_id)
    info = LimitsInfo(
        limits=effective.as_dict(), usage=await limits.usage_snapshot(user.user_id)
    )
    if db is None:
        return info

    import uuid as _uuid

    from sqlalchemy import func, select

    from graphrag.db.engine import session_scope
    from graphrag.db.models import File, Thread

    try:
        owner = _uuid.UUID(str(user.user_id))
    except (ValueError, AttributeError, TypeError):
        return info

    async with session_scope(db) as s:
        files, stored = (
            await s.execute(
                select(func.count(), func.coalesce(func.sum(File.size_bytes), 0))
                .where(File.user_id == owner)
            )
        ).one()
        threads = (
            await s.execute(
                select(func.count()).select_from(Thread).where(
                    Thread.user_id == owner, Thread.deleted_at.is_(None)
                )
            )
        ).scalar_one()
    info.files_used = int(files)
    info.storage_used_mb = round(int(stored) / (1024 * 1024), 2)
    info.threads_used = int(threads)
    return info


# -- personal API keys --------------------------------------------------------

@router.get("/keys", response_model=APIKeyList)
async def list_keys(
    user: AuthUser = Depends(get_current_user),
    key_store=Depends(get_key_store),
) -> APIKeyList:
    rows = await key_store.list_keys(user.user_id)
    return APIKeyList(
        keys=[
            APIKeyInfo(
                id=k.id,
                label=k.label,
                created_at=k.created_at.isoformat() if k.created_at else "",
                last_used_at=k.last_used_at.isoformat() if k.last_used_at else None,
            )
            for k in rows
        ]
    )


@router.post("/keys", response_model=APIKeyCreated)
async def create_key(
    payload: APIKeyCreate,
    user: AuthUser = Depends(get_current_user),
    key_store=Depends(get_key_store),
) -> APIKeyCreated:
    """Mint a key. The plaintext is returned exactly once — only its hash is
    stored, so a lost key is replaced, never recovered."""
    key_id, key = await key_store.create_key(user.user_id, payload.label)
    return APIKeyCreated(id=key_id, api_key=key)


@router.delete("/keys/{key_id}", response_model=Acknowledged)
async def revoke_key(
    key_id: int,
    user: AuthUser = Depends(get_current_user),
    key_store=Depends(get_key_store),
) -> Acknowledged:
    if not await key_store.revoke_one(user.user_id, key_id):
        raise HTTPException(status_code=404, detail="No such key.")
    return Acknowledged(message="Key revoked.")


# -- helpers ------------------------------------------------------------------

def _enabled_models(request: Request) -> list[str] | None:
    """The admin's narrowing of the model list, or None when they set none."""
    return getattr(request.app.state, "enabled_models", None)


def _model_options(
    container: Container, enabled: list[str] | None = None
) -> list[ModelOption]:
    return [
        ModelOption(model=m.model, label=m.label or m.model, provider=m.provider)
        for m in allowed_models(container.settings, enabled)
    ]


def _preset_options() -> list[PresetOption]:
    """The job presets the composer offers. Shipped on `/auth/me` alongside the
    models so the UI has everything it needs to render the picker from one
    call, and never carries its own copy of the list."""
    return [
        PresetOption(
            id=str(p.id), label=p.label, emoji=p.emoji, description=p.description
        )
        for p in preset_options()
    ]


def _me(principal, container: Container, enabled: list[str] | None = None) -> Me:
    return Me(
        user_id=principal.user_id,
        email=principal.email,
        role=principal.role,
        tenant_id=principal.tenant_id,
        models=_model_options(container, enabled),
        default_model=resolve_model(None, container.settings, enabled).model,
        presets=_preset_options(),
    )
