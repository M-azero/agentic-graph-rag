"""The account lifecycle against real Postgres: signup, verification, login,
sessions, and API keys.

These are the security-critical paths — an error here is an account takeover,
not a bad answer — so they run against the real database rather than a stub.
"""

from __future__ import annotations

import pytest

from graphrag.accounts import AccountError, AccountService, PgKeyStore
from graphrag.config.settings import Settings
from tests.integration.conftest import requires_db

pytestmark = [pytest.mark.integration, requires_db]

EMAIL = "alice@example.com"
PASSWORD = "correct-horse-battery"


def _service(db, email_sender, **auth) -> AccountService:
    settings = Settings()
    settings.auth.enabled = True
    for key, value in auth.items():
        setattr(settings.auth, key, value)
    return AccountService(db, settings, email_sender)


async def _verified(db, email_sender, email: str = EMAIL):
    svc = _service(db, email_sender)
    await svc.signup(email, PASSWORD)
    principal, token = await svc.verify(email, email_sender.last_code(email))
    return svc, principal, token


# -- signup + verification ----------------------------------------------------

async def test_signup_emails_a_code_and_verification_activates(db, email_sender):
    svc = _service(db, email_sender)
    await svc.signup(EMAIL, PASSWORD)
    assert len(email_sender.sent) == 1

    principal, token = await svc.verify(EMAIL, email_sender.last_code(EMAIL))
    assert principal.email == EMAIL
    assert principal.role == "user"
    assert token
    assert await svc.resolve_session(token) is not None


async def test_unverified_account_cannot_log_in(db, email_sender):
    svc = _service(db, email_sender)
    await svc.signup(EMAIL, PASSWORD)
    with pytest.raises(AccountError) as exc:
        await svc.login(EMAIL, PASSWORD)
    assert exc.value.code == "email_unverified"


async def test_wrong_code_is_rejected(db, email_sender):
    svc = _service(db, email_sender)
    await svc.signup(EMAIL, PASSWORD)
    with pytest.raises(AccountError) as exc:
        await svc.verify(EMAIL, "000000")
    assert exc.value.code == "invalid_code"


async def test_codes_are_attempt_capped(db, email_sender):
    """Six digits is a million guesses — without a cap that is minutes of work."""
    svc = _service(db, email_sender, otp_max_attempts=3)
    await svc.signup(EMAIL, PASSWORD)
    for _ in range(3):
        with pytest.raises(AccountError):
            await svc.verify(EMAIL, "000000")

    # Even the *correct* code is refused once the budget is spent.
    with pytest.raises(AccountError) as exc:
        await svc.verify(EMAIL, email_sender.last_code(EMAIL))
    assert exc.value.code == "too_many_attempts"


async def test_expired_code_is_rejected(db, email_sender):
    svc = _service(db, email_sender, otp_ttl_minutes=0)
    await svc.signup(EMAIL, PASSWORD)
    with pytest.raises(AccountError) as exc:
        await svc.verify(EMAIL, email_sender.last_code(EMAIL))
    assert exc.value.code == "expired_code"


async def test_resending_invalidates_the_previous_code(db, email_sender):
    """Two live codes would double the guess space for the same attempt cap."""
    svc = _service(db, email_sender)
    await svc.signup(EMAIL, PASSWORD)
    first = email_sender.last_code(EMAIL)
    await svc.resend_code(EMAIL)
    second = email_sender.last_code(EMAIL)
    assert first != second

    with pytest.raises(AccountError):
        await svc.verify(EMAIL, first)
    principal, _ = await svc.verify(EMAIL, second)
    assert principal.email == EMAIL


async def test_a_code_cannot_be_reused(db, email_sender):
    svc, _principal, _token = await _verified(db, email_sender)
    with pytest.raises(AccountError):
        await svc.verify(EMAIL, email_sender.last_code(EMAIL))


async def test_signup_does_not_reveal_existing_accounts(db, email_sender):
    """Signing up twice must look identical to signing up once, or the endpoint
    tells an attacker which addresses are registered."""
    svc, _principal, _token = await _verified(db, email_sender)
    before = len(email_sender.sent)
    await svc.signup(EMAIL, "another-password-entirely")  # no raise
    assert len(email_sender.sent) == before  # and no code to the real owner

    # The original password still works — a second signup cannot overwrite it.
    principal, _ = await svc.login(EMAIL, PASSWORD)
    assert principal.email == EMAIL


async def test_weak_password_is_rejected(db, email_sender):
    svc = _service(db, email_sender)
    with pytest.raises(AccountError) as exc:
        await svc.signup(EMAIL, "short")
    assert exc.value.code == "weak_password"


async def test_each_account_gets_its_own_tenant_namespace(db, email_sender):
    _svc, alice, _ = await _verified(db, email_sender, "alice@example.com")
    _svc2, bob, _ = await _verified(db, email_sender, "bob@example.com")
    assert alice.tenant_id != bob.tenant_id
    # The namespace becomes a filename and a Cypher value, so it must stay a
    # safe token and must not simply be the address.
    assert alice.email not in alice.tenant_id
    assert all(c.isalnum() or c in "-_" for c in alice.tenant_id)


# -- login + sessions ---------------------------------------------------------

async def test_login_opens_a_resolvable_session(db, email_sender):
    svc, _principal, _token = await _verified(db, email_sender)
    principal, token = await svc.login(EMAIL, PASSWORD)
    resolved = await svc.resolve_session(token)
    assert resolved is not None and resolved.user_id == principal.user_id


async def test_wrong_password_is_rejected(db, email_sender):
    svc, _principal, _token = await _verified(db, email_sender)
    with pytest.raises(AccountError) as exc:
        await svc.login(EMAIL, "not-the-password")
    assert exc.value.code == "invalid_credentials"


async def test_unknown_address_looks_like_a_wrong_password(db, email_sender):
    svc = _service(db, email_sender)
    with pytest.raises(AccountError) as exc:
        await svc.login("nobody@example.com", PASSWORD)
    assert exc.value.code == "invalid_credentials"


async def test_logout_revokes_the_session(db, email_sender):
    svc, _principal, token = await _verified(db, email_sender)
    await svc.logout(token)
    assert await svc.resolve_session(token) is None


async def test_garbage_token_resolves_to_nobody(db, email_sender):
    svc = _service(db, email_sender)
    assert await svc.resolve_session("not-a-real-token") is None
    assert await svc.resolve_session("") is None


async def test_suspension_takes_effect_on_the_next_request(db, email_sender):
    """The reason sessions are server-side: revocation is immediate."""
    from sqlalchemy import update

    from graphrag.db.engine import session_scope
    from graphrag.db.models import User

    svc, principal, token = await _verified(db, email_sender)
    assert await svc.resolve_session(token) is not None

    async with session_scope(db) as s:
        await s.execute(
            update(User).where(User.email == EMAIL).values(status="suspended")
        )
    assert await svc.resolve_session(token) is None


async def test_revoke_sessions_cuts_off_every_device(db, email_sender):
    svc, principal, first = await _verified(db, email_sender)
    _p, second = await svc.login(EMAIL, PASSWORD)

    assert await svc.revoke_sessions(principal.user_id) == 2
    assert await svc.resolve_session(first) is None
    assert await svc.resolve_session(second) is None


# -- admin bootstrap ----------------------------------------------------------

async def test_promote_admin_grants_the_role(db, email_sender):
    svc, principal, _token = await _verified(db, email_sender)
    assert principal.role == "user"

    assert await svc.promote_admin(EMAIL) is True
    _p, token = await svc.login(EMAIL, PASSWORD)
    resolved = await svc.resolve_session(token)
    assert resolved is not None and resolved.is_admin


async def test_promoting_a_pending_account_also_activates_it(db, email_sender):
    """The bootstrap admin should be usable without a trip through the inbox."""
    svc = _service(db, email_sender)
    await svc.signup(EMAIL, PASSWORD)
    assert await svc.promote_admin(EMAIL) is True

    principal, _token = await svc.login(EMAIL, PASSWORD)
    assert principal.is_admin


async def test_promoting_an_unknown_address_reports_failure(db, email_sender):
    svc = _service(db, email_sender)
    assert await svc.promote_admin("ghost@example.com") is False


# -- API keys -----------------------------------------------------------------

async def test_api_key_resolves_to_its_owner(db, email_sender):
    _svc, principal, _token = await _verified(db, email_sender)
    store = PgKeyStore(db)

    _key_id, key = await store.create_key(principal.user_id, label="ci")
    assert key.startswith("grk_")

    owner = await store.resolve(key)
    assert owner is not None
    assert owner.user_id == principal.user_id
    assert owner.tenant_id == principal.tenant_id


async def test_unknown_key_resolves_to_nobody(db, email_sender):
    assert await PgKeyStore(db).resolve("grk_nonsense") is None


async def test_revoked_key_stops_working(db, email_sender):
    _svc, principal, _token = await _verified(db, email_sender)
    store = PgKeyStore(db)
    _key_id, key = await store.create_key(principal.user_id)

    assert await store.revoke_user(principal.user_id) == 1
    assert await store.resolve(key) is None


async def test_suspended_owner_disables_their_keys(db, email_sender):
    """Suspending an account must cut off programmatic access too, not just
    the browser session."""
    from sqlalchemy import update

    from graphrag.db.engine import session_scope
    from graphrag.db.models import User

    _svc, principal, _token = await _verified(db, email_sender)
    store = PgKeyStore(db)
    _key_id, key = await store.create_key(principal.user_id)
    assert await store.resolve(key) is not None

    async with session_scope(db) as s:
        await s.execute(
            update(User).where(User.email == EMAIL).values(status="suspended")
        )
    assert await store.resolve(key) is None


async def test_revoking_one_key_leaves_the_others(db, email_sender):
    _svc, principal, _token = await _verified(db, email_sender)
    store = PgKeyStore(db)
    _keep_id, keep = await store.create_key(principal.user_id, "keep")
    drop_id, drop = await store.create_key(principal.user_id, "drop")

    # The id `create_key` hands back is the row it actually inserted. It used to
    # be recovered afterwards by listing the user's keys newest-first, which
    # ties for two keys created in the same second — so the caller could revoke
    # a different key from the one whose plaintext it was holding.
    rows = await store.list_keys(principal.user_id)
    assert drop_id == next(r.id for r in rows if r.label == "drop")
    assert _keep_id == next(r.id for r in rows if r.label == "keep")

    assert await store.revoke_one(principal.user_id, drop_id) is True
    assert await store.resolve(drop) is None
    assert await store.resolve(keep) is not None


async def test_cannot_revoke_another_users_key(db, email_sender):
    _s1, alice, _t1 = await _verified(db, email_sender, "alice@example.com")
    _s2, bob, _t2 = await _verified(db, email_sender, "bob@example.com")
    store = PgKeyStore(db)

    alice_key_id, alice_key = await store.create_key(alice.user_id)

    assert await store.revoke_one(bob.user_id, alice_key_id) is False
    assert await store.resolve(alice_key) is not None
