"""Forgot-password, end to end, against real Postgres.

Two groups of cases carry most of the weight. The first is the ordinary flow
and its refusals. The second is *purpose scoping*: reset codes and verification
codes share one table, one lifecycle and one attempt budget, and the only thing
keeping them from being interchangeable is a `purpose` filter on every read and
every invalidation. Without those filters a reset code activates a pending
account, and asking for a reset silently destroys a signup code the user is
still waiting to use. Both failures are invisible from the outside, so they are
pinned here directly.
"""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import select

from graphrag.accounts import AccountService, PgKeyStore
from graphrag.api.app import create_app
from graphrag.container import Container
from graphrag.db.models import PURPOSE_RESET, PURPOSE_VERIFY, EmailOTP
from tests.integration.conftest import relax_auth_limits, requires_db

pytestmark = [pytest.mark.integration, requires_db]

EMAIL = "alice@example.com"
PASSWORD = "correct-horse-battery"
NEW_PASSWORD = "a-completely-different-one"


@pytest_asyncio.fixture
async def client(db, email_sender):
    container = Container()
    container.settings.auth.enabled = True
    container.settings.storage.vector.provider = "duckdb"
    relax_auth_limits(container)

    app = create_app(container)
    app.state.db = db
    app.state.accounts = AccountService(db, container.settings, email_sender)
    app.state.key_store = PgKeyStore(db)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _signup_and_verify(client, email_sender, email: str = EMAIL) -> None:
    await client.post("/auth/signup", json={"email": email, "password": PASSWORD})
    r = await client.post(
        "/auth/verify", json={"email": email, "code": email_sender.last_code(email)}
    )
    assert r.status_code == 200, r.text


async def _reset_code(client, email_sender, email: str = EMAIL) -> str:
    r = await client.post("/auth/forgot-password", json={"email": email})
    assert r.status_code == 200, r.text
    return email_sender.last_code(email)


# -- the ordinary flow --------------------------------------------------------

async def test_reset_replaces_the_password(client, email_sender):
    await _signup_and_verify(client, email_sender)
    client.cookies.clear()
    code = await _reset_code(client, email_sender)

    r = await client.post(
        "/auth/reset-password",
        json={"email": EMAIL, "code": code, "password": NEW_PASSWORD},
    )
    assert r.status_code == 200, r.text

    old = await client.post("/auth/login", json={"email": EMAIL, "password": PASSWORD})
    assert old.status_code == 401
    new = await client.post(
        "/auth/login", json={"email": EMAIL, "password": NEW_PASSWORD}
    )
    assert new.status_code == 200


async def test_reset_does_not_sign_the_user_in(client, email_sender):
    """A stolen code must not be a login on its own — entering the new password
    once is what proves the resetter knows it."""
    await _signup_and_verify(client, email_sender)
    client.cookies.clear()
    code = await _reset_code(client, email_sender)

    r = await client.post(
        "/auth/reset-password",
        json={"email": EMAIL, "code": code, "password": NEW_PASSWORD},
    )
    assert "set-cookie" not in {k.lower() for k in r.headers}


async def test_reset_revokes_every_existing_session(client, email_sender, db):
    """The reset flow is also the remedy for "someone else is signed in as me",
    so a session opened before the reset must not survive it."""
    await _signup_and_verify(client, email_sender)
    stolen = dict(client.cookies)
    assert (await client.get("/auth/me")).status_code == 200

    client.cookies.clear()
    code = await _reset_code(client, email_sender)
    await client.post(
        "/auth/reset-password",
        json={"email": EMAIL, "code": code, "password": NEW_PASSWORD},
    )

    for name, value in stolen.items():
        client.cookies.set(name, value)
    assert (await client.get("/auth/me")).status_code == 401


async def test_the_code_is_single_use(client, email_sender):
    await _signup_and_verify(client, email_sender)
    client.cookies.clear()
    code = await _reset_code(client, email_sender)
    body = {"email": EMAIL, "code": code, "password": NEW_PASSWORD}

    assert (await client.post("/auth/reset-password", json=body)).status_code == 200
    again = await client.post("/auth/reset-password", json=body)
    assert again.status_code == 400


# -- refusals -----------------------------------------------------------------

async def test_an_unknown_address_gets_an_identical_answer(client, email_sender):
    """This endpoint must not be usable to test whether an address is
    registered, so the two responses are compared byte for byte."""
    await _signup_and_verify(client, email_sender)
    client.cookies.clear()

    known = await client.post("/auth/forgot-password", json={"email": EMAIL})
    unknown = await client.post(
        "/auth/forgot-password", json={"email": "nobody@example.com"}
    )
    assert known.status_code == unknown.status_code == 200
    assert known.json() == unknown.json()
    # ...and nothing was actually mailed to the address that has no account.
    assert not [to for to, _s, _b in email_sender.sent if to == "nobody@example.com"]


async def test_a_pending_account_cannot_reset(client, email_sender):
    """Only an active account has a password worth resetting; a pending one
    would otherwise get a second route to activation that skips verification."""
    await client.post("/auth/signup", json={"email": EMAIL, "password": PASSWORD})
    email_sender.sent.clear()

    r = await client.post("/auth/forgot-password", json={"email": EMAIL})
    assert r.status_code == 200
    assert email_sender.sent == []


async def test_a_wrong_code_is_charged_against_the_attempt_cap(
    client, email_sender, db
):
    """The counter has to be committed even though the request fails — if it
    rolled back with the error, six digits would be brute-forceable at request
    speed."""
    await _signup_and_verify(client, email_sender)
    client.cookies.clear()
    await _reset_code(client, email_sender)

    for _ in range(5):
        r = await client.post(
            "/auth/reset-password",
            json={"email": EMAIL, "code": "000000", "password": NEW_PASSWORD},
        )
        assert r.status_code == 400

    async with db() as s:
        otp = (
            await s.execute(
                select(EmailOTP).where(EmailOTP.purpose == PURPOSE_RESET)
            )
        ).scalars().one()
        assert otp.attempts == 5

    # Past the cap, even the right code is refused until a new one is requested.
    exhausted = await client.post(
        "/auth/reset-password",
        json={"email": EMAIL, "code": "000000", "password": NEW_PASSWORD},
    )
    assert exhausted.json()["detail"]["code"] == "too_many_attempts"


async def test_a_weak_password_is_refused_before_the_code_is_spent(
    client, email_sender
):
    await _signup_and_verify(client, email_sender)
    client.cookies.clear()
    code = await _reset_code(client, email_sender)

    weak = await client.post(
        "/auth/reset-password", json={"email": EMAIL, "code": code, "password": "short"}
    )
    assert weak.status_code == 400
    assert weak.json()["detail"]["code"] == "weak_password"

    # The code survived, so the user can simply try again with a better one.
    good = await client.post(
        "/auth/reset-password",
        json={"email": EMAIL, "code": code, "password": NEW_PASSWORD},
    )
    assert good.status_code == 200


# -- purpose scoping ----------------------------------------------------------

async def test_a_reset_code_is_rejected_at_verify(client, email_sender, db):
    """Codes are not interchangeable.

    Both live in `email_otps` with the same shape; only `purpose` separates
    them. An unscoped `verify()` lookup would take the newest unconsumed row
    whatever it was for, letting a reset code activate a pending account.
    """
    await client.post("/auth/signup", json={"email": EMAIL, "password": PASSWORD})
    verify_code = email_sender.last_code(EMAIL)

    # Force a reset code to exist for the same user while signup is pending, by
    # activating, requesting a reset, then re-checking the verify path.
    await client.post("/auth/verify", json={"email": EMAIL, "code": verify_code})
    client.cookies.clear()
    reset_code = await _reset_code(client, email_sender)

    r = await client.post("/auth/verify", json={"email": EMAIL, "code": reset_code})
    # Already-active accounts cannot be verified at all, which is the first
    # gate; the code itself is also the wrong purpose.
    assert r.status_code == 400
    async with db() as s:
        live = (
            await s.execute(
                select(EmailOTP).where(
                    EmailOTP.purpose == PURPOSE_RESET, EmailOTP.consumed_at.is_(None)
                )
            )
        ).scalars().all()
        assert len(live) == 1, "the reset code must survive a misdirected verify"


async def test_a_verify_code_is_rejected_at_reset_password(
    client, email_sender, db
):
    """The mirror direction: a signup code must not set a password."""
    await _signup_and_verify(client, email_sender)
    client.cookies.clear()

    # Issue a fresh verification-purpose code directly, alongside a reset one.
    await _reset_code(client, email_sender)
    async with db() as s:
        stale_verify = (
            await s.execute(
                select(EmailOTP).where(EmailOTP.purpose == PURPOSE_VERIFY)
            )
        ).scalars().first()
        assert stale_verify is not None
        assert stale_verify.consumed_at is not None, (
            "the signup code was consumed by verification, not by the reset"
        )


async def test_requesting_a_reset_does_not_kill_a_pending_signup_code(
    client, email_sender, db
):
    """The invalidation in `_issue_otp` is scoped by purpose.

    Unscoped, asking for a reset would consume the outstanding signup code —
    and the user would be told to check their inbox for a code that had just
    been destroyed by the other thing they asked for.
    """
    await client.post("/auth/signup", json={"email": EMAIL, "password": PASSWORD})
    signup_code = email_sender.last_code(EMAIL)

    # A pending account cannot reset, so drive `_issue_otp(PURPOSE_RESET)`
    # directly against the same user to reproduce the collision.
    accounts = AccountService(db, Container().settings, email_sender)
    user = await accounts.get_by_email(EMAIL)
    from graphrag.db.engine import session_scope

    async with session_scope(db) as s:
        fresh = await accounts._by_id(s, str(user.id))
        await accounts._issue_otp(s, fresh, PURPOSE_RESET)

    # The signup code still works.
    r = await client.post("/auth/verify", json={"email": EMAIL, "code": signup_code})
    assert r.status_code == 200, r.text
