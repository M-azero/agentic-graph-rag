"""Login lockout, against real Postgres.

The load-bearing case is `test_the_counter_survives_a_fresh_session`. `login()`
rejects a bad password by raising, and it does so from inside a transaction that
also just incremented the failure counter — so the naive implementation rolls
the increment back with the exception, the count never rises, and the lockout is
decorative. Nothing about that is visible from a single request; only reading
the row back in a *different* session catches it.
"""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import select

from graphrag.accounts import AccountService, PgKeyStore
from graphrag.api.app import create_app
from graphrag.container import Container
from graphrag.db.models import User
from tests.integration.conftest import relax_auth_limits, requires_db

pytestmark = [pytest.mark.integration, requires_db]

EMAIL = "alice@example.com"
PASSWORD = "correct-horse-battery"
THRESHOLD = 3


@pytest_asyncio.fixture
async def client(db, email_sender):
    container = Container()
    container.settings.auth.enabled = True
    container.settings.storage.vector.provider = "duckdb"
    relax_auth_limits(container)
    # Lower than the shipped 10 so a test does not need ten Argon2 verifications
    # (~0.1s each) to reach the interesting state.
    container.settings.auth.lockout_threshold = THRESHOLD
    container.settings.auth.lockout_base_seconds = 60

    app = create_app(container)
    app.state.db = db
    app.state.accounts = AccountService(db, container.settings, email_sender)
    app.state.key_store = PgKeyStore(db)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _register(client, email_sender) -> None:
    await client.post("/auth/signup", json={"email": EMAIL, "password": PASSWORD})
    r = await client.post(
        "/auth/verify", json={"email": EMAIL, "code": email_sender.last_code(EMAIL)}
    )
    assert r.status_code == 200, r.text
    client.cookies.clear()


async def _bad_login(client):
    return await client.post(
        "/auth/login", json={"email": EMAIL, "password": "not-the-password"}
    )


async def _user(db) -> User:
    async with db() as s:
        return (
            await s.execute(select(User).where(User.email == EMAIL))
        ).scalar_one()


async def test_the_counter_survives_a_fresh_session(client, email_sender, db):
    """The regression test for the rollback trap described in the module
    docstring: read the count back through a session that was not involved in
    recording it."""
    await _register(client, email_sender)

    for _ in range(2):
        assert (await _bad_login(client)).status_code == 401

    assert (await _user(db)).failed_logins == 2


async def test_the_account_locks_at_the_threshold(client, email_sender, db):
    await _register(client, email_sender)
    for _ in range(THRESHOLD):
        assert (await _bad_login(client)).status_code == 401

    locked = await _bad_login(client)
    assert locked.status_code == 429
    detail = locked.json()["detail"]
    assert detail["code"] == "rate_limited"
    assert detail["retry_after"] > 0
    assert locked.headers["retry-after"]
    assert (await _user(db)).locked_until is not None


async def test_a_locked_account_refuses_even_the_right_password(client, email_sender):
    """Otherwise the lock stops nothing — the attacker's next guess is the
    correct one and it would be accepted."""
    await _register(client, email_sender)
    for _ in range(THRESHOLD):
        await _bad_login(client)

    r = await client.post("/auth/login", json={"email": EMAIL, "password": PASSWORD})
    assert r.status_code == 429


async def test_a_successful_login_clears_the_count(client, email_sender, db):
    """The lock exists to slow guessing, not to punish someone who eventually
    remembered their password."""
    await _register(client, email_sender)
    for _ in range(THRESHOLD - 1):
        await _bad_login(client)
    assert (await _user(db)).failed_logins == THRESHOLD - 1

    ok = await client.post("/auth/login", json={"email": EMAIL, "password": PASSWORD})
    assert ok.status_code == 200
    row = await _user(db)
    assert row.failed_logins == 0
    assert row.locked_until is None


async def test_unlock_restores_access(client, email_sender, db):
    """Without this the only remedy is waiting out a backoff that reaches an
    hour; it is what `graphrag unlock` and the admin button both call."""
    await _register(client, email_sender)
    for _ in range(THRESHOLD):
        await _bad_login(client)
    assert (await _bad_login(client)).status_code == 429

    user = await _user(db)
    accounts = AccountService(db, Container().settings, None)
    assert await accounts.unlock(str(user.id)) is True

    r = await client.post("/auth/login", json={"email": EMAIL, "password": PASSWORD})
    assert r.status_code == 200


async def test_unlocking_an_unknown_user_is_false_not_an_error(db):
    import uuid

    accounts = AccountService(db, Container().settings, None)
    assert await accounts.unlock(str(uuid.uuid4())) is False
    # A dev-mode identity is not a UUID at all and must not raise.
    assert await accounts.unlock("default") is False


async def test_an_unknown_address_never_locks_anything(client, db):
    """There is no row to count against, and inventing one would turn the login
    endpoint into a way to create accounts."""
    for _ in range(THRESHOLD + 2):
        r = await client.post(
            "/auth/login", json={"email": "nobody@example.com", "password": "guessing"}
        )
        assert r.status_code == 401

    async with db() as s:
        assert (await s.execute(select(User))).scalars().all() == []


async def test_lockout_can_be_disabled(db, email_sender):
    """`lockout_threshold: 0` is a documented setting, so it has to actually
    switch the behaviour off rather than lock on the first failure."""
    container = Container()
    container.settings.auth.enabled = True
    container.settings.storage.vector.provider = "duckdb"
    relax_auth_limits(container)
    container.settings.auth.lockout_threshold = 0

    app = create_app(container)
    app.state.db = db
    app.state.accounts = AccountService(db, container.settings, email_sender)
    app.state.key_store = PgKeyStore(db)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        await _register(c, email_sender)
        for _ in range(5):
            assert (await _bad_login(c)).status_code == 401
        assert (await _user(db)).failed_logins == 0
