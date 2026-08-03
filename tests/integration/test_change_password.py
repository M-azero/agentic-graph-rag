"""Changing your own password while signed in.

The interesting property is the session handling: every existing session dies,
including the caller's, and the caller is handed a new cookie. Sparing the
current token instead would let a session captured before the change outlive
the password it was obtained under.
"""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio

from graphrag.accounts import AccountService, PgKeyStore
from graphrag.api.app import create_app
from graphrag.api.deps import SESSION_COOKIE
from graphrag.container import Container
from tests.integration.conftest import relax_auth_limits, requires_db

pytestmark = [pytest.mark.integration, requires_db]

EMAIL = "alice@example.com"
PASSWORD = "correct-horse-battery"
NEW_PASSWORD = "an-entirely-different-one"


def _build(db, email_sender):
    container = Container()
    container.settings.auth.enabled = True
    container.settings.storage.vector.provider = "duckdb"
    relax_auth_limits(container)

    app = create_app(container)
    app.state.db = db
    app.state.accounts = AccountService(db, container.settings, email_sender)
    app.state.key_store = PgKeyStore(db)
    return app


@pytest_asyncio.fixture
async def app(db, email_sender):
    return _build(db, email_sender)


@pytest_asyncio.fixture
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _register(client, email_sender) -> None:
    await client.post("/auth/signup", json={"email": EMAIL, "password": PASSWORD})
    r = await client.post(
        "/auth/verify", json={"email": EMAIL, "code": email_sender.last_code(EMAIL)}
    )
    assert r.status_code == 200, r.text


def _change(client, current=PASSWORD, new=NEW_PASSWORD):
    return client.post(
        "/auth/change-password",
        json={"current_password": current, "new_password": new},
    )


async def test_the_password_is_replaced(client, email_sender):
    await _register(client, email_sender)
    assert (await _change(client)).status_code == 200

    client.cookies.clear()
    old = await client.post("/auth/login", json={"email": EMAIL, "password": PASSWORD})
    assert old.status_code == 401
    new = await client.post(
        "/auth/login", json={"email": EMAIL, "password": NEW_PASSWORD}
    )
    assert new.status_code == 200


async def test_the_wrong_current_password_is_refused(client, email_sender):
    await _register(client, email_sender)
    r = await _change(client, current="not-my-password")
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "current_password_invalid"


async def test_a_weak_new_password_is_refused(client, email_sender):
    await _register(client, email_sender)
    r = await _change(client, new="short")
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "weak_password"


async def test_reusing_the_same_password_is_refused(client, email_sender):
    """A no-op change would report success and revoke every session for
    nothing, which reads as a bug to the user."""
    await _register(client, email_sender)
    r = await _change(client, new=PASSWORD)
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "password_unchanged"


async def test_other_devices_are_signed_out(app, client, email_sender):
    await _register(client, email_sender)

    transport = httpx.ASGITransport(app=app)
    other = httpx.AsyncClient(transport=transport, base_url="http://test")
    try:
        r = await other.post("/auth/login", json={"email": EMAIL, "password": PASSWORD})
        assert r.status_code == 200
        assert (await other.get("/auth/me")).status_code == 200

        assert (await _change(client)).status_code == 200
        assert (await other.get("/auth/me")).status_code == 401
    finally:
        await other.aclose()


async def test_the_calling_tab_stays_signed_in_on_a_new_token(client, email_sender):
    """The caller's old token is revoked too — they keep working only because
    the response hands them a fresh cookie."""
    await _register(client, email_sender)
    before = client.cookies.get(SESSION_COOKIE)

    r = await _change(client)
    assert SESSION_COOKIE in r.headers.get("set-cookie", "")
    after = client.cookies.get(SESSION_COOKIE)
    assert after and after != before

    assert (await client.get("/auth/me")).status_code == 200


async def test_change_password_requires_authentication(client):
    r = await client.post(
        "/auth/change-password",
        json={"current_password": PASSWORD, "new_password": NEW_PASSWORD},
    )
    assert r.status_code == 401
