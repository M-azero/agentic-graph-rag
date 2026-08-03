"""Self-serve session management: listing signed-in devices and cutting them off.

`sessions.ip` and `sessions.user_agent` have been written on every login since
the first migration and read by nothing. These tests cover the endpoints that
finally surface them, and the ownership scoping that keeps one user from
touching another's sessions.
"""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio

from graphrag.accounts import AccountService, PgKeyStore
from graphrag.api.app import create_app
from graphrag.container import Container
from tests.integration.conftest import relax_auth_limits, requires_db

pytestmark = [pytest.mark.integration, requires_db]

PASSWORD = "correct-horse-battery"
ALICE = "alice@example.com"
BOB = "bob@example.com"


def _app(db, email_sender):
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
    return _app(db, email_sender)


@pytest_asyncio.fixture
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _register(client, email_sender, email: str) -> None:
    await client.post("/auth/signup", json={"email": email, "password": PASSWORD})
    r = await client.post(
        "/auth/verify", json={"email": email, "code": email_sender.last_code(email)}
    )
    assert r.status_code == 200, r.text


async def _second_device(app, email: str, agent: str = "Mozilla/5.0 (Other Device)"):
    """A separate client with its own cookie jar — a second browser."""
    transport = httpx.ASGITransport(app=app)
    other = httpx.AsyncClient(transport=transport, base_url="http://test")
    r = await other.post(
        "/auth/login",
        json={"email": email, "password": PASSWORD},
        headers={"User-Agent": agent, "X-Real-IP": "203.0.113.9"},
    )
    assert r.status_code == 200, r.text
    return other


async def test_listing_marks_exactly_one_session_as_current(
    app, client, email_sender
):
    await _register(client, email_sender, ALICE)
    other = await _second_device(app, ALICE)
    try:
        body = (await client.get("/auth/sessions")).json()["sessions"]
        assert len(body) == 2
        assert [s["current"] for s in body].count(True) == 1
    finally:
        await other.aclose()


async def test_listing_surfaces_the_recorded_ip_and_agent(app, client, email_sender):
    """The columns exist and are written; this is the first thing that reads
    them, so an empty render would go unnoticed without a check."""
    await _register(client, email_sender, ALICE)
    other = await _second_device(app, ALICE, agent="Mozilla/5.0 (Test Runner)")
    try:
        rows = (await client.get("/auth/sessions")).json()["sessions"]
        remote = [s for s in rows if not s["current"]][0]
        assert remote["ip"] == "203.0.113.9"
        assert "Test Runner" in (remote["user_agent"] or "")
    finally:
        await other.aclose()


async def test_revoking_another_device_signs_it_out(app, client, email_sender):
    await _register(client, email_sender, ALICE)
    other = await _second_device(app, ALICE)
    try:
        assert (await other.get("/auth/me")).status_code == 200

        rows = (await client.get("/auth/sessions")).json()["sessions"]
        target = [s for s in rows if not s["current"]][0]
        assert (await client.delete(f"/auth/sessions/{target['id']}")).status_code == 200

        assert (await other.get("/auth/me")).status_code == 401
        # ...and the caller is still signed in.
        assert (await client.get("/auth/me")).status_code == 200
    finally:
        await other.aclose()


async def test_revoking_your_own_session_clears_the_cookie(client, email_sender):
    await _register(client, email_sender, ALICE)
    rows = (await client.get("/auth/sessions")).json()["sessions"]
    mine = [s for s in rows if s["current"]][0]

    r = await client.delete(f"/auth/sessions/{mine['id']}")
    assert r.status_code == 200
    assert (await client.get("/auth/me")).status_code == 401


async def test_another_users_session_is_a_404_not_a_403(app, client, email_sender):
    """404 rather than 403 throughout, so a session id cannot be probed for
    existence — the same rule threads, jobs and files already follow."""
    await _register(client, email_sender, ALICE)
    alice_session = (await client.get("/auth/sessions")).json()["sessions"][0]["id"]

    client.cookies.clear()
    await _register(client, email_sender, BOB)
    r = await client.delete(f"/auth/sessions/{alice_session}")
    assert r.status_code == 404


async def test_a_malformed_session_id_is_a_404_not_a_500(client, email_sender):
    await _register(client, email_sender, ALICE)
    assert (await client.delete("/auth/sessions/not-a-uuid")).status_code == 404


async def test_revoke_all_keeps_the_caller_signed_in(app, client, email_sender):
    """Otherwise "sign out everywhere else" would sign the user out of the very
    device they are using to secure the account."""
    await _register(client, email_sender, ALICE)
    first = await _second_device(app, ALICE)
    second = await _second_device(app, ALICE)
    try:
        r = await client.post("/auth/sessions/revoke-all")
        assert r.status_code == 200

        assert (await client.get("/auth/me")).status_code == 200
        assert (await first.get("/auth/me")).status_code == 401
        assert (await second.get("/auth/me")).status_code == 401

        remaining = (await client.get("/auth/sessions")).json()["sessions"]
        assert len(remaining) == 1
        assert remaining[0]["current"] is True
    finally:
        await first.aclose()
        await second.aclose()


async def test_sessions_require_authentication(client):
    assert (await client.get("/auth/sessions")).status_code == 401
    assert (await client.post("/auth/sessions/revoke-all")).status_code == 401
