"""The HTTP surface of authentication: cookies, status codes, and which routes
are actually protected.

The service-level tests cover the state machine; these cover the wiring — that
the cookie is set with the right flags, that 401/403 mean what the UI expects,
and that a signed-in caller is scoped to their own tenant.
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


@pytest_asyncio.fixture
async def client(db, email_sender):
    """An app wired to the test database, with auth on and email captured.

    Driven through ASGITransport rather than TestClient: TestClient runs the
    app on its own event loop, and asyncpg connections belong to the loop that
    opened them, so the shared engine would fail with "attached to a different
    loop". This keeps app and fixtures on one loop.

    The lifespan is skipped (it would build the real account services and touch
    Neo4j); the pieces it would create are injected directly.
    """
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
    r = await client.post("/auth/signup", json={"email": email, "password": PASSWORD})
    assert r.status_code == 200, r.text
    code = email_sender.last_code(email)
    r = await client.post("/auth/verify", json={"email": email, "code": code})
    assert r.status_code == 200, r.text


async def test_signup_verify_sets_a_hardened_session_cookie(client, email_sender):
    await client.post("/auth/signup", json={"email": EMAIL, "password": PASSWORD})
    r = await client.post(
        "/auth/verify", json={"email": EMAIL, "code": email_sender.last_code(EMAIL)}
    )
    assert r.status_code == 200
    assert r.json()["email"] == EMAIL

    raw = r.headers["set-cookie"]
    assert SESSION_COOKIE in raw
    # httpOnly keeps an XSS bug from reading the session; Lax keeps the browser
    # from attaching it to cross-site POSTs.
    assert "httponly" in raw.lower()
    assert "samesite=lax" in raw.lower()
    # This request arrived over plain http, so Secure must be off — with it on
    # the browser would discard the cookie and sign-in would fail silently.
    assert "secure" not in raw.lower()


async def test_https_requests_get_a_secure_cookie(client, email_sender):
    """The mirror of the above: behind a TLS proxy the flag must be set."""
    await client.post("/auth/signup", json={"email": EMAIL, "password": PASSWORD})
    r = await client.post(
        "/auth/verify",
        json={"email": EMAIL, "code": email_sender.last_code(EMAIL)},
        headers={"X-Forwarded-Proto": "https"},
    )
    assert "secure" in r.headers["set-cookie"].lower()


async def test_protected_routes_reject_anonymous_callers(client):
    assert (await client.get("/auth/me")).status_code == 401
    assert (await client.post("/query", json={"question": "hi"})).status_code == 401
    assert (await client.get("/ingest/files")).status_code == 401


async def test_me_reports_the_signed_in_account(client, email_sender):
    await _signup_and_verify(client, email_sender)
    body = (await client.get("/auth/me")).json()
    assert body["email"] == EMAIL
    assert body["role"] == "user"
    assert body["tenant_id"]
    # The UI needs the model list to render its selector.
    assert body["default_model"]


async def test_logout_clears_the_session(client, email_sender):
    await _signup_and_verify(client, email_sender)
    assert (await client.get("/auth/me")).status_code == 200

    assert (await client.post("/auth/logout")).status_code == 200
    client.cookies.clear()
    assert (await client.get("/auth/me")).status_code == 401


async def test_login_with_a_bad_password_is_401(client, email_sender):
    await _signup_and_verify(client, email_sender)
    client.cookies.clear()
    r = await client.post("/auth/login", json={"email": EMAIL, "password": "wrong-password"})
    assert r.status_code == 401
    assert r.json()["detail"]["code"] == "invalid_credentials"


async def test_unverified_login_is_403_so_the_ui_can_route_to_verify(client, email_sender):
    await client.post("/auth/signup", json={"email": EMAIL, "password": PASSWORD})
    r = await client.post("/auth/login", json={"email": EMAIL, "password": PASSWORD})
    assert r.status_code == 403
    assert r.json()["detail"]["code"] == "email_unverified"


async def test_signup_response_does_not_disclose_existing_accounts(client, email_sender):
    await _signup_and_verify(client, email_sender)
    client.cookies.clear()
    first = await client.post(
        "/auth/signup", json={"email": "new@example.com", "password": PASSWORD}
    )
    second = await client.post("/auth/signup", json={"email": EMAIL, "password": PASSWORD})
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()


async def test_api_key_authenticates_without_a_cookie(client, email_sender):
    await _signup_and_verify(client, email_sender)
    key = (await client.post("/auth/keys", json={"label": "ci"})).json()["api_key"]

    client.cookies.clear()
    assert (await client.get("/auth/me")).status_code == 401
    r = await client.get("/auth/me", headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 200
    assert r.json()["email"] == EMAIL


async def test_revoked_key_is_refused(client, email_sender):
    await _signup_and_verify(client, email_sender)
    created = (await client.post("/auth/keys", json={"label": "temp"})).json()
    key = created["api_key"]
    assert (await client.delete(f"/auth/keys/{created['id']}")).status_code == 200

    client.cookies.clear()
    r = await client.get("/auth/me", headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 401


async def test_key_is_returned_once_and_never_listed(client, email_sender):
    await _signup_and_verify(client, email_sender)
    key = (await client.post("/auth/keys", json={"label": "ci"})).json()["api_key"]

    listing = await client.get("/auth/keys")
    assert len(listing.json()["keys"]) == 1
    assert listing.json()["keys"][0]["label"] == "ci"
    assert key not in listing.text


async def test_admin_surface_is_closed_to_ordinary_accounts(client, email_sender):
    await _signup_and_verify(client, email_sender)
    assert (await client.get("/users")).status_code == 403
    assert (await client.get("/usage")).status_code == 403


async def test_x_user_id_header_is_ignored_when_auth_is_on(client, email_sender):
    """Dev mode's identity header must not be a way to impersonate a tenant."""
    await _signup_and_verify(client, email_sender)
    mine = (await client.get("/auth/me")).json()["tenant_id"]
    spoofed = await client.get("/auth/me", headers={"X-User-Id": "somebody-else"})
    assert spoofed.json()["tenant_id"] == mine


async def test_two_accounts_get_separate_tenants(client, email_sender):
    await _signup_and_verify(client, email_sender, "alice@example.com")
    alice = (await client.get("/auth/me")).json()["tenant_id"]

    client.cookies.clear()
    await _signup_and_verify(client, email_sender, "bob@example.com")
    bob = (await client.get("/auth/me")).json()["tenant_id"]
    assert alice != bob
