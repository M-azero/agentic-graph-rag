"""The admin surface, against real Postgres.

Two things matter most here and both are tested directly: that the surface is
closed to everyone except admins, and that an admin's edit actually changes
what a user experiences on their next request.
"""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio

from graphrag.accounts import AccountService, PgKeyStore
from graphrag.api.app import create_app
from graphrag.container import Container
from graphrag.limits import LimitService
from graphrag.usage import UsageRecorder
from tests.integration.conftest import relax_auth_limits, requires_db
from tests.integration.test_limits_api import StubQueryService
from tests.unit.test_limits_service import FakeRedis

pytestmark = [pytest.mark.integration, requires_db]

ADMIN_KEY = "test-admin-key"
PASSWORD = "correct-horse-battery"
ADMIN_HEADERS = {"X-Admin-Key": ADMIN_KEY}


@pytest_asyncio.fixture
async def client(db, email_sender):
    container = Container()
    container.settings.auth.enabled = True
    container.settings.storage.vector.provider = "duckdb"
    relax_auth_limits(container)
    container.secrets.admin_api_key = ADMIN_KEY

    app = create_app(container)
    app.state.db = db
    app.state.accounts = AccountService(db, container.settings, email_sender)
    app.state.key_store = PgKeyStore(db)
    app.state.limits = LimitService(db, FakeRedis())
    app.state.usage = UsageRecorder(db, app.state.limits)
    app.state.query_service = StubQueryService()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _signup(client, email_sender, email: str) -> str:
    await client.post("/auth/signup", json={"email": email, "password": PASSWORD})
    r = await client.post(
        "/auth/verify", json={"email": email, "code": email_sender.last_code(email)}
    )
    assert r.status_code == 200, r.text
    return r.json()["user_id"]


# -- access control -----------------------------------------------------------

async def test_admin_endpoints_reject_anonymous_callers(client):
    for path in ("/admin/users", "/admin/limits", "/admin/system", "/admin/usage"):
        assert (await client.get(path)).status_code in (401, 403), path


async def test_ordinary_users_are_refused(client, email_sender):
    await _signup(client, email_sender, "alice@example.com")
    assert (await client.get("/admin/users")).status_code == 403


async def test_the_admin_key_opens_the_door(client, email_sender):
    await _signup(client, email_sender, "alice@example.com")
    client.cookies.clear()
    r = await client.get("/admin/users", headers=ADMIN_HEADERS)
    assert r.status_code == 200
    assert r.json()["total"] == 1


async def test_an_admin_account_opens_the_door_without_the_key(client, email_sender):
    await _signup(client, email_sender, "boss@example.com")
    await client.patch(
        f"/admin/users/{(await client.get('/auth/me')).json()['user_id']}",
        json={"role": "admin"},
        headers=ADMIN_HEADERS,
    )
    # Same session, now an admin — no header needed.
    assert (await client.get("/admin/users")).status_code == 200


# -- user management ----------------------------------------------------------

async def test_users_can_be_listed_searched_and_filtered(client, email_sender):
    await _signup(client, email_sender, "alice@example.com")
    client.cookies.clear()
    await _signup(client, email_sender, "bob@example.com")
    client.cookies.clear()

    listed = (await client.get("/admin/users", headers=ADMIN_HEADERS)).json()
    assert listed["total"] == 2

    found = (
        await client.get("/admin/users?query=alice", headers=ADMIN_HEADERS)
    ).json()
    assert [u["email"] for u in found["users"]] == ["alice@example.com"]

    active = (
        await client.get("/admin/users?status=active", headers=ADMIN_HEADERS)
    ).json()
    assert active["total"] == 2


async def test_user_detail_reports_limits_and_storage(client, email_sender):
    user_id = await _signup(client, email_sender, "alice@example.com")
    await client.post("/threads", json={"title": "one"})
    client.cookies.clear()

    body = (await client.get(f"/admin/users/{user_id}", headers=ADMIN_HEADERS)).json()
    assert body["user"]["email"] == "alice@example.com"
    assert body["user"]["threads"] == 1
    assert body["limits"]["messages_per_day"] == 100
    assert all(v is None for v in body["overrides"].values())


async def test_suspending_a_user_cuts_off_their_session(client, email_sender):
    user_id = await _signup(client, email_sender, "alice@example.com")
    assert (await client.get("/auth/me")).status_code == 200

    r = await client.patch(
        f"/admin/users/{user_id}", json={"status": "suspended"}, headers=ADMIN_HEADERS
    )
    assert r.status_code == 200 and r.json()["status"] == "suspended"

    # Same cookie, now worthless.
    assert (await client.get("/auth/me")).status_code == 401


async def test_reactivating_a_user_lets_them_back_in(client, email_sender):
    user_id = await _signup(client, email_sender, "alice@example.com")
    await client.patch(
        f"/admin/users/{user_id}", json={"status": "suspended"}, headers=ADMIN_HEADERS
    )
    await client.patch(
        f"/admin/users/{user_id}", json={"status": "active"}, headers=ADMIN_HEADERS
    )
    client.cookies.clear()
    r = await client.post(
        "/auth/login", json={"email": "alice@example.com", "password": PASSWORD}
    )
    assert r.status_code == 200


async def test_invalid_status_or_role_is_rejected(client, email_sender):
    user_id = await _signup(client, email_sender, "alice@example.com")
    client.cookies.clear()
    for payload in ({"status": "deleted"}, {"role": "superuser"}):
        r = await client.patch(
            f"/admin/users/{user_id}", json=payload, headers=ADMIN_HEADERS
        )
        assert r.status_code == 400


async def test_unknown_user_is_a_404(client):
    import uuid

    r = await client.get(f"/admin/users/{uuid.uuid4()}", headers=ADMIN_HEADERS)
    assert r.status_code == 404
    assert (await client.get("/admin/users/not-a-uuid", headers=ADMIN_HEADERS)).status_code == 404


# -- limits -------------------------------------------------------------------

async def test_editing_global_limits_changes_what_a_user_gets(client, email_sender):
    await _signup(client, email_sender, "alice@example.com")

    r = await client.put(
        "/admin/limits", json={"messages_per_day": 7}, headers=ADMIN_HEADERS
    )
    assert r.status_code == 200 and r.json()["messages_per_day"] == 7

    # The user's own view reflects it immediately: the edit invalidated the cache.
    assert (await client.get("/auth/limits")).json()["limits"]["messages_per_day"] == 7


async def test_per_user_override_and_clearing_it(client, email_sender):
    user_id = await _signup(client, email_sender, "alice@example.com")

    await client.put(
        f"/admin/users/{user_id}/limits", json={"max_files": 3}, headers=ADMIN_HEADERS
    )
    assert (await client.get("/auth/limits")).json()["limits"]["max_files"] == 3

    await client.delete(f"/admin/users/{user_id}/limits", headers=ADMIN_HEADERS)
    assert (await client.get("/auth/limits")).json()["limits"]["max_files"] == 10


async def test_an_admin_edit_is_enforced_on_the_next_request(client, email_sender):
    """The end-to-end promise of the panel: change a number, the user feels it."""
    user_id = await _signup(client, email_sender, "alice@example.com")
    await client.put(
        f"/admin/users/{user_id}/limits",
        json={"messages_per_day": 1},
        headers=ADMIN_HEADERS,
    )

    allowed = await client.post("/query", json={"question": "one", "stream": False})
    assert allowed.status_code == 200
    blocked = await client.post("/query", json={"question": "two", "stream": False})
    assert blocked.status_code == 429
    assert blocked.json()["detail"]["limit"] == "messages_per_day"


async def test_bulk_limits_apply_to_everyone(client, email_sender):
    await _signup(client, email_sender, "alice@example.com")
    client.cookies.clear()
    await _signup(client, email_sender, "bob@example.com")

    r = await client.post(
        "/admin/limits/bulk",
        json={"set": {"messages_per_day": 42}},
        headers=ADMIN_HEADERS,
    )
    assert r.status_code == 200 and "2" in r.json()["message"]
    assert (await client.get("/auth/limits")).json()["limits"]["messages_per_day"] == 42


async def test_bulk_clear_restores_the_defaults(client, email_sender):
    user_id = await _signup(client, email_sender, "alice@example.com")
    await client.put(
        f"/admin/users/{user_id}/limits", json={"messages_per_day": 1},
        headers=ADMIN_HEADERS,
    )
    assert (await client.get("/auth/limits")).json()["limits"]["messages_per_day"] == 1

    await client.post("/admin/limits/bulk", json={"clear": True}, headers=ADMIN_HEADERS)
    assert (await client.get("/auth/limits")).json()["limits"]["messages_per_day"] == 100


async def test_empty_bulk_update_is_rejected(client):
    r = await client.post("/admin/limits/bulk", json={}, headers=ADMIN_HEADERS)
    assert r.status_code == 400


# -- usage, models, system ----------------------------------------------------

async def test_usage_series_reports_recorded_activity(client, email_sender):
    await _signup(client, email_sender, "alice@example.com")
    await client.post("/query", json={"question": "hi", "stream": False})
    client.cookies.clear()

    body = (await client.get("/admin/usage?days=7", headers=ADMIN_HEADERS)).json()
    assert "messages" in body["totals"]
    assert isinstance(body["points"], list)


async def test_models_can_be_narrowed_but_not_emptied(client):
    available = (await client.get("/admin/models", headers=ADMIN_HEADERS)).json()
    assert available["available"]
    first = available["available"][0]["model"]

    r = await client.put(
        "/admin/models", json={"enabled": [first]}, headers=ADMIN_HEADERS
    )
    assert r.status_code == 200 and r.json()["enabled"] == [first]

    assert (
        await client.put("/admin/models", json={"enabled": []}, headers=ADMIN_HEADERS)
    ).status_code == 400
    assert (
        await client.put(
            "/admin/models", json={"enabled": ["not-a-real-model"]}, headers=ADMIN_HEADERS
        )
    ).status_code == 400


async def test_system_status_reports_the_deployment(client, email_sender):
    await _signup(client, email_sender, "alice@example.com")
    client.cookies.clear()

    body = (await client.get("/admin/system", headers=ADMIN_HEADERS)).json()
    assert body["database"] is True
    assert body["users"] == 1
    assert body["active_users"] == 1
    assert body["vector_provider"] == "duckdb"
    assert body["default_model"]


# -- audit --------------------------------------------------------------------

async def test_mutations_are_recorded_in_the_audit_log(client, email_sender):
    user_id = await _signup(client, email_sender, "alice@example.com")
    client.cookies.clear()
    await client.patch(
        f"/admin/users/{user_id}", json={"status": "suspended"}, headers=ADMIN_HEADERS
    )
    await client.put("/admin/limits", json={"max_files": 4}, headers=ADMIN_HEADERS)

    entries = (await client.get("/admin/audit", headers=ADMIN_HEADERS)).json()
    actions = [e["action"] for e in entries]
    assert "user.patch" in actions
    assert "limits.global" in actions


async def test_reads_do_not_pollute_the_audit_log(client):
    await client.get("/admin/users", headers=ADMIN_HEADERS)
    await client.get("/admin/system", headers=ADMIN_HEADERS)
    assert (await client.get("/admin/audit", headers=ADMIN_HEADERS)).json() == []


# -- purge --------------------------------------------------------------------

async def test_deleting_a_user_removes_their_rows(client, email_sender):
    user_id = await _signup(client, email_sender, "alice@example.com")
    await client.post("/threads", json={"title": "will be gone"})
    client.cookies.clear()

    r = await client.delete(f"/admin/users/{user_id}", headers=ADMIN_HEADERS)
    assert r.status_code == 200
    assert r.json()["rows_removed"] is True

    assert (await client.get(f"/admin/users/{user_id}", headers=ADMIN_HEADERS)).status_code == 404
    assert (await client.get("/admin/users", headers=ADMIN_HEADERS)).json()["total"] == 0


async def test_keeping_the_account_wipes_only_the_content(client, email_sender):
    user_id = await _signup(client, email_sender, "alice@example.com")
    client.cookies.clear()

    r = await client.delete(
        f"/admin/users/{user_id}?keep_account=true", headers=ADMIN_HEADERS
    )
    assert r.status_code == 200

    # The login survives.
    login = await client.post(
        "/auth/login", json={"email": "alice@example.com", "password": PASSWORD}
    )
    assert login.status_code == 200


# -- account management -------------------------------------------------------

async def test_admin_can_invite_an_account(client, email_sender):
    """What makes `open_registration: false` usable rather than a dead end.

    No password crosses this boundary: the invitee gets a code and sets their
    own through the ordinary reset flow.
    """
    r = await client.post(
        "/admin/users",
        json={"email": "new@example.com", "role": "user"},
        headers=ADMIN_HEADERS,
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["email"] == "new@example.com"
    assert body["status"] == "active"
    assert body["email_verified"] is True
    assert body["tenant_id"]

    subject = email_sender.sent[-1][1]
    assert "password" in subject.lower()
    code = email_sender.last_code("new@example.com")

    # The invite code is a reset code, so it drives the ordinary reset flow.
    reset = await client.post(
        "/auth/reset-password",
        json={"email": "new@example.com", "code": code, "password": PASSWORD},
    )
    assert reset.status_code == 200, reset.text
    login = await client.post(
        "/auth/login", json={"email": "new@example.com", "password": PASSWORD}
    )
    assert login.status_code == 200


async def test_inviting_an_existing_address_is_a_409(client, email_sender):
    await _signup(client, email_sender, "alice@example.com")
    client.cookies.clear()
    r = await client.post(
        "/admin/users", json={"email": "alice@example.com"}, headers=ADMIN_HEADERS
    )
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "already_exists"


async def test_invite_rejects_an_unknown_role(client):
    r = await client.post(
        "/admin/users",
        json={"email": "new@example.com", "role": "superuser"},
        headers=ADMIN_HEADERS,
    )
    assert r.status_code == 400


async def test_forcing_a_reset_cuts_the_user_off_and_mails_a_code(
    client, email_sender
):
    user_id = await _signup(client, email_sender, "alice@example.com")
    assert (await client.get("/auth/me")).status_code == 200
    stolen = dict(client.cookies)
    client.cookies.clear()

    r = await client.post(
        f"/admin/users/{user_id}/force-password-reset", headers=ADMIN_HEADERS
    )
    assert r.status_code == 200, r.text

    # The session that existed before the reset is dead.
    for name, value in stolen.items():
        client.cookies.set(name, value)
    assert (await client.get("/auth/me")).status_code == 401
    client.cookies.clear()

    code = email_sender.last_code("alice@example.com")
    new_password = "a-brand-new-secret-value"
    reset = await client.post(
        "/auth/reset-password",
        json={"email": "alice@example.com", "code": code, "password": new_password},
    )
    assert reset.status_code == 200, reset.text


async def test_forcing_a_reset_on_an_unknown_user_is_404(client):
    import uuid

    r = await client.post(
        f"/admin/users/{uuid.uuid4()}/force-password-reset", headers=ADMIN_HEADERS
    )
    assert r.status_code == 404


async def test_unlock_clears_a_lockout(client, email_sender):
    user_id = await _signup(client, email_sender, "alice@example.com")
    client.cookies.clear()

    r = await client.post(f"/admin/users/{user_id}/unlock", headers=ADMIN_HEADERS)
    assert r.status_code == 200
    detail = (await client.get(f"/admin/users/{user_id}", headers=ADMIN_HEADERS)).json()
    assert detail["user"]["locked_until"] is None
    assert detail["user"]["failed_logins"] == 0


async def test_account_management_is_audited(client, email_sender):
    """"Who reset this user's password" is the first question asked afterwards."""
    user_id = await _signup(client, email_sender, "alice@example.com")
    client.cookies.clear()

    await client.post("/admin/users", json={"email": "new@example.com"}, headers=ADMIN_HEADERS)
    await client.post(
        f"/admin/users/{user_id}/force-password-reset", headers=ADMIN_HEADERS
    )
    await client.post(f"/admin/users/{user_id}/unlock", headers=ADMIN_HEADERS)

    actions = {
        e["action"] for e in (await client.get("/admin/audit", headers=ADMIN_HEADERS)).json()
    }
    assert {"user.create", "user.force_reset", "user.unlock"} <= actions
