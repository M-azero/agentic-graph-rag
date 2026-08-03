"""Limits and conversations over HTTP, against real Postgres.

The point of these is that a quota actually *stops* a request — counting usage
without refusing anything is the failure mode the old code had.
"""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio

from graphrag.accounts import AccountService, PgKeyStore
from graphrag.api.app import create_app
from graphrag.container import Container
from graphrag.db.engine import session_scope
from graphrag.db.models import GlobalLimit, UserLimit
from graphrag.limits import LimitService
from graphrag.usage import UsageRecorder
from tests.integration.conftest import relax_auth_limits, requires_db
from tests.unit.test_limits_service import FakeRedis

pytestmark = [pytest.mark.integration, requires_db]

EMAIL = "alice@example.com"
PASSWORD = "correct-horse-battery"


class StubQueryService:
    """Answers without a model.

    These tests are about the gate in front of the agent, not the agent. A real
    QueryService would reach for an embedder and an LLM that aren't running
    here, turning a limit assertion into a connectivity test.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def aanswer(self, question, style=None, thread_id="default", user_id=None, model=None):
        from graphrag.core.types import QueryResult

        self.calls.append(question)
        return QueryResult(answer=f"answer to: {question}", sources=[], tool_calls=[])

    async def stream(self, question, style=None, thread_id="default", user_id=None, model=None):
        self.calls.append(question)
        for token in ("answer ", "to ", question):
            yield "token", token, []


@pytest_asyncio.fixture
async def client(db, email_sender):
    """App on the test database. Redis is faked so limit counters are real but
    isolated per test — a shared Redis would leak windows between them."""
    container = Container()
    container.settings.auth.enabled = True
    container.settings.storage.vector.provider = "duckdb"
    relax_auth_limits(container)

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


async def _signup(client, email_sender, email: str = EMAIL) -> str:
    await client.post("/auth/signup", json={"email": email, "password": PASSWORD})
    r = await client.post(
        "/auth/verify", json={"email": email, "code": email_sender.last_code(email)}
    )
    assert r.status_code == 200, r.text
    return r.json()["user_id"]


async def _set_limits(db, user_id: str | None = None, **values) -> None:
    """Pin a limit, globally or for one user."""
    from sqlalchemy import select

    async with session_scope(db) as s:
        if user_id is None:
            row = (await s.execute(select(GlobalLimit))).scalar_one()
        else:
            row = (
                await s.execute(select(UserLimit).where(UserLimit.user_id == user_id))
            ).scalar_one_or_none()
            if row is None:
                row = UserLimit(user_id=user_id)
                s.add(row)
        for key, value in values.items():
            setattr(row, key, value)


# -- threads ------------------------------------------------------------------

async def test_threads_are_created_listed_and_deleted(client, email_sender):
    await _signup(client, email_sender)

    created = await client.post("/threads", json={"title": "First chat"})
    assert created.status_code == 200
    thread_id = created.json()["id"]

    listed = await client.get("/threads")
    assert [t["id"] for t in listed.json()["threads"]] == [thread_id]

    assert (await client.delete(f"/threads/{thread_id}")).status_code == 200
    assert (await client.get("/threads")).json()["threads"] == []


async def test_threads_can_be_renamed(client, email_sender):
    await _signup(client, email_sender)
    thread_id = (await client.post("/threads", json={"title": "x"})).json()["id"]

    renamed = await client.patch(f"/threads/{thread_id}", json={"title": "Renamed"})
    assert renamed.json()["title"] == "Renamed"


async def test_one_user_cannot_reach_another_users_thread(client, email_sender):
    """A thread id is not a capability — ownership is checked on every read."""
    await _signup(client, email_sender, "alice@example.com")
    alice_thread = (await client.post("/threads", json={"title": "private"})).json()["id"]

    client.cookies.clear()
    await _signup(client, email_sender, "bob@example.com")

    # Indistinguishable from a thread that never existed.
    assert (await client.get(f"/threads/{alice_thread}/messages")).status_code == 404
    assert (await client.patch(f"/threads/{alice_thread}", json={"title": "hi"})).status_code == 404
    assert (await client.delete(f"/threads/{alice_thread}")).status_code == 404
    assert (await client.get("/threads")).json()["threads"] == []


async def test_thread_limit_is_enforced(client, email_sender):
    user_id = await _signup(client, email_sender)
    await _set_limits(client_db(client), user_id, max_threads=2)

    for _ in range(2):
        assert (await client.post("/threads", json={"title": "ok"})).status_code == 200

    blocked = await client.post("/threads", json={"title": "one too many"})
    assert blocked.status_code == 429
    detail = blocked.json()["detail"]
    assert detail["limit"] == "max_threads"
    assert detail["max"] == 2


async def test_deleting_a_thread_frees_its_slot(client, email_sender):
    user_id = await _signup(client, email_sender)
    await _set_limits(client_db(client), user_id, max_threads=1)

    first = (await client.post("/threads", json={"title": "one"})).json()["id"]
    assert (await client.post("/threads", json={"title": "two"})).status_code == 429

    await client.delete(f"/threads/{first}")
    assert (await client.post("/threads", json={"title": "two"})).status_code == 200


# -- message limits -----------------------------------------------------------

async def test_message_limit_returns_a_structured_429(client, email_sender):
    user_id = await _signup(client, email_sender)
    await _set_limits(client_db(client), user_id, messages_per_day=1)

    first = await client.post("/query", json={"question": "hi", "stream": False})
    assert first.status_code == 200

    blocked = await client.post("/query", json={"question": "again", "stream": False})
    assert blocked.status_code == 429
    detail = blocked.json()["detail"]
    assert detail["code"] == "limit_exceeded"
    assert detail["limit"] == "messages_per_day"
    assert detail["max"] == 1
    assert blocked.headers.get("Retry-After")


async def test_a_blocked_request_never_reaches_the_agent(client, email_sender):
    """The point of the gate: a refused request must not cost a model call."""
    user_id = await _signup(client, email_sender)
    await _set_limits(client_db(client), user_id, messages_per_day=1)

    await client.post("/query", json={"question": "first", "stream": False})
    await client.post("/query", json={"question": "second", "stream": False})
    assert client_service(client).calls == ["first"]


async def test_per_user_override_beats_the_global_default(client, email_sender):
    user_id = await _signup(client, email_sender)
    await _set_limits(client_db(client), None, messages_per_day=1)
    await _set_limits(client_db(client), user_id, messages_per_day=3)

    for _ in range(3):
        r = await client.post("/query", json={"question": "hi", "stream": False})
        assert r.status_code == 200
    blocked = await client.post("/query", json={"question": "hi", "stream": False})
    assert blocked.status_code == 429


async def test_the_transcript_is_saved_to_the_thread(client, email_sender):
    """Chat history moved from localStorage to the server; this is that path."""
    await _signup(client, email_sender)
    thread_id = (await client.post("/threads", json={"title": "New chat"})).json()["id"]

    r = await client.post(
        "/query", json={"question": "what is RAG?", "thread_id": thread_id, "stream": False}
    )
    assert r.status_code == 200

    messages = (await client.get(f"/threads/{thread_id}/messages")).json()["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "what is RAG?"
    assert "what is RAG?" in messages[1]["content"]


async def test_a_new_thread_is_titled_from_its_first_question(client, email_sender):
    await _signup(client, email_sender)
    thread_id = (await client.post("/threads", json={})).json()["id"]
    await client.post(
        "/query",
        json={"question": "How does graph retrieval work?", "thread_id": thread_id,
              "stream": False},
    )
    listed = (await client.get("/threads")).json()["threads"]
    assert listed[0]["title"] == "How does graph retrieval work?"


async def test_posting_to_another_users_thread_is_refused(client, email_sender):
    await _signup(client, email_sender, "alice@example.com")
    alice_thread = (await client.post("/threads", json={})).json()["id"]

    client.cookies.clear()
    await _signup(client, email_sender, "bob@example.com")
    r = await client.post(
        "/query", json={"question": "peek", "thread_id": alice_thread, "stream": False}
    )
    assert r.status_code == 404


async def test_limits_are_reported_for_the_account_page(client, email_sender):
    user_id = await _signup(client, email_sender)
    await _set_limits(client_db(client), user_id, messages_per_day=7, max_files=3)

    body = (await client.get("/auth/limits")).json()
    assert body["limits"]["messages_per_day"] == 7
    assert body["limits"]["max_files"] == 3
    assert body["usage"]["messages_today"] == 0
    assert body["threads_used"] == 0


async def test_editing_limits_takes_effect_after_invalidation(client, email_sender):
    """Cached limits must not outlive an admin's edit."""
    user_id = await _signup(client, email_sender)
    await _set_limits(client_db(client), user_id, messages_per_day=1)
    assert (await client.get("/auth/limits")).json()["limits"]["messages_per_day"] == 1

    await _set_limits(client_db(client), user_id, messages_per_day=50)
    assert (await client.get("/auth/limits")).json()["limits"]["messages_per_day"] == 1  # cached

    client_limits(client).invalidate(user_id)
    assert (await client.get("/auth/limits")).json()["limits"]["messages_per_day"] == 50


# -- helpers ------------------------------------------------------------------

def client_db(client):
    return client._transport.app.state.db


def client_limits(client) -> LimitService:
    return client._transport.app.state.limits


def client_service(client) -> StubQueryService:
    return client._transport.app.state.query_service
