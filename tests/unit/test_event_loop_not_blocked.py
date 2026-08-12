"""Blocking retrieval does not run on the event loop.

`service.search` is fully synchronous — an embedding round trip, three Neo4j
queries and a rerank — and it was called inline from three `async def` handlers.
The API runs a single uvicorn worker by construction (the DuckDB vector provider
takes an exclusive lock per tenant file), so for the whole duration of any
search *every other request was frozen*, including token flushes for other
users' in-flight streams.

Thread identity is the test, because that is the property: correctness is
unchanged either way, only concurrency differs, and a latency assertion would be
flaky. `asyncio.to_thread` also copies the caller's context, which is what keeps
the token meter visible inside the retriever's own pools — asserted here too, so
a switch to a bare executor cannot silently unmeter retrieval.
"""

from __future__ import annotations

import asyncio
import threading

import pytest
from fastapi.testclient import TestClient

from graphrag.accounts import KeyOwner
from graphrag.api.app import create_app
from graphrag.config.settings import APICfg, AuthCfg, Secrets, Settings
from graphrag.container import Container
from graphrag.usage.meter import TokenMeter, active_meter

USER_KEY = "grk_loop-user"


class _FakeKeyStore:
    available = True

    async def resolve(self, key: str) -> KeyOwner | None:
        if key != USER_KEY:
            return None
        return KeyOwner("33333333-3333-3333-3333-333333333333", "tenant-l", "user",
                        "active", "l@example.com")


class _RecordingService:
    """Stands in for QueryService, capturing where and how `search` was called."""

    def __init__(self) -> None:
        self.thread: int | None = None
        self.meter_seen: TokenMeter | None = None

    def search(self, query, k=8, user_id=None, meter=None, shelf=None):
        self.thread = threading.get_ident()
        self.meter_seen = meter or active_meter()
        return []


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def client(monkeypatch) -> tuple[TestClient, _RecordingService]:
    settings = Settings(api=APICfg(docs_enabled=False), auth=AuthCfg(enabled=True))
    app = create_app(Container(settings, Secrets(GRAPHRAG_ADMIN_KEY="k")))
    app.state.key_store = _FakeKeyStore()
    service = _RecordingService()
    app.state.query_service = service
    return TestClient(app, raise_server_exceptions=False), service


def test_search_runs_off_the_event_loop(client):
    http, service = client
    response = http.post(
        "/search", json={"query": "anything"},
        headers={"Authorization": f"Bearer {USER_KEY}"},
    )
    assert response.status_code == 200
    assert service.thread is not None
    assert service.thread != threading.get_ident(), (
        "search ran on the caller's thread — the event loop is blocked for its "
        "whole duration"
    )


@pytest.mark.anyio
async def test_the_probe_runs_off_the_loop_and_carries_the_meter():
    from graphrag.api.routers.query import _probe

    service = _RecordingService()
    meter = TokenMeter()
    loop_thread = threading.get_ident()

    await _probe(service, "a question", "tenant-l", meter)

    assert service.thread != loop_thread
    assert service.meter_seen is meter, (
        "the probe's retrieval must stay metered: under a generative reranker "
        "it is one model call per candidate"
    )


@pytest.mark.anyio
async def test_to_thread_propagates_the_bound_context():
    """The mechanism the two tests above rely on, asserted directly."""
    from graphrag.usage.meter import use_meter

    meter = TokenMeter()
    with use_meter(meter):
        seen = await asyncio.to_thread(active_meter)
    assert seen is meter
