"""The trace endpoint end to end: the steps a real run produces, the order they
arrive in, and the two rules the endpoint exists to keep — a traced run is
metered like any other, and the answer waits for the output guard."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from graphrag.api.app import create_app
from graphrag.config.settings import APICfg, AuthCfg, RetrievalCfg, Secrets, Settings
from graphrag.container import Container
from graphrag.core.types import QueryResult, RetrievedChunk
from graphrag.trace import steps as registry
from graphrag.trace.store import TraceStore


class _StubService:
    """Stands in for QueryService: no model, no Neo4j, but it walks the same
    helpers the router calls, so the steps around it are the real ones."""

    review_enabled = False

    def __init__(self, chunks=None, answer="Robots live in the warehouse [warehouse.pdf]."):
        self._chunks = chunks if chunks is not None else [
            RetrievedChunk(
                chunk_id="c1", text="Pallet Pilot is a warehouse robot.",
                source="warehouse.pdf", score=0.82, retriever="hybrid",
            )
        ]
        self._answer = answer
        self.calls = 0

    def search(self, query, k=8, user_id=None, meter=None, shelf=None):
        return self._chunks

    async def aanswer(self, question, **kw):
        self.calls += 1
        return QueryResult(
            answer=self._answer,
            sources=self._chunks,
            tool_calls=[{"tool": "hybrid_search", "args": {"query": question}}],
            source_labels=[],
            input_tokens=140,
            output_tokens=32,
        )


# No `with`, so the lifespan never runs — the same shape `test_query_billing.py`
# uses, and for the same reason: entering it would open a database engine and a
# checkpointer against services that are not up in a unit run.
#
# Module-scoped because constructing the app loads models; each test resets the
# state it is allowed to touch.
@pytest.fixture(scope="module")
def _app():
    settings = Settings(
        # The module shares one rate-limit bucket (one app, one caller id), so
        # the stock 60/minute would start 429ing partway through the file.
        api=APICfg(docs_enabled=False, stream=False, rate_limit="1000/minute"),
        auth=AuthCfg(enabled=False),
        retrieval=RetrievalCfg(min_relevance=0.3),
    )
    # A deliberately dead Redis. Whether the developer happens to have the stack
    # running must not change what these tests do — and with a live Redis both
    # the rate limiter and the trace store become shared, persistent state, so
    # the module 429s partway through and traces survive between tests.
    secrets = Secrets(GRAPHRAG_ADMIN_KEY="k", GRAPHRAG_REDIS_URL="redis://127.0.0.1:6399/0")
    app = create_app(Container(settings, secrets))
    return app, TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def client(_app):
    app, c = _app
    app.state.query_service = _StubService()
    app.state.container.settings.trace.enabled = True
    # A store of its own, with no Redis. `create_app` wires one up against
    # `container.redis`, so on a machine where the stack happens to be running
    # these tests would share state with a live Redis — and with each other,
    # since a stored trace outlives the test that made it.
    app.state.trace_store = TraceStore(None)
    c.headers.update({"X-User-Id": "alice"})
    yield c


@pytest.fixture
def app(_app):
    return _app[0]


def _events(response) -> list[tuple[str, str]]:
    """Parse the SSE body into (event, data) pairs."""
    out, event = [], None
    for line in response.text.splitlines():
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:") and event:
            out.append((event, line[5:].strip()))
    return out


def _steps(response) -> list[dict]:
    """The steps a run reported, latest copy of each.

    A step is re-sent while it is still running, so the same id can appear more
    than once; the last copy is the settled one.
    """
    latest: dict[str, dict] = {}
    for event, data in _events(response):
        if event == "step":
            payload = json.loads(data)
            latest[payload["id"]] = payload
    return list(latest.values())


def _answer(response) -> dict:
    return json.loads(next(d for e, d in _events(response) if e == "answer"))


def test_the_diagram_is_served_and_matches_the_registry(client):
    body = client.get("/trace/pipelines").json()
    assert {p["pipeline"] for p in body["pipelines"]} == set(registry.PIPELINES)


def test_a_run_streams_steps_then_one_answer_then_done(client):
    r = client.post("/trace/query", json={"question": "what robots are there?"})
    assert r.status_code == 200
    events = _events(r)
    kinds = [e for e, _ in events]

    assert kinds.count("answer") == 1
    assert kinds[-1] == "done"
    # The answer is the last thing before `done`: it waits for the output guard,
    # which is what keeps block/redact enforceable on this path.
    assert kinds[-2] == "answer"
    assert kinds.index("answer") > kinds.index("step")


def test_the_real_blocks_of_the_query_path_are_recorded(client):
    r = client.post("/trace/query", json={"question": "what robots are there?"})
    names = [s["name"] for s in _steps(r)]
    for expected in (
        registry.GUARD_INPUT,
        registry.GATE_PROBE,
        registry.CHECK_CITATIONS,
        registry.GUARD_OUTPUT,
        registry.BILLING,
    ):
        assert expected in names, f"{expected} missing from {names}"


def test_a_step_carries_its_real_input_and_output(client):
    r = client.post("/trace/query", json={"question": "what robots are there?"})
    gate = next(s for s in _steps(r) if s["name"] == registry.GATE_PROBE)
    assert gate["input"]["min_relevance"] == 0.3
    assert gate["output"]["top_score"] == pytest.approx(0.82)
    assert gate["output"]["passed"] is True
    assert gate["status"] == "ok"
    assert gate["duration_ms"] is not None


def test_the_gate_refusal_is_visible_as_a_failed_check_not_a_missing_step(client, app):
    app.state.query_service = _StubService(
        chunks=[
            RetrievedChunk(
                chunk_id="c1", text="unrelated", source="other.pdf",
                score=0.01, retriever="hybrid",
            )
        ]
    )
    r = client.post("/trace/query", json={"question": "what is the capital of France?"})
    steps = _steps(r)
    gate = next(s for s in steps if s["name"] == registry.GATE_PROBE)
    assert gate["output"]["passed"] is False
    # The probe still cost retrieval, so the run is still billed.
    assert any(s["name"] == registry.BILLING for s in steps)
    answer = _answer(r)
    assert answer["outcome"] == "off_topic"


def test_a_disabled_guard_is_skipped_rather_than_silently_absent(client):
    r = client.post("/trace/query", json={"question": "what robots are there?"})
    guard = next(s for s in _steps(r) if s["name"] == registry.GUARD_INPUT)
    # safety.enabled is false by default: the block is drawn and reports why it
    # did nothing, which is not the same as never having been reached.
    assert guard["status"] == "skipped"
    assert "safety.enabled" in guard["meta"]["reason"]


def test_the_run_is_metered_like_any_other(client):
    """A traced run makes the same model calls as an untraced one. If it did not
    bill, this endpoint would be a free query."""
    r = client.post("/trace/query", json={"question": "what robots are there?"})
    billing = next(s for s in _steps(r) if s["name"] == registry.BILLING)
    assert billing["output"]["input_tokens"] >= 0
    answer = _answer(r)
    assert answer["answer"]


def test_the_trace_is_readable_afterwards_and_scoped_to_its_owner(client):
    r = client.post("/trace/query", json={"question": "what robots are there?"})
    trace_id = _answer(r)["trace_id"]

    stored = client.get(f"/trace/{trace_id}")
    assert stored.status_code == 200
    assert stored.json()["steps"]
    # Never echo the ACL field.
    assert "owner" not in stored.json()

    client.headers.update({"X-User-Id": "mallory"})
    assert client.get(f"/trace/{trace_id}").status_code == 404
    assert client.get("/trace/runs").json()["runs"] == []


def test_recent_runs_list_the_callers_own(client):
    client.post("/trace/query", json={"question": "first question"})
    client.post("/trace/query", json={"question": "second question"})
    runs = client.get("/trace/runs").json()["runs"]
    assert [r["title"] for r in runs] == ["second question", "first question"]


def test_tracing_off_hides_the_surface_entirely(client, app):
    app.state.container.settings.trace.enabled = False
    # 404 rather than 403: with tracing off the endpoint should not advertise
    # that it exists.
    assert client.get("/trace/pipelines").status_code == 404
    assert client.post("/trace/query", json={"question": "hi"}).status_code == 404
    assert client.get("/trace/runs").status_code == 404
