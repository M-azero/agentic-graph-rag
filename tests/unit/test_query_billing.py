"""One meter per request, and exactly one booking.

The closed-domain probe retrieves before the agent runs. Under a generative
reranker that is `candidate_k` model calls, and it happens on every question
including the ones that are then refused as off-topic — so the cheapest thing a
caller could do was ask questions the knowledge base does not cover, forever,
for free. `aanswer` created its own meter afterwards and never saw any of it.

What has to hold now:
  - the probe's spend is charged, including when the answer is a refusal;
  - it is charged once, not once per path;
  - the meter that the probe used is the one the answer continues on.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from graphrag.accounts import KeyOwner
from graphrag.api.app import create_app
from graphrag.api.routers import query as query_router
from graphrag.config.settings import APICfg, AuthCfg, RetrievalCfg, Secrets, Settings
from graphrag.container import Container
from graphrag.core.types import QueryResult, RetrievedChunk

USER_KEY = "grk_billing-user"
ACCOUNT = "44444444-4444-4444-4444-444444444444"


class _FakeKeyStore:
    available = True

    async def resolve(self, key):
        if key != USER_KEY:
            return None
        return KeyOwner(ACCOUNT, "tenant-b", "user", "active", "b@example.com")


class _Service:
    """A QueryService whose retrieval costs tokens, like a generative rerank."""

    review_enabled = False

    def __init__(self, top_score: float, probe_cost: int = 500) -> None:
        self.top_score = top_score
        self.probe_cost = probe_cost
        self.meters: list = []

    def search(self, query, k=8, user_id=None, meter=None, shelf=None):
        self.meters.append(meter)
        if meter is not None:
            meter.input_tokens += self.probe_cost
            meter.output_tokens += 10
        return [
            RetrievedChunk(
                chunk_id="c1", text="something", source="doc.md",
                score=self.top_score, retriever="vector",
                metadata={"rerank_calibrated": True},
            )
        ]

    async def aanswer(self, question, meter=None, **kwargs):
        self.meters.append(meter)
        if meter is not None:
            meter.input_tokens += 1_000   # the agent's own turns
            meter.output_tokens += 200
        tokens_in, tokens_out = meter.totals if meter else (0, 0)
        return QueryResult(
            answer="An answer [source: doc.md]",
            sources=[], tool_calls=[], source_labels=[],
            input_tokens=tokens_in, output_tokens=tokens_out,
        )


@pytest.fixture
def bookings(monkeypatch) -> list[dict]:
    captured: list[dict] = []

    async def _capture(recorder, redis_client, **kwargs):
        captured.append(kwargs)

    monkeypatch.setattr(query_router, "record_answer_tokens", _capture)
    return captured


def _client(service: _Service, min_relevance: float = 0.5) -> TestClient:
    settings = Settings(
        api=APICfg(docs_enabled=False, stream=False),
        auth=AuthCfg(enabled=True),
        retrieval=RetrievalCfg(min_relevance=min_relevance),
    )
    app = create_app(Container(settings, Secrets(GRAPHRAG_ADMIN_KEY="k")))
    app.state.key_store = _FakeKeyStore()
    app.state.query_service = service
    return TestClient(app, raise_server_exceptions=False)


def _ask(client: TestClient) -> dict:
    response = client.post(
        "/query",
        json={"question": "is this covered?", "stream": False},
        headers={"Authorization": f"Bearer {USER_KEY}"},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_a_refused_question_still_pays_for_its_retrieval(bookings):
    """The regression: an off-topic question ran a full rerank and charged
    nothing, making it the cheapest request in the system."""
    service = _Service(top_score=0.01)      # below min_relevance -> refused
    body = _ask(_client(service))

    assert "couldn't find anything" in body["answer"]
    assert len(bookings) == 1
    assert bookings[0]["input_tokens"] == 500
    assert bookings[0]["tokens"] == 10
    assert bookings[0]["account_id"] == ACCOUNT


def test_an_answered_question_pays_for_probe_and_agent_together(bookings):
    service = _Service(top_score=0.9)       # clears the gate -> answered
    _ask(_client(service))

    assert len(bookings) == 1, "the request was booked more than once"
    assert bookings[0]["input_tokens"] == 1_500   # 500 probe + 1000 agent
    assert bookings[0]["tokens"] == 210           # 10 probe + 200 agent


def test_the_probe_and_the_answer_share_one_meter():
    service = _Service(top_score=0.9)
    _ask(_client(service))

    assert len(service.meters) == 2
    assert service.meters[0] is service.meters[1], (
        "a second meter created inside aanswer cannot see what the probe spent"
    )


def test_no_probe_means_no_double_booking(bookings):
    """With the gate off there is no probe, and the answer is still booked once."""
    service = _Service(top_score=0.9)
    _ask(_client(service, min_relevance=0.0))

    assert service.meters == [service.meters[0]]  # only aanswer ran
    assert len(bookings) == 1
    assert bookings[0]["input_tokens"] == 1_000
