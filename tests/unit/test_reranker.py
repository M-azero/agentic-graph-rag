"""Generative reranking — score parsing and ordering.

Ollama has no cross-encoder endpoint, so a local reranker leans on a chat model
returning a bare number. These cover what happens when it doesn't.
"""

from types import SimpleNamespace

import pytest

from graphrag.config.settings import RerankCfg, Secrets
from graphrag.retrieval import reranker as rr


class _FakeLLM:
    """Replies are keyed by document text, not call order — `rerank` scores
    candidates concurrently, so order isn't deterministic."""

    def __init__(self, by_doc: dict[str, str]) -> None:
        self.by_doc = by_doc

    def invoke(self, prompt: str):
        for doc, reply in self.by_doc.items():
            if doc in prompt:
                return SimpleNamespace(content=reply)
        raise AssertionError(f"unexpected document in prompt: {prompt!r}")


@pytest.fixture
def build(monkeypatch):
    def _build(by_doc: dict[str, str], **cfg_kwargs):
        monkeypatch.setattr(rr, "build_chat_chain", lambda *a, **k: _FakeLLM(by_doc))
        cfg = RerankCfg(provider="ollama", model="fake-model", **cfg_kwargs)
        return rr.build_reranker(cfg, Secrets())

    return _build


def test_dispatches_to_generative_reranker(build):
    assert isinstance(build({}), rr.LLMReranker)


def test_orders_by_model_score(build, make_chunk):
    reranker = build({"good": "9", "meh": "4", "bad": "1"}, concurrency=2)
    chunks = [make_chunk("c1", "bad"), make_chunk("c2", "good"), make_chunk("c3", "meh")]
    out = reranker.rerank("q", chunks, top_k=3)
    assert [c.chunk_id for c in out] == ["c2", "c3", "c1"]
    assert out[0].score == pytest.approx(0.9)  # 0-10 reply normalized to 0-1


def test_unscorable_chunks_keep_retrieval_order_instead_of_being_dropped(build, make_chunk):
    # A model that ignores "reply with only the number" must not cost us recall.
    reranker = build({"good": "8", "chatty": "I'm unable to rate that."})
    chunks = [make_chunk("c1", "chatty", score=0.5), make_chunk("c2", "good")]
    out = reranker.rerank("q", chunks, top_k=5)
    assert [c.chunk_id for c in out] == ["c2", "c1"]
    assert out[1].score == 0.5  # fell back to its retrieval score


def test_out_of_range_score_is_clamped(build, make_chunk):
    reranker = build({"x": "99"})
    out = reranker.rerank("q", [make_chunk("c1", "x")], top_k=1)
    assert out[0].score == 1.0


def test_respects_top_k(build, make_chunk):
    reranker = build({"a": "9", "b": "8", "c": "7"})
    chunks = [make_chunk("c1", "a"), make_chunk("c2", "b"), make_chunk("c3", "c")]
    out = reranker.rerank("q", chunks, top_k=2)
    assert [c.chunk_id for c in out] == ["c1", "c2"]


def test_empty_candidates(build):
    assert build({}).rerank("q", [], top_k=5) == []


def test_ollama_defaults_to_reasoning_off(monkeypatch):
    # Thinking burns the token budget and returns empty content -> no score.
    captured = {}

    def _capture(provider, model, secrets, **kwargs):
        captured.update(kwargs)
        return _FakeLLM({})

    monkeypatch.setattr(rr, "build_chat_chain", _capture)
    rr.build_reranker(RerankCfg(provider="ollama", model="m"), Secrets())
    assert captured["extra"]["reasoning"] is False


def test_explicit_reasoning_is_not_overridden(monkeypatch):
    captured = {}

    def _capture(provider, model, secrets, **kwargs):
        captured.update(kwargs)
        return _FakeLLM({})

    monkeypatch.setattr(rr, "build_chat_chain", _capture)
    rr.build_reranker(
        RerankCfg(provider="ollama", model="m", extra={"reasoning": True}), Secrets()
    )
    assert captured["extra"]["reasoning"] is True


# --- failover, and what it means for the closed-domain gate ------------------
#
# `retrieval.min_relevance` refuses any question whose best chunk scores below
# it, on the *reranker's* scale. So losing the reranker doesn't just worsen
# ordering — it decides whether questions get answered at all.


class _Boom(rr.Reranker):
    def rerank(self, query, chunks, top_k):
        raise RuntimeError("rerank endpoint down")


class _Fixed(rr.Reranker):
    def __init__(self, score: float) -> None:
        self.score = score

    def rerank(self, query, chunks, top_k):
        return rr._mark(
            [
                rr.RetrievedChunk(
                    chunk_id=c.chunk_id, text=c.text, source=c.source,
                    score=self.score, retriever=c.retriever, metadata=c.metadata,
                )
                for c in chunks[:top_k]
            ],
            True,
        )


def test_failover_uses_the_next_reranker(make_chunk):
    chain = rr.FallbackReranker([_Boom(), _Fixed(0.8)], ["dead", "backup"])
    out = chain.rerank("q", [make_chunk("c1")], top_k=1)
    assert out[0].score == 0.8
    assert out[0].metadata[rr.CALIBRATED] is True


def test_all_rerankers_down_marks_scores_uncalibrated(make_chunk):
    """The scores that come back are raw retrieval similarities. Letting the
    0-1 relevance threshold read them would refuse whole conversations for a
    reason no log would explain."""
    chain = rr.FallbackReranker([_Boom(), _Boom()], ["a", "b"])
    out = chain.rerank("q", [make_chunk("c1", score=0.42)], top_k=1)
    assert out[0].score == 0.42
    assert out[0].metadata[rr.CALIBRATED] is False


def test_noop_reranker_is_never_treated_as_calibrated(make_chunk):
    out = rr.NoOpReranker().rerank("q", [make_chunk("c1", score=0.9)], top_k=1)
    assert out[0].metadata[rr.CALIBRATED] is False


def test_generative_scores_are_calibrated(build, make_chunk):
    out = build({"x": "8"}).rerank("q", [make_chunk("c1", "x")], top_k=1)
    assert out[0].metadata[rr.CALIBRATED] is True


def test_generative_reranker_that_scores_nothing_is_uncalibrated(build, make_chunk):
    """Every scoring call failed, so these are retrieval scores wearing a
    reranker's name."""
    out = build({"x": "no idea"}).rerank("q", [make_chunk("c1", "x", score=0.3)], top_k=1)
    assert out[0].metadata[rr.CALIBRATED] is False
