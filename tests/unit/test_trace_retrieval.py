"""The retrieval instrumentation, through the real `HybridRetriever`.

This is the part of the pipeline that is hardest to trace and the most worth
seeing: three legs running at once on a thread pool, fused, then reranked. The
endpoint tests stub the query service, so nothing there exercises any of it.
"""

from __future__ import annotations

from graphrag.core.types import RetrievedChunk
from graphrag.retrieval.hybrid import HybridRetriever
from graphrag.retrieval.reranker import NoOpReranker
from graphrag.trace import steps as registry
from graphrag.trace.recorder import Tracer, step, use_tracer


def _chunk(cid: str, score: float, retriever: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        text=f"passage {cid} about warehouse robots",
        source=f"{cid}.pdf",
        score=score,
        retriever=retriever,
    )


class _Vector:
    def retrieve(self, query, k):
        return [_chunk("a", 0.9, "vector"), _chunk("b", 0.7, "vector")]


class _GraphAug:
    def retrieve(self, query, k):
        return [_chunk("b", 0.6, "graph"), _chunk("c", 0.5, "graph")]


class _Graph:
    def fulltext_chunks(self, query, k):
        return [_chunk("c", 0.4, "keyword")]


class _Reranker(NoOpReranker):
    """Reverses the fused order, so the trace has a real ordering change to
    report rather than an identity transform that would pass either way."""

    calibrated = True

    def rerank(self, query, chunks, top_k):
        flipped = list(reversed(chunks))[:top_k]
        for c in flipped:
            c.metadata["rerank_calibrated"] = True
        return flipped


def _retriever(reranker=None) -> HybridRetriever:
    return HybridRetriever(
        _Vector(), _GraphAug(), _Graph(), reranker or _Reranker(), candidate_k=24
    )


def _run(retriever, k=3):
    tracer = Tracer("t", registry.QUERY, owner="u")
    with use_tracer(tracer), step(registry.TOOL_CALL, tool="hybrid_search"):
        chunks = retriever.retrieve("warehouse robots", k)
    return tracer, chunks


def test_all_five_retrieval_blocks_are_recorded():
    tracer, _ = _run(_retriever())
    names = [s.name for s in tracer.steps]
    assert names.count(registry.RETRIEVAL_VECTOR) == 1
    assert names.count(registry.RETRIEVAL_GRAPH) == 1
    assert names.count(registry.RETRIEVAL_KEYWORD) == 1
    assert names.count(registry.RETRIEVAL_FUSE) == 1
    assert names.count(registry.RETRIEVAL_RERANK) == 1


def test_the_three_legs_nest_under_the_tool_call_that_ran_them():
    """Each leg runs on its own context copy. If the parent were tracked with a
    stack instead, they would nest under each other at random."""
    tracer, _ = _run(_retriever())
    tool = next(s for s in tracer.steps if s.name == registry.TOOL_CALL)
    legs = [
        s
        for s in tracer.steps
        if s.name
        in (registry.RETRIEVAL_VECTOR, registry.RETRIEVAL_GRAPH, registry.RETRIEVAL_KEYWORD)
    ]
    assert {s.parent_id for s in legs} == {tool.id}


def test_each_leg_reports_what_it_actually_found():
    tracer, _ = _run(_retriever())
    vector = next(s for s in tracer.steps if s.name == registry.RETRIEVAL_VECTOR)
    assert vector.output["results"] == 2
    assert [t["chunk_id"] for t in vector.output["top"]] == ["a", "b"]

    keyword = next(s for s in tracer.steps if s.name == registry.RETRIEVAL_KEYWORD)
    assert keyword.output["results"] == 1


def test_fusion_reports_its_inputs_and_the_rrf_scores_it_produced():
    tracer, _ = _run(_retriever())
    fuse = next(s for s in tracer.steps if s.name == registry.RETRIEVAL_FUSE)
    assert fuse.input == {"vector": 2, "graph": 2, "keyword": 1}
    # Three distinct chunks across five hits: b and c were found twice.
    assert fuse.output["candidates"] == 3
    scores = {t["chunk_id"]: t["rrf"] for t in fuse.output["top"]}
    assert set(scores) == {"a", "b", "c"}
    # b appears in two lists, so RRF must rank it above a, which appears in one
    # at a better rank. That relationship is the whole reason fusion is a step.
    assert scores["b"] > scores["a"]


def test_rerank_records_the_order_it_changed():
    tracer, chunks = _run(_retriever())
    rerank = next(s for s in tracer.steps if s.name == registry.RETRIEVAL_RERANK)
    assert rerank.input["candidates"] == 3
    assert rerank.output["calibrated"] is True
    # `was_rank` is what makes the block legible: without it the reader sees
    # scores and cannot tell whether reranking moved anything.
    ranks = [t["was_rank"] for t in rerank.output["top"]]
    assert ranks == sorted(ranks, reverse=True)
    assert [c.chunk_id for c in chunks] == [t["chunk_id"] for t in rerank.output["top"]]


def test_a_failing_leg_is_recorded_and_still_fails_the_retrieval():
    """Tracing observes; it must not turn a broken retriever into a quiet one."""

    class _Broken:
        def retrieve(self, query, k):
            raise RuntimeError("neo4j is down")

    retriever = HybridRetriever(_Vector(), _Broken(), _Graph(), _Reranker(), candidate_k=24)
    tracer = Tracer("t", registry.QUERY, owner="u")
    with use_tracer(tracer):
        try:
            retriever.retrieve("warehouse robots", 3)
        except RuntimeError as exc:
            assert "neo4j is down" in str(exc)
        else:
            raise AssertionError("the failure was swallowed")

    graph_leg = next(s for s in tracer.steps if s.name == registry.RETRIEVAL_GRAPH)
    assert graph_leg.status == "error"
    assert "neo4j is down" in graph_leg.error


def test_retrieval_is_unchanged_when_nothing_is_traced():
    """The property the whole design rests on."""
    traced_tracer, traced = _run(_retriever())
    untraced = _retriever().retrieve("warehouse robots", 3)
    assert [c.chunk_id for c in traced] == [c.chunk_id for c in untraced]
    assert [round(c.score, 6) for c in traced] == [round(c.score, 6) for c in untraced]
    assert traced_tracer.steps  # ...and the traced run did record something
