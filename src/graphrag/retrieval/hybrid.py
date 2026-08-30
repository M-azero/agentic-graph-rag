"""Hybrid retriever: run vector + graph-augmented + keyword search, fuse the
rankings with RRF, then rerank the fused candidates. This is the strong default
the agent's `hybrid_search` tool calls."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

from graphrag.core.types import RetrievedChunk
from graphrag.retrieval.base import Retriever
from graphrag.retrieval.fusion import reciprocal_rank_fusion
from graphrag.retrieval.graph_augmented import GraphAugmentedRetriever
from graphrag.retrieval.plan import active_plan
from graphrag.retrieval.reranker import CALIBRATED, Reranker, describe
from graphrag.retrieval.vector import VectorRetriever
from graphrag.storage.graph.base import GraphStore
from graphrag.trace import step
from graphrag.trace import steps as trace_steps


class HybridRetriever(Retriever):
    def __init__(
        self,
        vector: VectorRetriever,
        graph_aug: GraphAugmentedRetriever,
        graph: GraphStore,
        reranker: Reranker,
        candidate_k: int = 24,
    ) -> None:
        self._vector = vector
        self._graph_aug = graph_aug
        self._graph = graph
        self._reranker = reranker
        self._candidate_k = candidate_k

    def _leg(self, name: str, fn, query: str, candidate_k: int) -> list[RetrievedChunk]:
        """One retrieval leg, traced. Called inside the pool, under this leg's
        own context copy — which is what makes the step nest under the tool call
        that spawned it rather than under whichever sibling started last."""
        with step(name, query=query, candidate_k=candidate_k) as s:
            chunks = fn(query, candidate_k)
            s.output(
                results=len(chunks),
                top=[
                    {"chunk_id": c.chunk_id, "source": c.source, "score": round(c.score, 4)}
                    for c in chunks[:5]
                ],
            )
            return chunks

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        plan = active_plan()
        candidate_k = plan.candidate_k if plan else self._candidate_k

        # The three retrievals are independent I/O (embedding call + two Neo4j
        # round-trips); running them together cuts the retrieval phase to the
        # slowest leg instead of the sum.
        #
        # Each leg gets its OWN context copy. The copy is needed at all so the
        # workers inherit the retrieval plan and the source sink — a bare
        # submit runs them under an empty context, silently retrieving at the
        # default depth. But one shared `Context` cannot be entered twice at
        # once ("cannot enter context: ... is already entered"), and with three
        # concurrent legs that is exactly what would happen. Copies are cheap,
        # and the values inside are shared object references, so the sink still
        # accumulates across all three.
        legs = (
            (trace_steps.RETRIEVAL_VECTOR, self._vector.retrieve),
            (trace_steps.RETRIEVAL_GRAPH, self._graph_aug.retrieve),
            (trace_steps.RETRIEVAL_KEYWORD, self._graph.fulltext_chunks),
        )
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(copy_context().run, self._leg, name, fn, query, candidate_k)
                for name, fn in legs
            ]
            lists: list[list[RetrievedChunk]] = [f.result() for f in futures]

        with step(
            trace_steps.RETRIEVAL_FUSE,
            vector=len(lists[0]), graph=len(lists[1]), keyword=len(lists[2]),
        ) as s:
            fused = reciprocal_rank_fusion(lists)[:candidate_k]
            s.output(
                candidates=len(fused),
                top=[
                    {"chunk_id": c.chunk_id, "source": c.source, "rrf": round(c.score, 5)}
                    for c in fused[:8]
                ],
            )

        if plan is not None and not plan.rerank:
            with step(trace_steps.RETRIEVAL_RERANK, candidates=len(fused)) as s:
                s.skip("the retrieval plan for this run disabled reranking")
            return fused[:k]

        # Traced here rather than inside the rerankers: there are five
        # implementations plus a fallback chain, and everything worth showing —
        # which provider answered, whether the scores came back calibrated, how
        # the order changed — is visible from this side of the call.
        with step(
            trace_steps.RETRIEVAL_RERANK,
            provider=describe(self._reranker),
            candidates=len(fused),
            top_k=k,
        ) as s:
            ranked = self._reranker.rerank(query, fused, k)
            before = {c.chunk_id: i for i, c in enumerate(fused)}
            s.output(
                results=len(ranked),
                calibrated=bool(ranked and ranked[0].metadata.get(CALIBRATED, True)),
                top=[
                    {
                        "chunk_id": c.chunk_id,
                        "source": c.source,
                        "score": round(c.score, 4),
                        # Where this chunk sat before reranking — the whole
                        # point of the stage, and invisible from the scores.
                        "was_rank": before.get(c.chunk_id),
                        "snippet": c.text[:280],
                    }
                    for c in ranked[:8]
                ],
            )
            return ranked
