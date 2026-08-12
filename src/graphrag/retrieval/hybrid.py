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
from graphrag.retrieval.reranker import Reranker
from graphrag.retrieval.vector import VectorRetriever
from graphrag.storage.graph.base import GraphStore


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
            (self._vector.retrieve, query, candidate_k),
            (self._graph_aug.retrieve, query, candidate_k),
            (self._graph.fulltext_chunks, query, candidate_k),
        )
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(copy_context().run, *leg) for leg in legs]
            lists: list[list[RetrievedChunk]] = [f.result() for f in futures]
        fused = reciprocal_rank_fusion(lists)[:candidate_k]
        if plan is not None and not plan.rerank:
            return fused[:k]
        return self._reranker.rerank(query, fused, k)
