"""Graph-augmented retrieval: find the entities a query is about, then pull the
chunks that mention them *and their graph neighborhood*, scored by how many hops
an entity sits from a seed. This is what makes it *graph* RAG rather than plain
vector RAG — it follows relationships, not just similarity."""

from __future__ import annotations

from graphrag.core.types import RetrievedChunk
from graphrag.retrieval.base import Retriever
from graphrag.retrieval.plan import active_plan
from graphrag.storage.graph.base import GraphStore

_DEFAULT_SEEDS = 5


class GraphAugmentedRetriever(Retriever):
    def __init__(self, graph: GraphStore, hops: int = 2) -> None:
        self._graph = graph
        self._hops = hops

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        # A bound plan wins entirely; without one nothing changes and the
        # configured depth stands.
        plan = active_plan()
        hops = plan.graph_hops if plan else self._hops
        seeds = self._graph.fulltext_entities(
            query, k=plan.entity_seeds if plan else _DEFAULT_SEEDS
        )
        if not seeds:
            return []
        return self._graph.expand_chunks(seeds, hops=hops, limit=k)
