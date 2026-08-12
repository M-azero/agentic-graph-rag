"""How much to retrieve, as opposed to which retriever to use.

*Which* retriever has always been the agent's decision — that is what the tools
are. *How much* was frozen in constructor arguments: `graph_hops` on the
graph-augmented retriever, `candidate_k` on the hybrid one, the entity-seed
count hardcoded outright. So a question that needed a wider net had no way to
ask for one, and the only escalation available was asking the same question
again at the same depth.

A `RetrievalPlan` unfreezes those, and travels in a ContextVar beside the
source sink for the same reason: the tools live inside a compiled graph that
outlives the query, so anything per-question has to arrive through the call
context rather than through a constructor.

Deliberately **not** exposed as tool arguments. Adding `k` to
`hybrid_search(query, k)` would put retrieval depth in the tool schema, where a
7B model would pick it badly and pay for the wider schema on every turn. The
graph sets the plan; the model never sees it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class RetrievalPlan:
    """One question's retrieval budget.

    Defaults reproduce the behaviour that was previously hardcoded, so an
    unplanned run retrieves exactly what it always did.
    """

    top_k: int = 8
    candidate_k: int = 24
    graph_hops: int = 2
    entity_seeds: int = 5
    rerank: bool = True

    def widened(self, factor: float = 2.0) -> RetrievalPlan:
        """A larger budget for a second pass, bounded.

        Escalation is for questions the first pass under-covered, so the net
        gets wider and the graph walk one hop deeper. The caps matter: each
        extra candidate is an LLM call under a generative reranker, so an
        unbounded widen turns one slow answer into a much more expensive one.
        """
        return replace(
            self,
            top_k=min(int(self.top_k * factor), 24),
            candidate_k=min(int(self.candidate_k * factor), 48),
            graph_hops=min(self.graph_hops + 1, 4),
            entity_seeds=min(int(self.entity_seeds * factor), 12),
        )


DEFAULT_PLAN = RetrievalPlan()

_PLAN: ContextVar[RetrievalPlan | None] = ContextVar("graphrag_retrieval_plan", default=None)


def active_plan() -> RetrievalPlan | None:
    """The plan bound to the query in flight, or None if none is bound.

    Retrievers want this rather than `current_plan()`: outside a planned run
    they must keep using their own configured depth, because a deployment that
    tuned `graph_hops` in config would otherwise be silently overridden by the
    plan's generic default. A bound plan wins entirely; an absent one changes
    nothing.
    """
    return _PLAN.get()


def current_plan() -> RetrievalPlan:
    """The plan for the query in flight, or the defaults outside one."""
    return _PLAN.get() or DEFAULT_PLAN


@contextmanager
def use_plan(plan: RetrievalPlan | None) -> Iterator[RetrievalPlan]:
    """Bind a plan for the duration of one retrieval pass.

    Must be entered by whatever *invokes* the agent, not by an earlier graph
    node: LangGraph runs each node under `copy_context()`, so a plan set in one
    node is invisible in the next.
    """
    active = plan or DEFAULT_PLAN
    token = _PLAN.set(active)
    try:
        yield active
    finally:
        _PLAN.reset(token)
