"""Retrieval depth is a per-query decision, not a constructor argument.

The load-bearing property is that this changed nothing by default: an unplanned
run must retrieve exactly what it retrieved before the plan existed. The golden
test below is what makes that checkable rather than assumed.
"""

from graphrag.agent.tools import ToolContext, build_tools
from graphrag.core.types import RetrievedChunk
from graphrag.retrieval.hybrid import HybridRetriever
from graphrag.retrieval.plan import (
    DEFAULT_PLAN,
    RetrievalPlan,
    active_plan,
    current_plan,
    use_plan,
)


class _Graph:
    def __init__(self):
        self.seed_ks: list[int] = []
        self.expand: list[tuple] = []
        self.fulltext_ks: list[int] = []

    def fulltext_entities(self, query, k=5):
        self.seed_ks.append(k)
        return ["acme"]

    def expand_chunks(self, seeds, hops=2, limit=12):
        self.expand.append((hops, limit))
        return []

    def fulltext_chunks(self, query, k=8):
        self.fulltext_ks.append(k)
        return []

    def neighbors(self, entity, hops=2):
        self.hops_seen = hops
        return "n"

    def chunks_for_entities(self, names, limit=12):
        self.entity_limit = limit
        return []


class _Vector:
    def __init__(self):
        self.ks: list[int] = []

    def retrieve(self, query, k):
        self.ks.append(k)
        return []


class _Reranker:
    def __init__(self):
        self.called = 0

    def rerank(self, query, chunks, top_k):
        self.called += 1
        return chunks[:top_k]


# ── the golden test ────────────────────────────────────────────────────────


def test_defaults_reproduce_the_previously_hardcoded_values():
    """These four numbers were frozen in constructors and one literal. If the
    refactor changed any of them, every deployment's retrieval changed with it."""
    assert (
        DEFAULT_PLAN.top_k,
        DEFAULT_PLAN.candidate_k,
        DEFAULT_PLAN.graph_hops,
        DEFAULT_PLAN.entity_seeds,
    ) == (8, 24, 2, 5)
    assert DEFAULT_PLAN.rerank is True


def test_no_plan_bound_means_no_plan_active():
    """Retrievers branch on this: an absent plan must leave configured values
    alone, rather than being silently replaced by the plan's generic default."""
    assert active_plan() is None
    assert current_plan() == DEFAULT_PLAN


def test_an_unplanned_retriever_uses_its_configured_depth():
    graph, reranker = _Graph(), _Reranker()
    from graphrag.retrieval.graph_augmented import GraphAugmentedRetriever

    hybrid = HybridRetriever(
        _Vector(), GraphAugmentedRetriever(graph, hops=3), graph, reranker, candidate_k=17
    )
    hybrid.retrieve("q", 4)
    assert graph.expand[0][0] == 3  # the configured hops, not the plan's 2
    assert graph.fulltext_ks == [17]  # the configured candidate_k
    assert reranker.called == 1


# ── a bound plan wins ──────────────────────────────────────────────────────


def test_a_bound_plan_overrides_construction():
    graph, vector, reranker = _Graph(), _Vector(), _Reranker()
    from graphrag.retrieval.graph_augmented import GraphAugmentedRetriever

    hybrid = HybridRetriever(
        vector, GraphAugmentedRetriever(graph, hops=2), graph, reranker, candidate_k=24
    )
    with use_plan(RetrievalPlan(candidate_k=40, graph_hops=4, entity_seeds=9)):
        hybrid.retrieve("q", 4)

    assert vector.ks == [40]
    assert graph.fulltext_ks == [40]
    assert graph.expand[0][0] == 4
    assert graph.seed_ks == [9]


def test_the_plan_reaches_the_parallel_legs():
    """`HybridRetriever` fans out onto a thread pool, and a bare submit would
    run the legs under an empty context — retrieving at the default depth while
    appearing to honour the plan.

    The legs are made genuinely concurrent with a barrier. Without it they
    finish faster than they overlap, which hid a real bug: one shared `Context`
    cannot be entered twice at once, so a single copy shared across the three
    submits raises "cannot enter context" only under real I/O latency.
    """
    import threading

    barrier = threading.Barrier(3, timeout=5)

    class _Slow(_Graph):
        def fulltext_entities(self, query, k=5):
            barrier.wait()
            return super().fulltext_entities(query, k)

        def fulltext_chunks(self, query, k=8):
            barrier.wait()
            return super().fulltext_chunks(query, k)

    class _SlowVector(_Vector):
        def retrieve(self, query, k):
            barrier.wait()
            return super().retrieve(query, k)

    from graphrag.retrieval.graph_augmented import GraphAugmentedRetriever

    graph, vector = _Slow(), _SlowVector()
    hybrid = HybridRetriever(
        vector, GraphAugmentedRetriever(graph, hops=2), graph, _Reranker(), candidate_k=24
    )
    with use_plan(RetrievalPlan(graph_hops=4, entity_seeds=9)):
        hybrid.retrieve("q", 4)

    # All three read values only visible inside worker threads, while all three
    # were genuinely running at the same time.
    assert graph.expand[0][0] == 4
    assert graph.seed_ks == [9]
    assert vector.ks == [24]


def test_rerank_can_be_switched_off_for_a_pass():
    graph, reranker = _Graph(), _Reranker()
    from graphrag.retrieval.graph_augmented import GraphAugmentedRetriever

    hybrid = HybridRetriever(
        _Vector(), GraphAugmentedRetriever(graph), graph, reranker, candidate_k=24
    )
    with use_plan(RetrievalPlan(rerank=False)):
        hybrid.retrieve("q", 4)
    assert reranker.called == 0


def test_the_plan_unbinds_after_the_block():
    with use_plan(RetrievalPlan(top_k=99)):
        assert current_plan().top_k == 99
    assert active_plan() is None


def test_nested_plans_restore_the_outer_one():
    with use_plan(RetrievalPlan(top_k=10)):
        with use_plan(RetrievalPlan(top_k=20)):
            assert current_plan().top_k == 20
        assert current_plan().top_k == 10


# ── widening ───────────────────────────────────────────────────────────────


def test_widening_grows_the_budget():
    wide = DEFAULT_PLAN.widened()
    assert wide.top_k > DEFAULT_PLAN.top_k
    assert wide.candidate_k > DEFAULT_PLAN.candidate_k
    assert wide.graph_hops == DEFAULT_PLAN.graph_hops + 1


def test_widening_is_bounded():
    """Each extra candidate is an LLM call under a generative reranker, so an
    unbounded widen turns one slow answer into a far more expensive one."""
    plan = DEFAULT_PLAN
    for _ in range(10):
        plan = plan.widened()
    assert plan.top_k <= 24
    assert plan.candidate_k <= 48
    assert plan.graph_hops <= 4
    assert plan.entity_seeds <= 12


def test_plans_are_immutable():
    assert DEFAULT_PLAN.widened() is not DEFAULT_PLAN
    assert DEFAULT_PLAN.top_k == 8


# ── the tools read the plan too ────────────────────────────────────────────


def _tools(graph):
    ctx = ToolContext(
        vector=_Vector(), hybrid=_Hybrid(), graph=graph, embedder=None,
        top_k=8, graph_hops=2,
    )
    return {t.name: t.func for t in build_tools(ctx)}


class _Hybrid:
    def __init__(self):
        self.ks: list[int] = []

    def retrieve(self, query, k):
        self.ks.append(k)
        return [
            RetrievedChunk(
                chunk_id="c1", text="t", source="a.pdf", score=1.0, retriever="test"
            )
        ]


def test_tools_use_the_configured_top_k_without_a_plan():
    graph = _Graph()
    _tools(graph)["fulltext_search"]("term")
    assert graph.fulltext_ks == [8]


def test_tools_follow_a_bound_plan():
    graph = _Graph()
    tools = _tools(graph)
    with use_plan(RetrievalPlan(top_k=15, graph_hops=3)):
        tools["fulltext_search"]("term")
        tools["graph_neighbors"]("Acme")
    assert graph.fulltext_ks == [15]
    assert graph.hops_seen == 3
