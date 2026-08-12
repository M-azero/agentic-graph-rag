"""Compiled agents are reused per tenant, and sources still don't bleed.

Reusing one compiled graph across questions moves the per-query source
collector out of the tool closures and into a ContextVar. That trade is only
safe if two guarantees hold, so both are pinned here: the cache actually hits
(and stays bounded), and concurrent queries never see each other's sources.
"""

import asyncio

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from graphrag.agent.graph import _AGENT_CACHE_SIZE, AgentRunner
from graphrag.agent.tools import _collect, collect_sources
from graphrag.core.types import AnswerStyle, RetrievedChunk


def _chunk(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, text=f"text {chunk_id}", source="doc.pdf",
        score=1.0, retriever="test",
    )


class _Model(FakeListChatModel):
    """Stand-in chat model. `create_react_agent` type-checks for a real
    `BaseChatModel` and binds tools to it; nothing here invokes it."""

    def __init__(self) -> None:
        super().__init__(responses=["unused"])

    def bind_tools(self, tools, **kwargs):
        return self


def _runner(**kwargs) -> AgentRunner:
    return AgentRunner(
        _Model(), vector=None, hybrid=None, graph=None, embedder=None, **kwargs
    )


# --------------------------------------------------------------------------
# Compilation is cached
# --------------------------------------------------------------------------


def test_same_style_and_model_reuse_one_compiled_agent():
    runner = _runner()
    first = runner.session("q1", style="concise")._agent
    second = runner.session("q2", style="concise")._agent
    assert first is second


def test_unknown_style_folds_onto_the_default_key():
    """Style comes from request input. If it keyed the cache raw, every junk
    value would compile (and retain) another agent."""
    runner = _runner()
    baseline = runner.session("q", style="detailed")._agent
    for junk in ("banana", "", "DETAILED-ish", "../etc"):
        assert runner.session("q", style=junk)._agent is baseline
    assert len(runner._agents) == 1


def test_each_real_style_gets_its_own_agent():
    runner = _runner()
    agents = {runner.session("q", style=s.value)._agent for s in AnswerStyle}
    assert len(agents) == len(AnswerStyle)


def test_model_override_does_not_reuse_the_default_agent():
    runner = _runner()
    default = runner.session("q", style="concise")._agent
    override = runner.session("q", style="concise", model=_Model())._agent
    assert override is not default


def test_cache_is_bounded_and_evicts_lru():
    runner = _runner()
    models = [_Model() for _ in range(_AGENT_CACHE_SIZE + 3)]
    for model in models:
        runner.session("q", style="concise", model=model)
    assert len(runner._agents) == _AGENT_CACHE_SIZE
    # The oldest entries went first. Key is (preset, style, model id, remember).
    live = {key[2] for key in runner._agents}
    assert id(models[0]) not in live
    assert id(models[-1]) in live


# --------------------------------------------------------------------------
# Source collection stays per-query
# --------------------------------------------------------------------------


def test_collect_outside_a_session_is_a_no_op():
    _collect([_chunk("orphan")])  # must not raise


def test_sink_dedupes_by_chunk_id():
    with collect_sources() as sink:
        _collect([_chunk("a"), _chunk("b")])
        _collect([_chunk("b"), _chunk("c")])
        assert [c.chunk_id for c in sink.chunks] == ["a", "b", "c"]


def test_nested_scopes_restore_the_outer_sink():
    with collect_sources() as outer:
        _collect([_chunk("outer-1")])
        with collect_sources() as inner:
            _collect([_chunk("inner-1")])
        assert [c.chunk_id for c in inner.chunks] == ["inner-1"]
        _collect([_chunk("outer-2")])
    assert [c.chunk_id for c in outer.chunks] == ["outer-1", "outer-2"]


@pytest.mark.asyncio
async def test_concurrent_queries_do_not_see_each_other_s_sources():
    """The property the whole refactor rests on. Each asyncio task gets its own
    copy of the context, so interleaved queries collect independently."""

    async def query(tag: str, sources: dict) -> None:
        with collect_sources() as sink:
            for i in range(5):
                _collect([_chunk(f"{tag}-{i}")])
                await asyncio.sleep(0)  # force interleaving
            sources[tag] = [c.chunk_id for c in sink.chunks]

    collected: dict[str, list[str]] = {}
    await asyncio.gather(*(query(tag, collected) for tag in ("a", "b", "c")))

    for tag, ids in collected.items():
        assert ids == [f"{tag}-{i}" for i in range(5)]


@pytest.mark.asyncio
async def test_sources_survive_in_threads_the_way_langgraph_runs_sync_tools():
    """LangGraph and LangChain both submit under `copy_context()`, which is why
    a sync tool on a worker thread still records into the right query."""
    import contextvars

    with collect_sources() as sink:
        ctx = contextvars.copy_context()
        await asyncio.get_running_loop().run_in_executor(
            None, lambda: ctx.run(_collect, [_chunk("from-thread")])
        )
        assert [c.chunk_id for c in sink.chunks] == ["from-thread"]
