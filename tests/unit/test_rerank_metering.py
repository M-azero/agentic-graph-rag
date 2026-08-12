"""Retrieval spend reaches the bill.

The token quota covered the agent, the critic and the reviser — everything that
went through LangGraph's config. It did not cover the reranker, which is not
part of the graph: it runs inside a tool, and under a generative provider it is
one model call per candidate (`retrieval.candidate_k`, 24 in the shipped
production profile). That made retrieval the largest unmetered line on a
question, and the closed-domain probe spent it *before* the answer existed — so
a question refused as off-topic cost real money and charged nothing.

Two things have to hold, and they fail independently:
  - the meter reaches the reranker at all (a ContextVar across two thread pools);
  - the arithmetic survives that concurrency (`+=` is not atomic).
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from graphrag.config.settings import RerankCfg, Secrets
from graphrag.retrieval import reranker as rr
from graphrag.usage.meter import TokenMeter, active_meter, meter_config, use_meter


class _CountingLLM:
    """Reports usage the way an OpenAI-compatible provider does."""

    def __init__(self, tokens_in: int = 10, tokens_out: int = 2) -> None:
        self.tokens = (tokens_in, tokens_out)
        self.seen_meter: list[bool] = []

    def invoke(self, prompt: str, config: dict | None = None, **kwargs):
        callbacks = (config or {}).get("callbacks") or []
        self.seen_meter.append(any(isinstance(c, TokenMeter) for c in callbacks))
        for handler in callbacks:
            message = SimpleNamespace(
                usage_metadata={"input_tokens": self.tokens[0],
                                "output_tokens": self.tokens[1]}
            )
            response = SimpleNamespace(
                generations=[[SimpleNamespace(message=message, text="7")]],
                llm_output={},
            )
            _run(handler.on_llm_end(response))
        return SimpleNamespace(content="7")


def _run(coro):
    """Drive one callback coroutine, as LangChain's sync manager does."""
    import asyncio

    asyncio.run(coro)


@pytest.fixture
def reranker(monkeypatch):
    def _build(llm, **cfg_kwargs):
        monkeypatch.setattr(rr, "build_chat_chain", lambda *a, **k: llm)
        return rr.build_reranker(
            RerankCfg(provider="ollama", model="fake", **cfg_kwargs), Secrets()
        )

    return _build


def test_the_meter_reaches_every_scoring_call(reranker, make_chunk):
    """Across the reranker's own thread pool: a bare `pool.map` runs workers
    under an empty context, and the whole rerank goes unmetered."""
    llm = _CountingLLM()
    rank = reranker(llm, concurrency=4)
    chunks = [make_chunk(f"c{i}", f"doc {i}") for i in range(8)]

    meter = TokenMeter()
    with use_meter(meter):
        rank.rerank("q", chunks, top_k=8)

    assert llm.seen_meter == [True] * 8
    assert meter.totals == (80, 16)  # 8 candidates x (10 in, 2 out)


def test_concurrent_scoring_charges_every_candidate(reranker, make_chunk):
    llm = _CountingLLM(tokens_in=1, tokens_out=1)
    rank = reranker(llm, concurrency=8)
    chunks = [make_chunk(f"c{i}", f"doc {i}") for i in range(200)]

    meter = TokenMeter()
    with use_meter(meter):
        rank.rerank("q", chunks, top_k=200)

    assert meter.totals == (200, 200)


def test_mutation_and_readout_are_serialized():
    """Every path that touches a counter does so under the meter's lock.

    Deliberately structural rather than statistical. `self.input_tokens += n` is
    a load-add-store and two threads landing in it together lose an update, but
    that window is a few bytecodes wide: a hammer-it-with-threads test passes
    just as happily with the lock deleted, which is worse than no test at all —
    it would certify the race as fixed the moment someone removed the fix.
    Asserting the lock is actually entered is the part that can fail.
    """
    meter = TokenMeter()
    entries: list[str] = []
    real = meter._lock

    class _Watched:
        def __enter__(self):
            entries.append("in")
            return real.__enter__()

        def __exit__(self, *args):
            return real.__exit__(*args)

    meter._lock = _Watched()

    reported = SimpleNamespace(
        generations=[[SimpleNamespace(
            message=SimpleNamespace(
                usage_metadata={"input_tokens": 3, "output_tokens": 4}
            ),
            text="ok",
        )]],
        llm_output={},
    )
    unreported = SimpleNamespace(
        generations=[[SimpleNamespace(message=None, text="some streamed text")]],
        llm_output={},
    )

    _run(meter.on_chat_model_start({}, [[SimpleNamespace(content="hi")]]))
    _run(meter.on_llm_start({}, ["hi"]))
    _run(meter.on_llm_end(reported))
    _run(meter.on_llm_end(unreported))   # the estimate branch counts too
    _ = meter.totals                     # and so does the read

    assert len(entries) == 5, "a counter was touched outside the lock"


def test_nothing_bound_means_nothing_charged(reranker, make_chunk):
    """The CLI, scripts and tests call retrievers directly and have no bill."""
    llm = _CountingLLM()
    rank = reranker(llm)
    rank.rerank("q", [make_chunk("c1", "doc 1")], top_k=1)
    assert llm.seen_meter == [False]


# -- the binding primitive ----------------------------------------------------

def test_meter_config_is_empty_when_unbound():
    assert meter_config() == {}
    assert meter_config({"tags": ["x"]}) == {"tags": ["x"]}


def test_meter_config_appends_rather_than_replaces():
    meter = TokenMeter()
    other = object()
    with use_meter(meter):
        config = meter_config({"callbacks": [other]})
    assert config["callbacks"] == [other, meter]


def test_binding_is_restored_on_exit():
    outer, inner = TokenMeter(), TokenMeter()
    with use_meter(outer):
        with use_meter(inner):
            assert active_meter() is inner
        assert active_meter() is outer
    assert active_meter() is None


def test_the_meter_is_visible_from_a_worker_thread():
    """`copy_context().run` is what carries it; a plain thread does not."""
    meter = TokenMeter()
    seen: list = []

    def _worker():
        seen.append(active_meter())

    with use_meter(meter):
        from contextvars import copy_context

        thread = threading.Thread(target=copy_context().run, args=(_worker,))
        thread.start()
        thread.join()

    assert seen == [meter]
