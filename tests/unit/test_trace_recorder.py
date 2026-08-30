"""The step recorder: inert when unbound, correct when bound, and bounded."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

import pytest

from graphrag.trace import steps as registry
from graphrag.trace.recorder import (
    NULL_STEP,
    OK,
    SKIPPED,
    Tracer,
    active_tracer,
    step,
    use_tracer,
)


def _tracer(**kw) -> Tracer:
    return Tracer("t", registry.QUERY, owner="u", **kw)


# -- the property that makes this safe to leave in the hot path ---------------


def test_unbound_step_is_a_no_op():
    assert active_tracer() is None
    with step(registry.RETRIEVAL_FUSE, lists=3) as s:
        assert s is NULL_STEP
        # Every method still callable, so instrumentation never branches.
        s.output(candidates=24)
        s.meta(provider="none")
        s.skip("nothing to do")


def test_binding_is_scoped_to_the_block():
    tracer = _tracer()
    with use_tracer(tracer):
        assert active_tracer() is tracer
    assert active_tracer() is None


# -- structure ----------------------------------------------------------------


def test_steps_nest_under_the_step_that_opened_them():
    tracer = _tracer()
    with (
        use_tracer(tracer),
        step(registry.AGENT_RUN),
        step(registry.TOOL_CALL),
        step(registry.RETRIEVAL_VECTOR),
    ):
        pass

    run, tool, vector = tracer.steps
    assert run.parent_id is None
    assert tool.parent_id == run.id
    assert vector.parent_id == tool.id


def test_concurrent_legs_nest_under_their_own_parent_not_a_sibling():
    """The case that motivates a per-context parent rather than a stack: the
    hybrid retriever runs three legs at once, each under its own context copy."""
    tracer = _tracer()
    names = (
        registry.RETRIEVAL_VECTOR,
        registry.RETRIEVAL_GRAPH,
        registry.RETRIEVAL_KEYWORD,
    )

    def leg(name):
        with step(name):
            return None

    with use_tracer(tracer), step(registry.TOOL_CALL) as parent:
        parent_id = parent.id
        with ThreadPoolExecutor(max_workers=3) as pool:
            # `copy_context()` in the *submitting* thread, exactly as
            # HybridRetriever does. Calling it inside the worker would copy the
            # worker's empty context and trace nothing at all — which is what
            # the comment at that call site is warning about.
            futures = [pool.submit(copy_context().run, leg, n) for n in names]
            [f.result() for f in futures]

    legs = [s for s in tracer.steps if s.name in names]
    assert len(legs) == 3
    assert {s.parent_id for s in legs} == {parent_id}


def test_status_records_ok_error_and_skipped():
    tracer = _tracer()
    with use_tracer(tracer):
        with step(registry.GUARD_INPUT) as s:
            s.skip("safety.enabled is false")
        with pytest.raises(ValueError), step(registry.RETRIEVAL_RERANK):
            raise ValueError("provider down: key=sk-abcdefghijklmnop")
        with step(registry.BILLING):
            pass

    guard, rerank, billing = tracer.steps
    assert guard.status == SKIPPED
    assert guard.meta["reason"] == "safety.enabled is false"
    assert rerank.status == "error"
    assert billing.status == OK
    # The exception propagated: tracing observes, it does not swallow.
    assert "provider down" in rerank.error
    # ...and the key that rode along in the message did not.
    assert "sk-abcdefghijklmnop" not in rerank.error


def test_durations_are_recorded():
    tracer = _tracer()
    with use_tracer(tracer), step(registry.AGENT_RUN):
        pass
    assert tracer.steps[0].duration_ms >= 0


# -- bounds -------------------------------------------------------------------


def test_long_text_is_truncated():
    tracer = _tracer(max_preview_chars=100)
    with use_tracer(tracer), step(registry.AGENT_RUN, question="x" * 5000):
        pass
    text = tracer.steps[0].input["question"]
    assert len(text) < 200
    assert text.endswith("…[truncated]")


def test_include_text_off_withholds_the_document_but_keeps_the_shape():
    tracer = _tracer(include_text=False)
    with use_tracer(tracer), step(registry.AGENT_RUN, question="y" * 400) as s:
        s.output(sources=3)
    recorded = tracer.steps[0]
    assert "y" * 50 not in recorded.input["question"]
    assert "400 chars withheld" in recorded.input["question"]
    # Counts are not text and must survive, or the trace says nothing at all.
    assert recorded.output["sources"] == 3


def test_secrets_are_scrubbed_from_payloads():
    tracer = _tracer()
    with use_tracer(tracer), step(registry.AGENT_RUN, question="key=sk-livekey1234567890"):
        pass
    assert "sk-livekey1234567890" not in tracer.steps[0].input["question"]


def test_step_count_is_capped_and_says_so():
    tracer = _tracer(max_steps=5)
    with use_tracer(tracer):
        for _ in range(40):
            with step(registry.TOOL_CALL):
                pass
    names = [s.name for s in tracer.steps]
    assert len(names) == 6  # the cap, plus one marker
    assert names[-1] == registry.TRUNCATED


def test_nested_payloads_are_bounded_by_depth_and_width():
    tracer = _tracer()
    deep = {"a": {"b": {"c": {"d": {"e": 1}}}}}
    with use_tracer(tracer), step(registry.RETRIEVAL_FUSE, tree=deep, wide=list(range(500))):
        pass
    recorded = tracer.steps[0].input
    assert len(recorded["wide"]) <= 24
    assert "dict" in str(recorded["tree"])  # collapsed rather than walked forever


# -- streaming ----------------------------------------------------------------


def test_drain_replays_a_running_step_then_advances():
    tracer = _tracer()
    with use_tracer(tracer):
        with step(registry.AGENT_RUN):
            # Mid-run: the step is visible but not settled, so a drain must not
            # advance past it — the UI needs to see it finish.
            first = tracer.drain()
            assert [s.name for s in first] == [registry.AGENT_RUN]
            assert first[0].status == "running"
            assert [s.name for s in tracer.drain()] == [registry.AGENT_RUN]
        settled = tracer.drain()
        assert settled[0].status == OK
    assert tracer.drain() == []


def test_public_never_echoes_the_owner():
    tracer = _tracer()
    with use_tracer(tracer), step(registry.AGENT_RUN):
        pass
    assert tracer.to_dict()["owner"] == "u"
    assert "owner" not in tracer.public()
