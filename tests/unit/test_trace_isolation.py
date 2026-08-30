"""A trace belongs to one tenant. It carries their question and excerpts of
their documents, so the store must not hand it to anyone else — the same rule
`test_job_isolation.py` holds the job store to."""

from __future__ import annotations

from graphrag.trace import steps as registry
from graphrag.trace.store import TraceStore


def _record(trace_id: str, owner: str, **kw) -> dict:
    return {
        "trace_id": trace_id,
        "pipeline": registry.QUERY,
        "owner": owner,
        "status": "ok",
        "title": "what is in the warehouse report",
        "started_at": 1.0,
        "duration_ms": 12.0,
        "steps": [{"id": "s1", "name": registry.AGENT_RUN, "input": {}, "output": {}}],
        **kw,
    }


def test_the_owner_can_read_their_own_trace():
    store = TraceStore()
    store.save(_record("abc", "alice"))
    assert store.get("abc", owner="alice")["trace_id"] == "abc"


def test_another_tenant_gets_nothing():
    store = TraceStore()
    store.save(_record("abc", "alice"))
    assert store.get("abc", owner="bob") is None


def test_a_foreign_trace_is_indistinguishable_from_a_missing_one():
    """Both None, so the response cannot be used to discover which ids exist."""
    store = TraceStore()
    store.save(_record("abc", "alice"))
    assert store.get("abc", owner="bob") == store.get("does-not-exist", owner="bob")


def test_recent_lists_only_the_callers_own_runs():
    store = TraceStore()
    store.save(_record("a1", "alice"))
    store.save(_record("b1", "bob"))
    store.save(_record("a2", "alice"))
    assert {r["trace_id"] for r in store.recent("alice")} == {"a1", "a2"}
    assert {r["trace_id"] for r in store.recent("bob")} == {"b1"}


def test_recent_is_newest_first():
    store = TraceStore()
    for i in range(3):
        store.save(_record(f"t{i}", "alice"))
    assert [r["trace_id"] for r in store.recent("alice")] == ["t2", "t1", "t0"]


def test_the_in_process_fallback_is_bounded():
    """No Redis means no TTL to expire entries, so the dict is capped by count
    instead — otherwise it is a slow leak in exactly the configuration that has
    nothing to trim it."""
    store = TraceStore()
    for i in range(200):
        store.save(_record(f"t{i}", "alice"))
    assert len(store._mem) <= 50
    assert store.get("t199", owner="alice") is not None
    assert store.get("t0", owner="alice") is None


def test_a_record_without_a_trace_id_is_dropped_rather_than_stored_under_none():
    store = TraceStore()
    store.save({"owner": "alice", "steps": []})
    assert store.recent("alice") == []
