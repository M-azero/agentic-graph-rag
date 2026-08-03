"""Output-token accounting.

The bug these exist to prevent: `/query` with `stream: false` and `/compare`
produced answers without booking a single token, so `tokens_per_day` and
`tokens_per_month` enforced nothing on the paths a script is most likely to
use. Nothing failed and no log said so — the meter simply read 0 forever.
"""

from __future__ import annotations

import inspect

import pytest

from graphrag.usage import (
    UsageRecorder,
    estimate_tokens,
    record_answer_tokens,
)


class FakeLimits:
    def __init__(self) -> None:
        self.tokens: dict[str, int] = {}

    def record_tokens(self, user_id: str, amount: int) -> None:
        self.tokens[user_id] = self.tokens.get(user_id, 0) + amount


class FakeRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, int]] = {}

    def hincrby(self, key: str, field: str, amount: int) -> None:
        self.hashes.setdefault(key, {})
        self.hashes[key][field] = self.hashes[key].get(field, 0) + amount


# ── estimate_tokens ─────────────────────────────────────────────────────────
def test_empty_answer_costs_nothing():
    assert estimate_tokens("") == 0


def test_any_real_answer_costs_at_least_one():
    """Rounding must never make a short answer free — that would be a hole big
    enough to drive a loop through."""
    assert estimate_tokens("ok") >= 1
    assert estimate_tokens(".") >= 1


def test_estimate_is_roughly_four_characters_per_token_for_ascii():
    assert estimate_tokens("a" * 400) == pytest.approx(100, rel=0.05)


def test_non_ascii_scripts_are_not_charged_at_the_ascii_rate():
    """Arabic/Hebrew/CJK pack ~2 chars per token, not ~4.

    Billing them at the ASCII ratio hands those users roughly double the real
    budget — a quota that is twice as generous depending on the language.
    """
    arabic = estimate_tokens("مرحبا بالعالم" * 20)
    latin = estimate_tokens("hello world!!" * 20)
    assert arabic > latin * 1.5


def test_estimate_grows_with_length():
    short, long = estimate_tokens("word " * 10), estimate_tokens("word " * 100)
    assert long > short * 5


# ── record_answer_tokens ────────────────────────────────────────────────────
async def test_prompt_and_completion_are_separate_events_but_one_ceiling():
    """They are priced differently — often 4-5x — so a merged event could not
    explain a bill. The quota is a single number because it bounds spend, and
    charging only the completion side watches the cheaper half."""
    limits, redis = FakeLimits(), FakeRedis()
    recorder = UsageRecorder(None, limits)
    await record_answer_tokens(
        recorder, redis, tenant_id="t", account_id="a",
        tokens=200, input_tokens=4800,
    )
    assert limits.tokens == {"a": 5000}
    assert redis.hashes["graphrag:usage:tokens"]["t"] == 5000


async def test_prompt_only_run_is_still_charged():
    """A blocked or empty answer still sent the prompt."""
    limits, redis = FakeLimits(), FakeRedis()
    await record_answer_tokens(
        UsageRecorder(None, limits), redis,
        tenant_id="t", account_id="a", tokens=0, input_tokens=3000,
    )
    assert limits.tokens == {"a": 3000}


async def test_both_counters_are_written_together():
    """The Redis counter enforces the quota; the usage_events row feeds the
    admin charts. A path that writes one and not the other either enforces
    without a record or records without enforcing."""
    limits, redis = FakeLimits(), FakeRedis()
    recorder = UsageRecorder(None, limits)
    await record_answer_tokens(
        recorder, redis, tenant_id="tenant-a", account_id="acct-1", tokens=250,
    )
    assert limits.tokens == {"acct-1": 250}
    assert redis.hashes["graphrag:usage:tokens"]["tenant-a"] == 250


async def test_quota_is_charged_to_the_account_not_the_tenant():
    """`check_tokens` reads by account id; charging the tenant string instead
    would leave the limit reading zero while the meter looked busy."""
    limits, redis = FakeLimits(), FakeRedis()
    await record_answer_tokens(
        UsageRecorder(None, limits), redis,
        tenant_id="mohamed-ham-5df2", account_id="uuid-1", tokens=10,
    )
    assert "uuid-1" in limits.tokens and "mohamed-ham-5df2" not in limits.tokens


async def test_zero_tokens_records_nothing():
    limits, redis = FakeLimits(), FakeRedis()
    await record_answer_tokens(
        UsageRecorder(None, limits), redis,
        tenant_id="t", account_id="a", tokens=0,
    )
    assert limits.tokens == {} and redis.hashes == {}


async def test_anonymous_caller_still_moves_the_redis_counter():
    """No account (dev mode / auth off) means no quota row to charge, but the
    legacy /usage report should still see the traffic."""
    redis = FakeRedis()
    await record_answer_tokens(
        None, redis, tenant_id="dev", account_id=None, tokens=7,
    )
    assert redis.hashes["graphrag:usage:tokens"]["dev"] == 7


async def test_a_dead_redis_does_not_break_the_answer():
    """Usage is bookkeeping. It must never fail a response the user already has."""
    class Broken(FakeRedis):
        def hincrby(self, *a, **kw):
            raise ConnectionError("redis is down")

    limits = FakeLimits()
    await record_answer_tokens(
        UsageRecorder(None, limits), Broken(),
        tenant_id="t", account_id="a", tokens=5,
    )
    assert limits.tokens == {"a": 5}   # the quota write still landed


# ── the routes actually call it ─────────────────────────────────────────────
def test_every_answer_path_books_tokens():
    """Guards the regression directly: if a handler that produces an answer
    stops recording, this fails rather than the meter quietly reading 0."""
    from graphrag.api import streaming
    from graphrag.api.routers import query

    for fn in (query.query, query.compare, streaming.sse_answer):
        assert "record_answer_tokens" in inspect.getsource(fn), (
            f"{fn.__name__} produces an answer without booking its tokens"
        )


def test_non_streaming_paths_measure_before_the_guard_can_swap_the_text():
    """The model has already run and spent the tokens by the time the output
    guard decides to block. Billing the short refusal that replaces the answer
    would make the most expensive requests the cheapest."""
    from graphrag.api.routers import query

    for fn in (query.query, query.compare):
        src = inspect.getsource(fn)
        # `v_out.blocked` is the branch that swaps the answer for a refusal —
        # anchor on that rather than on check_output, which also appears in the
        # streaming closure further up.
        assert src.index("estimate_tokens(answer)") < src.index("v_out.blocked"), (
            f"{fn.__name__} measures the answer after the guard may have replaced it"
        )
