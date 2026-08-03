"""Judge failover in the vendored guardrails service.

Why this is tested here: the guard **fails open**. When no judge answers, every
verdict is `judge_unavailable` and requests pass unscreened — so a broken chain
does not surface as an error anywhere, it surfaces as security quietly not
happening. These tests are the only place that notices.
"""

from __future__ import annotations

import json

import pytest
from guardrails.config import Settings
from guardrails.judge.errors import JudgeAPIError, JudgeParseError, JudgeTimeout
from guardrails.judge.judge import ChainHealth, InputVerdict, Judge
from guardrails.judge.providers import (
    PRESETS,
    MockProvider,
    build_provider_chain,
    parse_chain_spec,
)
from guardrails.policy import Policy

VERDICT = json.dumps(
    {"prompt_injection": 0.0, "jailbreak": 0.0, "off_topic": 0.0,
     "harmful_content": 0.0, "reason": "ok", "flagged_phrases": []}
)


class FakeProvider:
    """Scripted judge link. `script` entries are either raw text or an exception."""

    def __init__(self, name: str, model: str, script: list) -> None:
        self.name = name
        self.model = model
        self._script = list(script)
        self.calls = 0
        self.timeouts: list[float] = []

    async def complete(self, *, system, user, max_tokens, timeout, json_schema=None) -> str:
        self.calls += 1
        self.timeouts.append(timeout)
        item = self._script[min(self.calls - 1, len(self._script) - 1)]
        if isinstance(item, Exception):
            raise item
        return item


def settings(**kw) -> Settings:
    base = {
        "llm_provider": "mock", "llm_model": "mock-1", "llm_timeout_s": 5.0,
        "llm_failover_max_failures": 2, "llm_failover_cooldown_s": 300.0,
    }
    base.update(kw)
    return Settings(**base)


@pytest.fixture
def policy() -> Policy:
    return Policy(id="test")


# ── Spec parsing ────────────────────────────────────────────────────────────
def test_spec_splits_provider_on_the_first_colon_only():
    # Model ids carry colons and slashes of their own; only the first colon
    # separates the provider.
    (link,) = parse_chain_spec("deepinfra:meta-llama/Llama-3.3:70b")
    assert (link.provider, link.model) == ("deepinfra", "meta-llama/Llama-3.3:70b")


def test_spec_reads_a_base_url_suffix():
    (link,) = parse_chain_spec("custom:gemma@https://api.example.com/v1")
    assert (link.provider, link.model, link.base_url) == (
        "custom", "gemma", "https://api.example.com/v1",
    )


def test_spec_leaves_a_non_url_at_sign_in_the_model():
    # An `@` that isn't a URL belongs to the model id. Truncating it would build
    # a link that 404s on a name nobody wrote.
    (link,) = parse_chain_spec("deepinfra:org/model@v2")
    assert (link.model, link.base_url) == ("org/model@v2", None)


def test_spec_ignores_blanks_and_whitespace():
    links = parse_chain_spec("  deepseek:a , , cohere:b  ,")
    assert [(x.provider, x.model) for x in links] == [("deepseek", "a"), ("cohere", "b")]


def test_spec_rejects_an_entry_with_no_provider():
    with pytest.raises(ValueError):
        parse_chain_spec(":just-a-model")


# ── Chain construction ──────────────────────────────────────────────────────
def test_link_without_its_key_is_dropped_and_named(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    providers, skipped = build_provider_chain(settings(llm_fallbacks="cohere:command-a-03-2025"))
    assert [p.name for p in providers] == ["mock"]
    assert "COHERE_API_KEY" in skipped[0]


def test_fallback_does_not_inherit_the_primarys_key(monkeypatch):
    """GUARD_LLM_API_KEY belongs to the primary vendor.

    Handing it to a different vendor doesn't fail loudly — it 401s on every
    request, so the link reports healthy at startup and screens nothing.
    """
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    cfg = settings(
        llm_provider="custom", llm_model="m", llm_base_url="https://x/v1",
        llm_api_key="deepinfra-token", llm_fallbacks="deepseek:deepseek-v4-flash",
    )
    providers, skipped = build_provider_chain(cfg)
    assert len(providers) == 1
    assert "no DEEPSEEK_API_KEY" in skipped[0]


def test_fallback_on_the_same_provider_may_reuse_the_primarys_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    cfg = settings(
        llm_provider="deepseek", llm_model="deepseek-v4-flash", llm_api_key="sk-real",
        llm_fallbacks="deepseek:deepseek-v4-pro",
    )
    providers, _ = build_provider_chain(cfg)
    assert [p.model for p in providers] == ["deepseek-v4-flash", "deepseek-v4-pro"]


def test_duplicate_of_the_primary_is_dropped(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
    cfg = settings(llm_provider="deepseek", llm_model="deepseek-v4-flash",
                   llm_fallbacks="deepseek:deepseek-v4-flash")
    providers, skipped = build_provider_chain(cfg)
    assert len(providers) == 1          # a chain that retries one model is a retry loop
    assert "duplicate" in skipped[0]


def test_unknown_provider_is_skipped_not_fatal():
    providers, skipped = build_provider_chain(settings(llm_fallbacks="nope:x"))
    assert len(providers) == 1 and "unknown provider" in skipped[0]


def test_cohere_preset_points_at_the_compatibility_path():
    # Cohere's OpenAI surface is /compatibility/v1; plain /v1 is the native
    # protocol and 404s the chat-completions shape.
    assert PRESETS["cohere"].base_url.endswith("/compatibility/v1")
    assert PRESETS["cohere"].key_env == "COHERE_API_KEY"


# ── Failover behaviour ──────────────────────────────────────────────────────
async def test_second_link_serves_when_the_first_errors(policy):
    a = FakeProvider("a", "m1", [JudgeAPIError("503")])
    b = FakeProvider("b", "m2", [VERDICT])
    outcome = await Judge([a, b], settings()).evaluate_input(policy, "hi", [])
    assert isinstance(outcome.verdict, InputVerdict)
    assert (outcome.provider, outcome.model) == ("b", "m2")


async def test_failover_covers_a_model_that_cannot_produce_the_verdict(policy):
    """The failure a provider-level wrapper would miss.

    A model that answers 200 with prose instead of JSON never raises inside
    `complete()` — it raises in the parser. If the chain didn't wrap parsing,
    this link would look healthy and no verdict would ever be produced.
    """
    a = FakeProvider("a", "m1", ["I'm afraid I can't do that."])
    b = FakeProvider("b", "m2", [VERDICT])
    outcome = await Judge([a, b], settings()).evaluate_input(policy, "hi", [])
    assert outcome.provider == "b"
    assert a.calls == 2  # the repair retry happens before moving on


async def test_a_link_gets_one_repair_retry_before_the_next_link(policy):
    a = FakeProvider("a", "m1", ["not json", VERDICT])
    b = FakeProvider("b", "m2", [VERDICT])
    outcome = await Judge([a, b], settings()).evaluate_input(policy, "hi", [])
    assert outcome.provider == "a" and b.calls == 0


async def test_exhausted_chain_raises_the_last_real_error_kind(policy):
    a = FakeProvider("a", "m1", [JudgeAPIError("503")])
    b = FakeProvider("b", "m2", [JudgeTimeout("slow")])
    with pytest.raises(JudgeTimeout) as err:
        await Judge([a, b], settings()).evaluate_input(policy, "hi", [])
    # fail_mode reads `kind`; a synthetic error here would mislabel the outage.
    assert err.value.kind == "timeout"
    assert (err.value.provider, err.value.model) == ("b", "m2")


async def test_parse_failure_on_the_last_link_still_names_that_link(policy):
    a = FakeProvider("a", "m1", [JudgeAPIError("503")])
    b = FakeProvider("b", "m2", ["prose", "still prose"])
    with pytest.raises(JudgeParseError) as err:
        await Judge([a, b], settings()).evaluate_input(policy, "hi", [])
    assert err.value.provider == "b"


async def test_single_provider_still_propagates_its_error(policy):
    a = FakeProvider("a", "m1", [JudgeAPIError("boom")])
    with pytest.raises(JudgeAPIError):
        await Judge(a, settings()).evaluate_input(policy, "hi", [])


async def test_a_bare_provider_is_accepted_like_before(policy):
    judge = Judge(MockProvider(), settings())
    assert judge.chain == ["mock:mock-1"]
    outcome = await judge.evaluate_input(policy, "hello", [])
    assert outcome.provider == "mock"


# ── Circuit breaker ─────────────────────────────────────────────────────────
async def test_breaker_takes_a_dead_link_out_of_rotation(policy):
    a = FakeProvider("a", "m1", [JudgeAPIError("503")])
    b = FakeProvider("b", "m2", [VERDICT])
    judge = Judge([a, b], settings(llm_failover_max_failures=2))
    for _ in range(4):
        await judge.evaluate_input(policy, "hi", [])
    # Two failures trip it; the remaining requests skip `a` entirely instead of
    # paying its timeout every time.
    assert a.calls == 2
    assert b.calls == 4


async def test_a_tripped_link_is_still_tried_when_nothing_else_works(policy):
    """A five-minute-old guess about a provider must not be why nothing is screened."""
    a = FakeProvider("a", "m1", [JudgeAPIError("503"), VERDICT])
    b = FakeProvider("b", "m2", [JudgeAPIError("down")])
    judge = Judge([a, b], settings(llm_failover_max_failures=1))
    with pytest.raises(JudgeAPIError):
        await judge.evaluate_input(policy, "hi", [])   # both fail, both trip
    outcome = await judge.evaluate_input(policy, "hi", [])
    assert outcome.provider == "a"  # tried again despite being tripped


def test_breaker_order_puts_healthy_links_first():
    h = ChainHealth(3, max_failures=1, cooldown_s=300)
    h.record_failure(0)
    assert h.order() == [1, 2, 0]
    h.record_success(0)
    assert h.order() == [0, 1, 2]


def test_breaker_resets_the_counter_on_success():
    h = ChainHealth(2, max_failures=2, cooldown_s=300)
    h.record_failure(0)
    h.record_success(0)
    assert h.record_failure(0) is False   # the earlier failure no longer counts
    assert not h.is_open(0)


def test_cooldown_of_zero_never_parks_a_link():
    h = ChainHealth(2, max_failures=1, cooldown_s=0)
    h.record_failure(0)
    assert h.order() == [0, 1]


# ── Deployment shape ────────────────────────────────────────────────────────
def test_blank_total_timeout_means_unset_not_a_startup_crash():
    """Compose can pass a variable but cannot omit one.

    An unset `GUARD_LLM_TOTAL_TIMEOUT_S` arrives as `""`, and a float field
    would refuse to start the process on it — taking the guard down over a
    variable the operator deliberately left empty.
    """
    assert settings(llm_total_timeout_s="").llm_total_timeout_s is None


def test_blank_fallbacks_is_a_single_provider_not_an_error():
    providers, skipped = build_provider_chain(settings(llm_fallbacks=""))
    assert len(providers) == 1 and skipped == []


# ── Deadline ────────────────────────────────────────────────────────────────
async def test_per_call_timeout_shrinks_to_the_remaining_budget(policy):
    a = FakeProvider("a", "m1", [VERDICT])
    await Judge([a], settings(llm_timeout_s=30.0, llm_total_timeout_s=4.0)).evaluate_input(
        policy, "hi", []
    )
    # The link asked for 30s; the overall budget is 4s, so it gets 4s.
    assert a.timeouts[0] == pytest.approx(4.0, abs=0.2)


async def test_no_total_timeout_leaves_the_per_link_timeout_alone(policy):
    a = FakeProvider("a", "m1", [VERDICT])
    await Judge([a], settings(llm_timeout_s=7.0)).evaluate_input(policy, "hi", [])
    assert a.timeouts[0] == 7.0


async def test_budget_stops_the_chain_instead_of_overrunning_the_caller(policy):
    """Three links x per-link timeout can outlast the caller's own timeout.

    Past that point the caller has already given up and failed open, so further
    attempts only spend tokens on a verdict nobody will read.
    """
    class Slow(FakeProvider):
        async def complete(self, **kw):
            import time as _t
            _t.sleep(0.6)  # blocking on purpose: burn wall-clock budget
            raise JudgeAPIError("503")

    links = [Slow(f"p{i}", f"m{i}", []) for i in range(4)]
    judge = Judge(links, settings(llm_timeout_s=5.0, llm_total_timeout_s=1.6))
    with pytest.raises(JudgeAPIError):
        await judge.evaluate_input(policy, "hi", [])
    assert sum(p.calls for p in links) < 4
