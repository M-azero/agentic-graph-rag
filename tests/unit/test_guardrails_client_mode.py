"""A request cannot decide how hard the guard checks it.

`GuardPipeline._should_judge` read `mode` straight off the request body, and
`mode: "fast"` skipped the LLM judge entirely — leaving only the regex rules.
That is the expensive half of the guard, and the half that catches what a regex
cannot: groundedness, subtle jailbreaks, harm.

It is a legitimate escape hatch for a trusted bulk caller, so it is kept behind
a server setting rather than removed. What it must not be is the default, which
is what "read it off the body" amounts to. Today the graphrag API never sends
the field and the service is not published — so this is a latch closing before
the day either of those changes.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from guardrails.config import Settings
from guardrails.pipeline import GuardPipeline


def _policy(*, enabled: bool = True, trigger: str = "always"):
    return SimpleNamespace(judge=SimpleNamespace(enabled=enabled, trigger=trigger))


def _pipeline(**settings_kwargs) -> GuardPipeline:
    settings = Settings(**settings_kwargs)
    return GuardPipeline(settings, registry=None, judge=None)


def test_client_mode_is_ignored_by_default():
    """The regression: `{"mode": "fast"}` used to turn the judge off."""
    pipeline = _pipeline()
    assert pipeline._should_judge(_policy(), "fast", []) is True


def test_full_mode_judges_as_before():
    assert _pipeline()._should_judge(_policy(), "full", []) is True


def test_client_mode_is_honoured_when_the_operator_allows_it():
    pipeline = _pipeline(allow_client_mode=True)
    assert pipeline._should_judge(_policy(), "fast", []) is False
    assert pipeline._should_judge(_policy(), "full", []) is True


def test_the_setting_defaults_to_off():
    assert Settings().allow_client_mode is False


@pytest.mark.parametrize("allow", [False, True])
def test_the_policy_stays_authoritative_either_way(allow):
    """`judge.trigger` is the server-side control. Whatever the client asks
    for, a policy that says never must still mean never."""
    pipeline = _pipeline(allow_client_mode=allow)
    assert pipeline._should_judge(_policy(trigger="never"), "full", []) is False
    assert pipeline._should_judge(_policy(enabled=False), "full", []) is False


@pytest.mark.parametrize("allow", [False, True])
def test_on_rule_flag_still_gates_on_a_hit(allow):
    pipeline = _pipeline(allow_client_mode=allow)
    policy = _policy(trigger="on_rule_flag")
    assert pipeline._should_judge(policy, "full", []) is False
    assert pipeline._should_judge(policy, "full", [object()]) is True
