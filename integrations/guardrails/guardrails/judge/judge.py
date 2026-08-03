"""Judge orchestration: prompt -> provider call -> defensive parse -> typed verdict.

Concurrency is bounded by a semaphore. Each call is schema-constrained where the provider
supports it, but the defensive parser always runs regardless (models lie about JSON). One
repair retry is attempted on a parse failure before giving up with a ``JudgeParseError``.

A judge may be a **chain** of providers (``GUARD_LLM_FALLBACKS``). Failover lives here
rather than behind the ``LLMProvider`` interface on purpose: ``JudgeParseError`` is raised
by the parser, not by ``complete()``, and a model that cannot produce the structured
verdict is exactly the failure a provider-level wrapper would miss — the endpoint answers
200, the service looks healthy, and no verdict is ever produced.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError, model_validator

from ..config import Settings
from ..policy import Policy
from ..schemas import ChatTurn, ContextDoc
from .errors import JudgeAPIError, JudgeError, JudgeParseError
from .prompts import (
    INPUT_VERDICT_SCHEMA,
    OUTPUT_VERDICT_SCHEMA_DOCS,
    OUTPUT_VERDICT_SCHEMA_NODOCS,
    build_input_prompt,
    build_output_prompt,
)
from .providers import LLMProvider

logger = logging.getLogger("guardrails")

# Appended to the system prompt on the single repair retry (also the MockProvider's cue).
REPAIR_SUFFIX = "\n\nReturn ONLY valid JSON matching the schema. No prose, no code fences."

_MAX_REASON = 200
_MAX_PHRASE = 200
_MAX_LIST = 10

# Don't start an attempt with less budget than this: a sub-second window buys a
# connect and a timeout, never a verdict, and spends the caller's remaining time
# producing the same failure it already has.
_MIN_ATTEMPT_S = 1.0


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


class InputVerdict(BaseModel):
    prompt_injection: float = 0.0
    jailbreak: float = 0.0
    off_topic: float = 0.0
    harmful_content: float = 0.0
    reason: str = ""
    flagged_phrases: list[str] = []

    @model_validator(mode="after")
    def _sanitize(self) -> "InputVerdict":
        self.prompt_injection = _clamp(self.prompt_injection)
        self.jailbreak = _clamp(self.jailbreak)
        self.off_topic = _clamp(self.off_topic)
        self.harmful_content = _clamp(self.harmful_content)
        self.reason = self.reason[:_MAX_REASON]
        self.flagged_phrases = [p[:_MAX_PHRASE] for p in self.flagged_phrases[:_MAX_LIST]]
        return self

    def scores(self) -> dict[str, float]:
        return {
            "prompt_injection": self.prompt_injection,
            "jailbreak": self.jailbreak,
            "off_topic": self.off_topic,
            "harmful_content": self.harmful_content,
        }


class OutputVerdict(BaseModel):
    ungrounded: float = 0.0
    harmful_content: float = 0.0
    off_topic: float = 0.0
    unsupported_claims: list[str] = []
    reason: str = ""

    @model_validator(mode="after")
    def _sanitize(self) -> "OutputVerdict":
        self.ungrounded = _clamp(self.ungrounded)
        self.harmful_content = _clamp(self.harmful_content)
        self.off_topic = _clamp(self.off_topic)
        self.reason = self.reason[:_MAX_REASON]
        self.unsupported_claims = [c[:_MAX_PHRASE] for c in self.unsupported_claims[:_MAX_LIST]]
        return self

    def scores(self) -> dict[str, float]:
        return {
            "ungrounded": self.ungrounded,
            "harmful_content": self.harmful_content,
            "off_topic": self.off_topic,
        }


@dataclass
class JudgeOutcome:
    verdict: InputVerdict | OutputVerdict
    latency_ms: float
    # Which link actually answered. Reported per-verdict rather than read off
    # the Judge, because one Judge serves concurrent requests and the answer
    # differs between them mid-failover.
    provider: str = ""
    model: str | None = None


# ── Defensive parsing ───────────────────────────────────────────────────────
def _first_balanced_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` substring, respecting string escapes."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_verdict(raw: str, model_cls: type[BaseModel]) -> BaseModel:
    """Strip fences/prose, grab the first JSON object, validate. Raise on any failure."""
    obj = _first_balanced_object(raw or "")
    if obj is None:
        raise JudgeParseError("no JSON object in response")
    try:
        data = json.loads(obj)
    except json.JSONDecodeError as exc:
        raise JudgeParseError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise JudgeParseError("top-level JSON is not an object")
    try:
        return model_cls.model_validate(data)
    except ValidationError as exc:
        raise JudgeParseError(f"schema mismatch: {exc}") from exc


# ── Circuit breaker ─────────────────────────────────────────────────────────
class ChainHealth:
    """Consecutive-failure counters and cooldowns, one slot per chain link."""

    def __init__(self, size: int, max_failures: int, cooldown_s: float) -> None:
        self.size = size
        self._max = max_failures
        self._cooldown = cooldown_s
        self._fails: dict[int, int] = {}
        self._open_until: dict[int, float] = {}

    def order(self) -> list[int]:
        """Healthy links in configured order, then tripped ones as a last resort.

        Tripped links stay reachable deliberately. The alternative is that a
        breaker opened five minutes ago decides nothing gets screened now, and
        an out-of-date guess about a provider is a worse input than the provider
        itself.
        """
        now = time.monotonic()
        healthy = [i for i in range(self.size) if self._open_until.get(i, 0.0) <= now]
        tripped = [i for i in range(self.size) if self._open_until.get(i, 0.0) > now]
        return healthy + tripped

    def is_open(self, i: int) -> bool:
        return self._open_until.get(i, 0.0) > time.monotonic()

    def record_failure(self, i: int) -> bool:
        """Count a failure; return True if this one tripped the breaker."""
        self._fails[i] = self._fails.get(i, 0) + 1
        if self._fails[i] >= self._max:
            self._open_until[i] = time.monotonic() + self._cooldown
            self._fails[i] = 0
            return True
        return False

    def record_success(self, i: int) -> None:
        self._fails.pop(i, None)
        self._open_until.pop(i, None)


# ── Judge ───────────────────────────────────────────────────────────────────
class Judge:
    """One verdict, from the first link in the chain that can produce one."""

    def __init__(
        self, provider: LLMProvider | Sequence[LLMProvider], settings: Settings
    ) -> None:
        self._providers: list[LLMProvider] = (
            list(provider) if isinstance(provider, (list, tuple)) else [provider]
        )
        if not self._providers:
            raise ValueError("Judge needs at least one provider")
        self._settings = settings
        self._sem = asyncio.Semaphore(settings.max_concurrent_judge)
        self._health = ChainHealth(
            len(self._providers),
            settings.llm_failover_max_failures,
            settings.llm_failover_cooldown_s,
        )

    @property
    def provider_name(self) -> str:
        return self._providers[0].name

    @property
    def model(self) -> str | None:
        return self._providers[0].model

    @property
    def chain(self) -> list[str]:
        """Every link, primary first — what /health and the startup log report."""
        return [f"{p.name}:{p.model}" for p in self._providers]

    async def evaluate_input(
        self, policy: Policy, text: str, context: list[ChatTurn]
    ) -> JudgeOutcome:
        system, user = build_input_prompt(policy, text, context)
        return await self._run(system, user, INPUT_VERDICT_SCHEMA, InputVerdict)

    async def evaluate_output(
        self, policy: Policy, user_input: str, output: str, docs: list[ContextDoc]
    ) -> JudgeOutcome:
        system, user, has_docs = build_output_prompt(policy, user_input, output, docs)
        schema = OUTPUT_VERDICT_SCHEMA_DOCS if has_docs else OUTPUT_VERDICT_SCHEMA_NODOCS
        return await self._run(system, user, schema, OutputVerdict)

    async def _run(
        self, system: str, user: str, schema: dict, model_cls: type[BaseModel]
    ) -> JudgeOutcome:
        t0 = time.perf_counter()
        budget = self._settings.llm_total_timeout_s
        # The semaphore is held across the whole chain: a failing link must not
        # release its concurrency slot only to have the retry queue behind
        # someone else's first attempt.
        async with self._sem:
            last: JudgeError | None = None
            for idx in self._health.order():
                provider = self._providers[idx]
                remaining = None if budget is None else budget - (time.perf_counter() - t0)
                if remaining is not None and remaining < _MIN_ATTEMPT_S:
                    logger.warning(json.dumps({
                        "event": "judge_budget_exhausted", "skipped": provider.name,
                        "elapsed_s": round(time.perf_counter() - t0, 2),
                    }))
                    break
                try:
                    verdict = await self._attempt(
                        provider, system, user, schema, model_cls, remaining
                    )
                except JudgeError as exc:
                    last = exc
                    tripped = self._health.record_failure(idx)
                    logger.warning(json.dumps({
                        "event": "judge_link_failed",
                        "provider": provider.name, "model": provider.model,
                        "kind": exc.kind, "error": str(exc)[:200],
                        "breaker_tripped": tripped,
                        "remaining_links": len(self._providers) - idx - 1,
                    }))
                    continue

                self._health.record_success(idx)
                if idx != 0:
                    logger.warning(json.dumps({
                        "event": "judge_degraded", "serving": f"{provider.name}:{provider.model}",
                        "primary": self.chain[0],
                    }))
                latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                return JudgeOutcome(
                    verdict=verdict,  # type: ignore[arg-type]
                    latency_ms=latency_ms,
                    provider=provider.name,
                    model=provider.model,
                )

        # Every link failed (or the budget ran out). Re-raise the last real
        # failure so fail_mode sees the true error kind, not a synthetic one.
        if last is None:  # budget gone before any attempt started
            last = JudgeAPIError("judge budget exhausted before any provider was tried")
            last.provider, last.model = self.provider_name, self.model
        raise last

    async def _attempt(
        self, provider: LLMProvider, system: str, user: str, schema: dict,
        model_cls: type[BaseModel], remaining: float | None,
    ) -> BaseModel:
        """One link, with every failure out of it stamped with which link it was."""
        deadline = None if remaining is None else time.perf_counter() + remaining
        try:
            return await self._once(provider, system, user, schema, model_cls, deadline)
        except JudgeError as exc:
            _annotate(exc, provider)
            raise

    async def _once(
        self, provider: LLMProvider, system: str, user: str, schema: dict,
        model_cls: type[BaseModel], deadline: float | None,
    ) -> BaseModel:
        """Call, parse, and on malformed JSON one repair retry.

        Both calls draw on the same deadline, so a link whose repair retry would
        overrun gives up and leaves the time to the next link.
        """
        raw = await self._call(provider, system, user, schema, deadline)
        try:
            return parse_verdict(raw, model_cls)
        except JudgeParseError:
            if deadline is not None and deadline - time.perf_counter() < _MIN_ATTEMPT_S:
                raise
            raw = await self._call(provider, system + REPAIR_SUFFIX, user, schema, deadline)
            return parse_verdict(raw, model_cls)  # propagate on 2nd failure

    async def _call(
        self, provider: LLMProvider, system: str, user: str,
        schema: dict, deadline: float | None,
    ) -> str:
        timeout = self._settings.llm_timeout_s
        if deadline is not None:
            timeout = max(_MIN_ATTEMPT_S, min(timeout, deadline - time.perf_counter()))
        return await provider.complete(
            system=system,
            user=user,
            max_tokens=self._settings.llm_max_tokens,
            timeout=timeout,
            json_schema=schema,
        )


def _annotate(exc: JudgeError, provider: LLMProvider) -> JudgeError:
    exc.provider = provider.name
    exc.model = provider.model
    return exc
