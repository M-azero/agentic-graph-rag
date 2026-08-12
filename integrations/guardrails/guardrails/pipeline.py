"""Decision engine — the behavioural heart.

Cheap deterministic checks run first and can short-circuit to BLOCK with zero judge cost.
Risk is **monotonic**: signals only raise it, and a judge ``allow`` can never overturn a
deterministic block (``score = max(rule, judge)`` per category). Every well-formed request
yields an HTTP-200 verdict; judge outages become ``fail_mode`` behaviour, never a 5xx.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections import OrderedDict
from typing import Any

from .checks import injection_rules, pii
from .checks.normalize import normalize
from .config import Settings
from .judge.errors import JudgeError
from .judge.judge import InputVerdict, Judge, OutputVerdict
from .policy import CheckConfig, Policy
from .schemas import (
    Action,
    CategoryResult,
    ChatTurn,
    ContextDoc,
    GroundednessInfo,
    GuardInputRequest,
    GuardInputResponse,
    GuardOutputRequest,
    GuardOutputResponse,
    JudgeInfo,
    RuleHit,
)

RULE_BLOCK_SCORE = 1.0
RULE_FLAG_SCORE = 0.65

# Thresholds used for categories with no explicit policy check (e.g. custom rules).
_UNKNOWN_CHECK = CheckConfig(flag_at=0.6, block_at=1.1)


# ── Verdict cache ───────────────────────────────────────────────────────────
class VerdictCache:
    """LRU + TTL cache of successful judge verdicts (never caches failures).

    Values are ``(verdict, provider, model)``: the link that produced a verdict
    is cached with it, so a hit reports the judge that actually decided rather
    than whichever link happens to be primary when the hit is served.
    """

    def __init__(self, *, max_size: int, ttl_seconds: int, enabled: bool = True) -> None:
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._max = max_size
        self._ttl = ttl_seconds
        self._enabled = enabled

    def get(self, key: str) -> Any | None:
        if not self._enabled:
            return None
        item = self._data.get(key)
        if item is None:
            return None
        ts, value = item
        if self._ttl and time.time() - ts > self._ttl:
            self._data.pop(key, None)
            return None
        self._data.move_to_end(key)
        return value

    def set(self, key: str, value: Any) -> None:
        if not self._enabled:
            return
        self._data[key] = (time.time(), value)
        self._data.move_to_end(key)
        while len(self._data) > self._max:
            self._data.popitem(last=False)

    @staticmethod
    def key(*parts: str) -> str:
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


# ── Pipeline ────────────────────────────────────────────────────────────────
class GuardPipeline:
    def __init__(self, settings: Settings, registry: Any, judge: Judge) -> None:
        self._settings = settings
        self._registry = registry
        self._judge = judge
        pol_cache = settings.cache_enabled
        self._cache = VerdictCache(
            max_size=settings.cache_size,
            ttl_seconds=settings.cache_ttl_s,
            enabled=pol_cache,
        )

    # ── INPUT ───────────────────────────────────────────────────────────────
    async def check_input(self, req: GuardInputRequest) -> GuardInputResponse:
        t0 = time.perf_counter()
        policy = self._registry.get(req.policy_id)  # raises KeyError -> 404 in main
        rid = uuid.uuid4().hex

        if len(req.input) > policy.limits.max_input_chars:
            return self._too_long(req.policy_id, rid, t0, "input_too_long", policy)

        norm = normalize(req.input)
        checks = policy.checks.input

        hits: list[RuleHit] = injection_rules.scan(norm.matchable)
        pii_hits: list[RuleHit] = pii.scan(req.input)
        hits += pii_hits
        hits += policy.scan_policy_rules(norm.matchable)

        rule_scores, rule_evidence, block_cats = self._aggregate_rules(hits, checks)
        # Synthetic evasion signal from normalization.
        if checks.evasion.enabled and norm.evasion_score > 0:
            rule_scores["evasion"] = max(rule_scores.get("evasion", 0.0), norm.evasion_score)
            if norm.mixed_script_words:
                rule_evidence.setdefault("evasion", []).append(
                    "mixed-script: " + ", ".join(norm.mixed_script_words[:3])
                )

        # 5. Hard short-circuit — deterministic block, no judge.
        if block_cats:
            return self._blocked_by_rules(
                policy, req.policy_id, rid, t0, block_cats, rule_scores, rule_evidence
            )

        # 6-8. Judge gate + call.
        judge_info = JudgeInfo(invoked=False)
        judge_scores: dict[str, float] = {}
        judge_failed = False
        if self._should_judge(policy, req.mode, hits):
            judge_text = _redact_for_judge(req.input, checks, pii_hits)
            key = VerdictCache.key("input", policy.content_hash, norm.matchable,
                                   _turns_repr(req.context, policy.judge.include_context_turns))
            cached = self._cache.get(key)
            if cached is not None:
                verdict, c_prov, c_model = cached
                judge_scores = verdict.scores()
                judge_info = self._judge_info(cached_flag=True, provider=c_prov, model=c_model)
            else:
                try:
                    outcome = await self._judge.evaluate_input(policy, judge_text, req.context)
                    verdict: InputVerdict = outcome.verdict  # type: ignore[assignment]
                    self._cache.set(key, (verdict, outcome.provider, outcome.model))
                    judge_scores = verdict.scores()
                    judge_info = self._judge_info(
                        latency_ms=outcome.latency_ms,
                        provider=outcome.provider, model=outcome.model,
                    )
                except JudgeError as exc:
                    judge_failed = True
                    judge_info = self._judge_info(
                        error=exc.kind, provider=exc.provider, model=exc.model
                    )

        return self._decide_input(
            policy, req.policy_id, rid, t0, checks,
            rule_scores, rule_evidence, judge_scores, judge_info, judge_failed,
        )

    # ── OUTPUT ──────────────────────────────────────────────────────────────
    async def check_output(self, req: GuardOutputRequest) -> GuardOutputResponse:
        t0 = time.perf_counter()
        policy = self._registry.get(req.policy_id)
        rid = uuid.uuid4().hex
        checks = policy.checks.output

        if len(req.output) > policy.limits.max_output_chars:
            base = self._too_long(req.policy_id, rid, t0, "output_too_long", policy)
            return _as_output(base, sanitized_output=None, modified=False,
                              groundedness=GroundednessInfo(checked=False))

        norm = normalize(req.output)
        pii_hits = pii.scan(req.output)

        # Sanitized output (independent of the final action).
        redact_cats = tuple(
            c for c in ("pii", "secrets")
            if getattr(checks, c).enabled and getattr(checks, c).redact
        )
        sanitized, _ = (
            pii.redact(req.output, categories=redact_cats, hits=pii_hits)
            if redact_cats else (req.output, pii_hits)
        )
        modified = sanitized != req.output

        hits = [h for h in pii_hits if getattr(checks, h.category, None) and getattr(checks, h.category).enabled]

        # Local system-prompt leak check (system prompt NEVER sent to the judge).
        leak_cfg = checks.system_prompt_leak
        if leak_cfg.enabled and req.system_prompt:
            leaked = _system_prompt_leak(req.output, req.system_prompt, leak_cfg.min_overlap_chars)
            if leaked:
                hits.append(RuleHit("system_prompt_leak", "system_prompt_leak", "block",
                                    (0, 0), leaked[:120]))

        rule_scores, rule_evidence, block_cats = self._aggregate_rules(hits, checks)

        if block_cats:
            base = self._blocked_by_rules(
                policy, req.policy_id, rid, t0, block_cats, rule_scores, rule_evidence
            )
            return _as_output(base, sanitized_output=None, modified=modified,
                              groundedness=GroundednessInfo(checked=False))

        # Judge (groundedness + harm + off_topic).
        judge_info = JudgeInfo(invoked=False)
        judge_scores: dict[str, float] = {}
        judge_failed = False
        has_docs = bool(req.context_docs)
        unsupported: list[str] = []
        if self._should_judge(policy, req.mode, hits):
            judge_text = _redact_for_judge(req.output, checks, pii_hits)
            key = VerdictCache.key("output", policy.content_hash, norm.matchable,
                                   req.input[:2000], _docs_repr(req.context_docs))
            cached = self._cache.get(key)
            if cached is not None:
                verdict, c_prov, c_model = cached
                judge_scores = verdict.scores()
                unsupported = list(verdict.unsupported_claims)
                judge_info = self._judge_info(cached_flag=True, provider=c_prov, model=c_model)
            else:
                try:
                    outcome = await self._judge.evaluate_output(
                        policy, req.input, judge_text, req.context_docs
                    )
                    verdict: OutputVerdict = outcome.verdict  # type: ignore[assignment]
                    self._cache.set(key, (verdict, outcome.provider, outcome.model))
                    judge_scores = verdict.scores()
                    unsupported = list(verdict.unsupported_claims)
                    judge_info = self._judge_info(
                        latency_ms=outcome.latency_ms,
                        provider=outcome.provider, model=outcome.model,
                    )
                except JudgeError as exc:
                    judge_failed = True
                    judge_info = self._judge_info(
                        error=exc.kind, provider=exc.provider, model=exc.model
                    )

        grounded = GroundednessInfo(
            checked=has_docs and judge_info.invoked and not judge_failed and checks.ungrounded.enabled,
            score=judge_scores.get("ungrounded") if has_docs else None,
            unsupported_claims=unsupported if has_docs else [],
        )
        base = self._decide_input(
            policy, req.policy_id, rid, t0, checks,
            rule_scores, rule_evidence, judge_scores, judge_info, judge_failed,
        )
        if base.action == "block":
            sanitized = None  # type: ignore[assignment]
        return _as_output(base, sanitized_output=sanitized, modified=modified, groundedness=grounded)

    # ── Shared helpers ───────────────────────────────────────────────────────
    def _aggregate_rules(
        self, hits: list[RuleHit], checks: Any
    ) -> tuple[dict[str, float], dict[str, list[str]], set[str]]:
        scores: dict[str, float] = {}
        evidence: dict[str, list[str]] = {}
        block_cats: set[str] = set()
        for hit in hits:
            cfg = getattr(checks, hit.category, None)
            known = cfg is not None
            if known and not cfg.enabled:
                continue
            sc = RULE_BLOCK_SCORE if hit.tier == "block" else RULE_FLAG_SCORE
            scores[hit.category] = max(scores.get(hit.category, 0.0), sc)
            if hit.snippet:
                evidence.setdefault(hit.category, []).append(hit.snippet)
            if hit.tier == "block" and (not known or cfg.rule_action == "block"):
                block_cats.add(hit.category)
        return scores, evidence, block_cats

    def _should_judge(self, policy: Policy, mode: str, hits: list[RuleHit]) -> bool:
        # `mode` arrives in the request body, so honouring it lets the caller
        # decide how hard it is screened — `{"mode": "fast"}` skips the judge and
        # leaves only the regex rules. Gated behind a server setting rather than
        # trusted outright; the policy's own `judge.trigger` is the server-side
        # control and stays authoritative either way.
        if mode == "fast" and self._settings.allow_client_mode:
            return False
        if not policy.judge.enabled or policy.judge.trigger == "never":
            return False
        if policy.judge.trigger == "on_rule_flag" and not hits:
            return False
        return True

    def _judge_info(
        self, *, cached_flag: bool = False, latency_ms: float | None = None,
        error: str | None = None, provider: str | None = None, model: str | None = None,
    ) -> JudgeInfo:
        # `provider`/`model` name the chain link that actually served this
        # verdict; without them a degraded chain reports its primary forever and
        # a failover is invisible to whoever reads the response or the log.
        return JudgeInfo(
            invoked=True,
            provider=provider or self._judge.provider_name,
            model=model if provider else self._judge.model,
            cached=cached_flag,
            error=error,  # type: ignore[arg-type]
            latency_ms=latency_ms,
        )

    def _decide_input(
        self, policy: Policy, policy_id: str, rid: str, t0: float, checks: Any,
        rule_scores: dict[str, float], rule_evidence: dict[str, list[str]],
        judge_scores: dict[str, float], judge_info: JudgeInfo, judge_failed: bool,
    ) -> GuardInputResponse:
        categories: list[CategoryResult] = []
        for cat in sorted(set(rule_scores) | set(judge_scores)):
            cfg = getattr(checks, cat, None) or _UNKNOWN_CHECK
            if getattr(checks, cat, None) is not None and not cfg.enabled:
                continue
            r = rule_scores.get(cat, 0.0)
            j = judge_scores.get(cat, 0.0)
            score = max(r, j)
            action = _score_action(score, cfg)
            if action == "allow":
                continue
            source = "combined" if r > 0 and j > 0 else ("rules" if r > 0 else "judge")
            categories.append(CategoryResult(
                category=cat, score=round(score, 4), triggered=True, action=action,
                source=source, evidence=rule_evidence.get(cat, [])[:5],
            ))

        overall: Action = _combine_action(categories)

        # Judge failure -> fail_mode.
        if judge_failed:
            fm = policy.fail_mode
            if fm == "closed":
                overall = "block"
            elif fm == "flag" and overall == "allow":
                overall = "flag"
            if fm != "open":
                categories.append(CategoryResult(
                    category="judge_unavailable",
                    score=1.0 if fm == "closed" else RULE_FLAG_SCORE,
                    triggered=True, action="block" if fm == "closed" else "flag",
                    source="judge", evidence=[f"judge error: {judge_info.error}"],
                ))

        categories.sort(key=lambda c: c.score, reverse=True)
        reasons = [f"{c.category} (score {c.score:.2f}, {c.source})" for c in categories]
        refusal = policy.refusal_message if overall == "block" else None
        return GuardInputResponse(
            action=overall, categories=categories, reasons=reasons,
            refusal_message=refusal, policy_id=policy_id, judge=judge_info,
            latency_ms=_ms(t0), request_id=rid,
        )

    def _blocked_by_rules(
        self, policy: Policy, policy_id: str, rid: str, t0: float,
        block_cats: set[str], rule_scores: dict[str, float], rule_evidence: dict[str, list[str]],
    ) -> GuardInputResponse:
        categories = [
            CategoryResult(
                category=cat, score=max(rule_scores.get(cat, 1.0), 1.0), triggered=True,
                action="block", source="rules", evidence=rule_evidence.get(cat, [])[:5],
            )
            for cat in sorted(block_cats)
        ]
        reasons = [f"{c.category} (deterministic block)" for c in categories]
        return GuardInputResponse(
            action="block", categories=categories, reasons=reasons,
            refusal_message=policy.refusal_message, policy_id=policy_id,
            judge=JudgeInfo(invoked=False), latency_ms=_ms(t0), request_id=rid,
        )

    def _too_long(
        self, policy_id: str, rid: str, t0: float, reason: str, policy: Policy
    ) -> GuardInputResponse:
        cat = CategoryResult(category="length", score=1.0, triggered=True, action="block",
                             source="rules", evidence=[reason])
        return GuardInputResponse(
            action="block", categories=[cat], reasons=[reason],
            refusal_message=policy.refusal_message, policy_id=policy_id,
            judge=JudgeInfo(invoked=False), latency_ms=_ms(t0), request_id=rid,
        )


# ── Module-level helpers ────────────────────────────────────────────────────
def _score_action(score: float, cfg: CheckConfig) -> Action:
    if score >= cfg.block_at:
        return "block"
    if score >= cfg.flag_at:
        return "flag"
    return "allow"


def _combine_action(categories: list[CategoryResult]) -> Action:
    if any(c.action == "block" for c in categories):
        return "block"
    if any(c.action == "flag" for c in categories):
        return "flag"
    return "allow"


def _redact_for_judge(text: str, checks: Any, pii_hits: list[RuleHit]) -> str:
    cats: list[str] = []
    for c in ("secrets", "pii"):
        cfg = getattr(checks, c, None)
        if cfg and cfg.enabled and cfg.redact_before_judge:
            cats.append(c)
    if not cats:
        return text
    redacted, _ = pii.redact(text, categories=tuple(cats), hits=pii_hits)  # type: ignore[arg-type]
    return redacted


def _system_prompt_leak(output: str, system_prompt: str, min_overlap: int) -> str | None:
    """Return the leaked span if `output` shares a >= min_overlap contiguous run with the
    system prompt (case-folded), else None. The system prompt never leaves the process.

    Linear-time: hash every length-`w` window of the (bounded) system prompt into a set,
    then slide once over the output — avoids an O(len_sp * len_out) substring scan.
    """
    w = min_overlap
    if not system_prompt or w <= 0 or len(output) < w:
        return None
    out = output.casefold()
    sp = system_prompt.casefold()[:20_000]
    if len(sp) < w:
        return None
    windows = {sp[i : i + w] for i in range(len(sp) - w + 1)}
    for j in range(len(out) - w + 1):
        candidate = out[j : j + w]
        if candidate in windows:
            return candidate
    return None


def _turns_repr(turns: list[ChatTurn], limit: int) -> str:
    return json.dumps([[t.role, t.content] for t in turns[-limit:]], ensure_ascii=False)


def _docs_repr(docs: list[ContextDoc]) -> str:
    return json.dumps([d.text for d in docs], ensure_ascii=False)


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)


def _as_output(
    base: GuardInputResponse, *, sanitized_output: str | None, modified: bool,
    groundedness: GroundednessInfo,
) -> GuardOutputResponse:
    return GuardOutputResponse(
        **base.model_dump(),
        sanitized_output=sanitized_output,
        modified=modified,
        groundedness=groundedness,
    )
