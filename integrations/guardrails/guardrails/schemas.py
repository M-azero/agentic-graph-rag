"""Public request/response contract + one internal type (``RuleHit``).

Scores are **RISK values in [0.0, 1.0]** with uniform semantics across rules and the
judge: 1.0 == certain violation. This module holds data shapes only — no logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

# ── Primitives ──────────────────────────────────────────────────────────────
Action = Literal["allow", "flag", "block"]
Role = Literal["user", "assistant", "system"]
Mode = Literal["full", "fast"]
Source = Literal["rules", "judge", "combined"]
JudgeErrorKind = Literal["timeout", "api_error", "parse_error", "refusal"]

MAX_INPUT_CHARS = 100_000


class Scope(BaseModel):
    """The app's declared purpose — fed to the judge and shown in policy summaries."""

    app_description: str = ""
    allowed_topics: list[str] = Field(default_factory=list)
    deny_topics: list[str] = Field(default_factory=list)


class ChatTurn(BaseModel):
    """A prior conversation turn passed for multi-turn awareness."""

    role: Role
    content: str


class ContextDoc(BaseModel):
    """A retrieved document supporting a RAG answer (enables groundedness)."""

    id: str | None = None
    text: str
    source: str | None = None


# ── Requests ────────────────────────────────────────────────────────────────
class GuardInputRequest(BaseModel):
    """Body for ``POST /v1/guard/input`` — checked *before* the app's LLM call."""

    input: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)
    policy_id: str = "default"
    session_id: str | None = None
    context: list[ChatTurn] = Field(default_factory=list)
    mode: Mode = "full"
    metadata: dict[str, str] = Field(default_factory=dict)


class GuardOutputRequest(BaseModel):
    """Body for ``POST /v1/guard/output`` — checked *after* the app's LLM call."""

    input: str = Field(default="", max_length=MAX_INPUT_CHARS)
    output: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)
    context_docs: list[ContextDoc] = Field(default_factory=list)
    # Used ONLY for the local leakage check — never sent to the judge.
    system_prompt: str | None = None
    policy_id: str = "default"
    session_id: str | None = None
    mode: Mode = "full"
    metadata: dict[str, str] = Field(default_factory=dict)


# ── Response building blocks ────────────────────────────────────────────────
class CategoryResult(BaseModel):
    """Per-category outcome after combining rule + judge signals."""

    category: str
    score: float = Field(ge=0.0, le=1.0)
    triggered: bool
    action: Action
    source: Source
    evidence: list[str] = Field(default_factory=list)


class JudgeInfo(BaseModel):
    """What the judge did for this request (for observability)."""

    invoked: bool
    provider: str | None = None
    model: str | None = None
    cached: bool = False
    error: JudgeErrorKind | None = None
    latency_ms: float | None = None


class GroundednessInfo(BaseModel):
    """RAG groundedness result (output direction only)."""

    checked: bool
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    unsupported_claims: list[str] = Field(default_factory=list)


# ── Responses ───────────────────────────────────────────────────────────────
class GuardInputResponse(BaseModel):
    """Verdict returned by ``POST /v1/guard/input``."""

    action: Action
    categories: list[CategoryResult] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    refusal_message: str | None = None
    policy_id: str
    judge: JudgeInfo
    latency_ms: float
    request_id: str


class GuardOutputResponse(GuardInputResponse):
    """Verdict returned by ``POST /v1/guard/output`` (extends the input verdict)."""

    sanitized_output: str | None = None
    modified: bool = False
    groundedness: GroundednessInfo


# ── Meta / info endpoints ───────────────────────────────────────────────────
class PolicySummary(BaseModel):
    """One entry in ``GET /v1/policies``."""

    id: str
    description: str
    scope: Scope
    fail_mode: str
    input_checks: list[str] = Field(default_factory=list)
    output_checks: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Body for ``GET /health``."""

    status: str
    version: str
    provider: str
    model: str | None
    # Every judge link, primary first. One entry means no failover configured —
    # worth being able to see from outside, since the guard fails open and a
    # single dead provider silently stops all screening.
    chain: list[str] = Field(default_factory=list)
    policies: list[str] = Field(default_factory=list)


# ── Internal ────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class RuleHit:
    """A single deterministic-rule match (internal; never serialized to clients)."""

    rule_id: str
    category: str
    tier: Literal["block", "flag"]
    span: tuple[int, int]
    snippet: str
