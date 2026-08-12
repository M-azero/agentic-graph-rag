"""State and verdict types for the review loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

from graphrag.agent.review.citations import CitationReport
from graphrag.core.types import RetrievedChunk
from graphrag.retrieval.plan import RetrievalPlan

# What `check` can decide to do next.
SHIP = "ship"
REVISE = "revise"
RETRIEVE_MORE = "retrieve_more"

# How the answer ended up, for the client and the logs.
OK = "ok"
REVISED = "revised"
REFUSED = "refused"
FLAGGED = "flagged"
ESCALATED = "escalated"


@dataclass(frozen=True)
class ReviewVerdict:
    """The critic's reading of a draft.

    `action` is the only field that changes control flow. The rest explain the
    decision — to the reviser, to the client, and to whoever reads the logs
    deciding whether this loop is worth its tokens.
    """

    action: str = SHIP
    complete: bool = True
    missing: str = ""
    unsupported: tuple[str, ...] = ()
    reason: str = ""


class ReviewState(TypedDict, total=False):
    """LangGraph channel state for one reviewed answer.

    Every key the nodes read or write must be declared here — a StateGraph only
    propagates the channels its schema names, so an undeclared key is silently
    dropped between nodes rather than failing loudly.
    """

    # -- the request
    question: str
    style: str | None
    preset: str | None
    thread_id: str
    user_id: str | None
    model: Any
    meter: Any
    # Carries the request's TokenMeter to the critic and reviser, so their
    # tokens land on the same bill as the answer they reviewed.
    llm_config: dict[str, Any]

    # -- retrieval
    plan: RetrievalPlan
    sink: Any  # SourceSink, shared across research passes
    window_gain: list[RetrievedChunk]

    # -- what research produced
    draft: str
    tool_calls: list[dict[str, Any]]
    sources: list[RetrievedChunk]
    labels: list[str]

    # -- what review concluded
    report: CitationReport
    verdict: ReviewVerdict | None

    rounds: int
    action: str
    answer: str
    outcome: str


@dataclass
class ReviewOutcome:
    """What the review loop hands back to the query service."""

    answer: str = ""
    sources: list[RetrievedChunk] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    outcome: str = OK
    rounds: int = 0
    critic_called: bool = False
    report: CitationReport | None = None
    verdict: ReviewVerdict | None = None
