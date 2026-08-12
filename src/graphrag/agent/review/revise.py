"""One tool-free correction pass over a drafted answer.

Deliberately has no edge back to the checker. A revise → check → revise loop
would be unbounded in exactly the case it is most likely to trigger — a model
that cannot stop citing something it invented — and the cost of looping is paid
on the questions that are already going worst. So the reviser gets one attempt,
and anything still wrong afterwards is repaired deterministically and flagged.
"""

from __future__ import annotations

from graphrag.agent.review.citations import (
    CitationReport,
    strip_fabricated,
    verify_citations,
)
from graphrag.agent.review.prompts import REVISE_PROMPT
from graphrag.agent.review.state import FLAGGED, OK, REVISED, ReviewVerdict
from graphrag.core.logging import get_logger
from graphrag.core.messages import content_to_text

log = get_logger(__name__)

_MAX_DRAFT_CHARS = 6000
_MAX_SOURCES = 40


def _problems(report: CitationReport, verdict: ReviewVerdict | None) -> str:
    lines = []
    if report.fabricated:
        lines.append(
            "- These cited sources were never retrieved, so no claim may rest "
            "on them: " + ", ".join(sorted(report.fabricated))
        )
    if report.uncited:
        lines.append("- The answer is substantive but cites nothing.")
    if verdict is not None:
        if verdict.missing:
            lines.append(f"- Not covered by the draft: {verdict.missing}")
        for claim in verdict.unsupported:
            lines.append(f"- Unsupported claim: {claim}")
    return "\n".join(lines) or "- None recorded."


async def revise(
    llm, draft: str, available: set[str], report: CitationReport,
    verdict: ReviewVerdict | None = None, *, config: dict | None = None,
) -> tuple[str, str]:
    """Return `(answer, outcome)`.

    Falls back to the original draft on any failure: the draft is a real answer
    that was already paid for, and a failed repair must not cost the user the
    answer they would otherwise have received.
    """
    names = sorted(available)[:_MAX_SOURCES]
    prompt = REVISE_PROMPT.format(
        available="\n".join(f"- {n}" for n in names) or "(none)",
        problems=_problems(report, verdict),
        draft=(draft or "")[:_MAX_DRAFT_CHARS],
    )
    try:
        reply = await llm.ainvoke(prompt, config=config or {})
        revised = content_to_text(reply.content).strip()
    except Exception as exc:
        log.warning("revise_failed", error=str(exc))
        return draft, OK

    if not revised:
        return draft, OK

    # Re-check rather than trust: the reviser is the same class of model that
    # produced the fabrication in the first place.
    recheck = verify_citations(revised, available)
    if recheck.fabricated:
        # Last resort. Strip the tag *and* flag it — removing the tag alone
        # would turn a visible fabrication into an invisible unsupported claim,
        # which is the worse of the two failures because nothing signals it.
        log.warning("revise_left_fabrications", cited=sorted(recheck.fabricated))
        return strip_fabricated(revised, recheck.fabricated), FLAGGED

    return revised, REVISED
