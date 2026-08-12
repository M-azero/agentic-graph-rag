"""The critic: one tool-free model call, spent only when the free checks can't
settle the answer.

Two rules shape everything here.

**It is gated.** A draft that cites honestly, after a single tool call, does not
get a critic call — the deterministic report already answered the question. That
gate is the affordability argument for the whole loop, so it is tested directly.

**It fails open.** A critic that can fail a request is worse than no critic: the
answer already exists and was already paid for. Every failure path — a dead
provider, unparseable output, a model that ignored the format — resolves to
"ship", the same posture `GuardrailsClient` takes.
"""

from __future__ import annotations

import json

from graphrag.agent.review.citations import CitationReport
from graphrag.agent.review.prompts import CRITIC_PROMPT
from graphrag.agent.review.state import RETRIEVE_MORE, REVISE, SHIP, ReviewVerdict
from graphrag.core.logging import get_logger
from graphrag.core.messages import content_to_text

log = get_logger(__name__)

_ACTIONS = {SHIP, REVISE, RETRIEVE_MORE}

# Caps on what the critic sees. It reviews structure, not content, so a long
# draft adds tokens without adding signal — and the source list is names only.
_MAX_DRAFT_CHARS = 4000
_MAX_SOURCES = 40


def needs_critic(
    report: CitationReport, tool_calls: list, draft: str, *,
    free_tool_calls: int = 1, free_chars: int = 1200,
) -> bool:
    """Whether this draft is worth a model call.

    A short answer that cites honestly after one tool call has nothing left to
    check that a model would catch and the regex would not. Anything longer, or
    anything the deterministic pass flagged, gets looked at.
    """
    if report.refusal:
        return False  # a correct refusal is a good answer; ship it
    if not report.clean:
        return True  # fabricated or uncited — the critic decides the repair
    return len(tool_calls) > free_tool_calls or len(draft or "") > free_chars


def _first_json_object(text: str) -> dict | None:
    """Pull the first balanced {...} out of a model reply.

    Models wrap JSON in prose or a code fence however firmly they are asked not
    to. Scanning for a balanced object tolerates all of that without a
    provider-specific structured-output path — `with_structured_output` routes
    through `bind_tools(tool_choice=...)`, which is not uniformly supported
    across the providers this runs on, and the shipped default is a local 7B.
    """
    start = text.find("{")
    while start != -1:
        depth, in_string, escaped = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : i + 1])
                    except (ValueError, TypeError):
                        break
                    return parsed if isinstance(parsed, dict) else None
        start = text.find("{", start + 1)
    return None


def parse_verdict(text: str) -> ReviewVerdict:
    """A verdict from raw model output, defaulting to ship on anything odd."""
    data = _first_json_object(text or "")
    if data is None:
        log.warning("critic_unparseable", length=len(text or ""))
        return ReviewVerdict(action=SHIP, reason="critic output unparseable")

    action = str(data.get("action", SHIP)).strip().lower()
    if action not in _ACTIONS:
        log.warning("critic_unknown_action", action=action[:40])
        action = SHIP

    raw_unsupported = data.get("unsupported") or []
    if isinstance(raw_unsupported, str):
        raw_unsupported = [raw_unsupported]

    return ReviewVerdict(
        action=action,
        complete=bool(data.get("complete", True)),
        missing=str(data.get("missing") or "")[:500],
        unsupported=tuple(str(u)[:200] for u in raw_unsupported[:10]),
        reason=str(data.get("reason") or "")[:300],
    )


def _summarize(report: CitationReport) -> str:
    if report.refusal:
        return "The draft is the closed-domain refusal."
    parts = []
    if report.fabricated:
        parts.append(
            "Cites sources that were never retrieved: "
            + ", ".join(sorted(report.fabricated))
        )
    if report.uncited:
        parts.append("A substantive answer with no citations at all.")
    if not parts:
        parts.append("Citations all name retrieved sources.")
    return " ".join(parts)


async def run_critic(
    llm, question: str, draft: str, available: set[str], report: CitationReport,
    *, config: dict | None = None,
) -> ReviewVerdict:
    """Review one draft. Never raises."""
    names = sorted(available)[:_MAX_SOURCES]
    prompt = CRITIC_PROMPT.format(
        question=question[:2000],
        draft=(draft or "")[:_MAX_DRAFT_CHARS],
        available="\n".join(f"- {n}" for n in names) or "(none)",
        report=_summarize(report),
    )
    try:
        # `config` carries the request's TokenMeter. Without it the critic's
        # tokens are invisible to metering and quota — spent, but free to the
        # user, which is the exact bug `record_answer_tokens` was written to
        # close on the streaming path.
        reply = await llm.ainvoke(prompt, config=config or {})
    except Exception as exc:
        log.warning("critic_failed", error=str(exc))
        return ReviewVerdict(action=SHIP, reason="critic unavailable")
    return parse_verdict(content_to_text(reply.content))
