"""Check an answer's citations against what retrieval actually returned.

No model calls: this is a regex and two set operations, so it can run on every
answer without touching the token budget. It is the load-bearing half of
review — the critic that follows is only spent when this cannot settle things.

What it can and cannot prove: the `[source: ...]` tags carry a *document name*
(or one of the graph labels), not a chunk id, so this validates provenance at
document granularity. It cannot tell you a cited document genuinely supports
the sentence it is attached to — that needs a model. It can tell you, with
certainty and for free, that a cited name was never retrieved at all, which is
the more serious failure and the one nothing checked until now.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from graphrag.agent.prompts import CLOSED_DOMAIN_REFUSAL
from graphrag.agent.tools import _src
from graphrag.core.types import RetrievedChunk

# Mirrors the tag `_format` and `_graph_data` emit: "[source: <name>]".
# `_src` caps names at 200 chars; 220 leaves room for the truncation marker
# without letting a runaway match swallow a paragraph.
_TAG = re.compile(r"\[source:\s*([^\]\n]{1,220})\]")

# Below this a draft is a refusal, an apology or an acknowledgement — the kind
# of answer that legitimately cites nothing. Above it, silence is suspicious.
_SUBSTANTIVE_CHARS = 240


@dataclass(frozen=True)
class CitationReport:
    """What the draft cited, and how that compares to what was retrieved."""

    cited: frozenset[str] = frozenset()
    available: frozenset[str] = frozenset()
    fabricated: frozenset[str] = frozenset()
    unused: frozenset[str] = frozenset()
    uncited: bool = False
    refusal: bool = False
    findings: tuple[str, ...] = field(default=())

    @property
    def clean(self) -> bool:
        """True when the draft is citing honestly and citing at all.

        `unused` deliberately does not count: retrieval over-fetches by design,
        so surfacing more than the answer needed is correct behaviour, not a
        defect.
        """
        return not self.fabricated and not self.uncited


def source_key(source: str) -> str:
    """Canonical form of a source name, for comparing against `report.cited`.

    A thin public alias for the normalization the tools already apply when they
    write a tag. Callers outside this package should use it rather than reaching
    for the private helper, so there is exactly one definition of "same source".
    """
    return _src(source)


def extract_citations(text: str) -> set[str]:
    """The source names a draft claims to be citing, normalized.

    Normalized through `_src` — the same function that wrote the tags — so the
    comparison is exact. Doing it by hand here would reintroduce the mismatch
    the shared helper exists to prevent: `_src` truncates at 200 characters,
    collapses newlines and rewrites double quotes, so a document whose name
    contains a quote would otherwise read as fabricated. `_src` is idempotent,
    so applying it to an already-written tag is safe.
    """
    return {_src(m.group(1).strip()) for m in _TAG.finditer(text or "")}


def is_refusal(text: str) -> bool:
    """Whether the draft is the closed-domain refusal.

    Compared on collapsed whitespace rather than exactly: the prompt asks for
    the refusal verbatim, but models reflow it, and a reflowed refusal is still
    a refusal. Containment rather than equality for the same reason — some
    models append a trailing newline or a stray sentence.
    """
    if not text:
        return False
    squash = " ".join(text.split())
    return " ".join(CLOSED_DOMAIN_REFUSAL.split()) in squash


def available_sources(
    chunks: Iterable[RetrievedChunk], labels: Iterable[str] = ()
) -> set[str]:
    """Everything the draft is allowed to cite.

    Chunk source names plus the graph labels (`knowledge-graph`,
    `community-summaries`) — three of the eight tools answer from the graph and
    surface no chunk at all, so without the labels every graph-grounded
    citation would read as fabricated.

    Takes the two collections rather than a `SourceSink` so it works equally on
    a live run and on a finished `QueryResult`, which is what the API holds.
    """
    return {_src(c.source) for c in chunks} | set(labels)


def verify_citations(draft: str, available: Iterable[str]) -> CitationReport:
    """Compare a draft's citations against what the tools surfaced."""
    available = set(available)

    if is_refusal(draft):
        # A refusal cites nothing by design, and its sources are dropped
        # downstream — reporting "uncited" here would be noise.
        return CitationReport(
            available=frozenset(available),
            refusal=True,
            findings=("refusal",),
        )

    cited = extract_citations(draft)
    fabricated = cited - available
    unused = available - cited
    uncited = not cited and len((draft or "").strip()) > _SUBSTANTIVE_CHARS

    findings: list[str] = []
    if fabricated:
        findings.append("fabricated")
    if uncited:
        findings.append("uncited")

    return CitationReport(
        cited=frozenset(cited),
        available=frozenset(available),
        fabricated=frozenset(fabricated),
        unused=frozenset(unused),
        uncited=uncited,
        refusal=False,
        findings=tuple(findings),
    )


def order_by_citation(
    sources: list[RetrievedChunk], report: CitationReport
) -> list[RetrievedChunk]:
    """Cited chunks first, each group keeping its retrieval order.

    `sources` is everything every tool surfaced, which is usually more than the
    answer used. Sorting the ones the answer actually cited to the front is what
    lets a client show evidence-used above also-found instead of one flat list.
    """
    if not report.cited:
        return sources
    cited = [c for c in sources if _src(c.source) in report.cited]
    rest = [c for c in sources if _src(c.source) not in report.cited]
    return cited + rest


def strip_fabricated(text: str, fabricated: frozenset[str] | set[str]) -> str:
    """Remove citation tags naming sources that were never retrieved.

    Last resort, after a revise pass failed to fix them. The caller must also
    mark the answer as flagged: removing the tag alone would turn a visible
    fabrication into an invisible unsupported claim, which is worse — the
    reader loses the one signal that something was wrong.
    """
    if not fabricated:
        return text

    def _drop(match: re.Match[str]) -> str:
        return "" if _src(match.group(1).strip()) in fabricated else match.group()

    # Collapse the double space a removed inline tag leaves behind, but keep
    # paragraph breaks intact.
    return re.sub(r"[ \t]{2,}", " ", _TAG.sub(_drop, text))
