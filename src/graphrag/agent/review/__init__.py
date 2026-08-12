"""Review of a drafted answer before it reaches the user.

Two tiers, deliberately in this order:

1. **Deterministic** (`citations`) — parse the `[source: ...]` tags out of the
   draft and check them against what the tools actually surfaced. Costs no
   tokens, runs on every answer, and catches the failure that matters most: a
   citation naming a document the retriever never returned.
2. **Model** — a single tool-free critic call, spent only when tier 1 cannot
   settle the question. Advisory plus routing, never load-bearing.

Keeping the cheap tier load-bearing is what makes the expensive tier optional.
"""

from __future__ import annotations

from graphrag.agent.review.citations import (
    CitationReport,
    extract_citations,
    verify_citations,
)

__all__ = ["CitationReport", "extract_citations", "verify_citations"]
