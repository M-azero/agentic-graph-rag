"""Pick a chunking strategy per document.

One strategy for a whole corpus is a compromise: the separators that make
`recursive` good on prose are meaningless in a CSV, and `semantic` — which
embeds every sentence — is worth its cost on unstructured OCR output and
nowhere else.

Deterministic on purpose. An LLM router would add a model call per *file*, so a
folder of fifty documents would pay fifty calls to answer a question that a
tab-density check answers for free. The signals below are the ones that
actually change which strategy wins; anything subtler is not worth a call.

Heterogeneous chunk sizes across a corpus are safe — chunks are independent
rows, and only the *embedding model* has to stay fixed (see the fallback
validator in `config/settings.py`, which blocks changing that for exactly this
reason).
"""

from __future__ import annotations

import re

from graphrag.core.types import Document
from graphrag.ingestion.chunking.base import Chunker

# Markdown ATX headings or an HTML heading tag — structure `recursive`'s
# paragraph separators can actually use.
_HEADING = re.compile(r"(?m)^\s{0,3}#{1,6}\s+\S|<h[1-6][\s>]", re.IGNORECASE)

# Delimiters that make a line tabular. Commas alone are too weak a signal —
# ordinary prose is full of them.
_DELIMITERS = ("\t", "|", ";")

_TABULAR_SUFFIXES = (".csv", ".tsv", ".xls", ".xlsx")
_MARKUP_SUFFIXES = (".md", ".markdown", ".html", ".htm", ".rst")

# Above this, `semantic` is too expensive to justify: it embeds every sentence,
# so a long document turns one ingest into thousands of embedding rows.
_SEMANTIC_MAX_CHARS = 40_000
# Below this there is not enough text for drift detection to mean anything.
_SEMANTIC_MIN_CHARS = 1_500


def _is_ocr(document: Document) -> bool:
    """Whether this document came through OCR.

    Two shapes, because the loaders record it differently: the image loader
    sets `ocr: True`, the PDF loader counts `ocr_pages` (0 when the PDF had a
    real text layer and never needed OCR).
    """
    meta = document.metadata or {}
    return bool(meta.get("ocr")) or int(meta.get("ocr_pages") or 0) > 0


def _looks_tabular(text: str) -> bool:
    """Delimiter-heavy lines, judged on a sample rather than the whole file."""
    lines = [line for line in text.splitlines()[:60] if line.strip()]
    if len(lines) < 3:
        return False
    delimited = sum(
        1 for line in lines if any(line.count(d) >= 2 for d in _DELIMITERS)
    )
    return delimited >= max(3, int(len(lines) * 0.6))


def choose_strategy(document: Document, configured: str, available: set[str]) -> str:
    """The strategy to chunk this document with.

    Falls back to `configured` whenever no signal fires or the chosen strategy
    was not built — the configured value stays the default, never a suggestion.
    """
    source = (document.source or "").lower()
    text = document.content or ""

    def pick(strategy: str) -> str | None:
        return strategy if strategy in available else None

    # Tabular: recursive's sentence separators do nothing here, and a row split
    # mid-record is worse than a fixed window.
    if source.endswith(_TABULAR_SUFFIXES) or _looks_tabular(text):
        return pick("token") or configured

    # Real heading structure: recursive's separators have something to grip.
    if source.endswith(_MARKUP_SUFFIXES) or _HEADING.search(text[:8000]):
        return pick("recursive") or configured

    # Structureless prose of a workable size — typically OCR output, which
    # arrives as one wall of text with no paragraph breaks to split on.
    if _is_ocr(document) and _SEMANTIC_MIN_CHARS <= len(text) <= _SEMANTIC_MAX_CHARS:
        return pick("semantic") or configured

    return configured


def route(
    document: Document, chunkers: dict[str, Chunker], configured: str
) -> tuple[Chunker, str]:
    """`(chunker, strategy_name)` for one document."""
    strategy = choose_strategy(document, configured, set(chunkers))
    chunker = chunkers.get(strategy) or chunkers[configured]
    return chunker, strategy
