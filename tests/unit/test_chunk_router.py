"""Per-document chunk strategy routing.

Two properties matter beyond the routing table itself: the configured strategy
stays the default whenever no signal fires, and building every strategy loads
the tokenizer once rather than once per document.
"""

from unittest.mock import patch

import pytest

from graphrag.config.settings import ChunkingCfg, EmbeddingCfg
from graphrag.core.types import Document
from graphrag.ingestion.chunking import build_chunkers
from graphrag.ingestion.chunking.router import choose_strategy, route

ALL = {"recursive", "token", "semantic"}


def _doc(source="notes.txt", content="hello world", **meta) -> Document:
    return Document(source=source, content=content, metadata=meta)


def _pick(document, configured="recursive", available=ALL) -> str:
    return choose_strategy(document, configured, available)


# ── tabular ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["data.csv", "sheet.xlsx", "export.TSV"])
def test_tabular_suffixes_use_token_windows(name):
    """Recursive's sentence separators mean nothing in a table, and a split
    mid-record is worse than a fixed window."""
    assert _pick(_doc(name)) == "token"


def test_delimiter_heavy_content_is_detected_without_a_suffix():
    body = "\n".join("a|b|c|d" for _ in range(10))
    assert _pick(_doc("dump.txt", body)) == "token"


def test_prose_full_of_commas_is_not_mistaken_for_a_table():
    """Commas alone are too weak a signal — ordinary writing is full of them."""
    body = "\n".join(
        "First, we looked at the data, then, after some thought, we wrote it up."
        for _ in range(10)
    )
    assert _pick(_doc("essay.txt", body)) == "recursive"


def test_a_couple_of_delimited_lines_is_not_a_table():
    assert _pick(_doc("notes.txt", "a|b\nc|d")) == "recursive"


# ── markup ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["readme.md", "page.html", "doc.rst"])
def test_markup_suffixes_use_recursive(name):
    assert _pick(_doc(name), configured="token") == "recursive"


def test_markdown_headings_are_detected_without_a_suffix():
    body = "# Title\n\nSome prose here.\n\n## Section\n\nMore prose."
    assert _pick(_doc("notes.txt", body), configured="token") == "recursive"


def test_a_hash_that_is_not_a_heading_does_not_count():
    body = "The issue #42 was closed.\nSee also #tags and #hashtags in the text."
    assert _pick(_doc("notes.txt", body), configured="token") == "token"


# ── OCR prose ──────────────────────────────────────────────────────────────


def test_ocr_prose_of_a_workable_size_uses_semantic():
    """OCR arrives as a wall of text with no paragraph breaks to split on."""
    assert _pick(_doc("scan.png", "word " * 1000, ocr=True)) == "semantic"


def test_pdf_ocr_pages_count_as_ocr():
    """The PDF loader records a page count, not a boolean."""
    assert _pick(_doc("scan.pdf", "word " * 1000, ocr_pages=3)) == "semantic"


def test_a_pdf_with_a_real_text_layer_is_not_treated_as_ocr():
    assert _pick(_doc("clean.pdf", "word " * 1000, ocr_pages=0)) == "recursive"


def test_huge_ocr_documents_skip_semantic():
    """Semantic embeds every sentence; on a long document that turns one ingest
    into thousands of embedding rows."""
    assert _pick(_doc("scan.png", "word " * 40_000, ocr=True)) == "recursive"


def test_tiny_ocr_documents_skip_semantic():
    assert _pick(_doc("scan.png", "a few words", ocr=True)) == "recursive"


# ── fallbacks ──────────────────────────────────────────────────────────────


def test_plain_prose_keeps_the_configured_strategy():
    assert _pick(_doc("notes.txt", "Just some ordinary prose."), "token") == "token"


def test_an_unavailable_strategy_falls_back_to_the_configured_one():
    """A deployment with no embedder still routes between what it can build,
    rather than failing every ingest."""
    got = choose_strategy(
        _doc("scan.png", "word " * 1000, ocr=True), "recursive", {"recursive", "token"}
    )
    assert got == "recursive"


def test_route_returns_the_chunker_and_its_name():
    chunkers = {"recursive": object(), "token": object()}
    chunker, name = route(_doc("data.csv"), chunkers, "recursive")
    assert name == "token"
    assert chunker is chunkers["token"]


def test_route_falls_back_when_the_choice_was_not_built():
    chunkers = {"recursive": object()}
    chunker, name = route(_doc("data.csv"), chunkers, "recursive")
    assert chunker is chunkers["recursive"]
    assert name == "recursive"


# ── construction ───────────────────────────────────────────────────────────


def test_building_every_strategy_loads_the_tokenizer_once():
    """The reason `build_chunkers` exists: `build_chunker` resolves a tokenizer
    on every call, so routing through it per document would re-resolve per
    file."""
    # A non-None sentinel: `TokenChunker` rejects a None tokenizer outright,
    # so patching it to None would test the skip path instead of this one.
    with patch(
        "graphrag.ingestion.chunking.load_hf_tokenizer", return_value=object()
    ) as loader:
        built = build_chunkers(ChunkingCfg(), EmbeddingCfg(), embedder=object())

    assert loader.call_count == 1
    assert set(built) == ALL


def test_strategies_that_cannot_be_built_are_skipped():
    """No embedder means no semantic chunker — but the other two still work."""
    with patch("graphrag.ingestion.chunking.load_hf_tokenizer", return_value=object()):
        built = build_chunkers(ChunkingCfg(), EmbeddingCfg(), embedder=None)

    assert "semantic" not in built
    assert {"recursive", "token"} <= set(built)
