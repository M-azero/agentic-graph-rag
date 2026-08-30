"""The structured-text loaders — content shaping, not the parsing libraries.

Each of these asserts on the *shape* a loader gives its text, because that shape
is what gets embedded: a labeled row retrieves, an unlabeled grid does not.
"""

from __future__ import annotations

import json

import pytest

from graphrag.core.errors import IngestionError
from graphrag.ingestion.loaders import supported_suffixes
from graphrag.ingestion.loaders.csv import CSVLoader
from graphrag.ingestion.loaders.html import HTMLLoader
from graphrag.ingestion.loaders.json import JSONLoader
from graphrag.ingestion.loaders.pptx import PPTXLoader
from graphrag.ingestion.loaders.xlsx import XLSXLoader


def test_csv_rows_become_labeled_sentences(tmp_path):
    path = tmp_path / "products.csv"
    path.write_text("name,price\nWidget,12.50\nGadget,3.99\n", encoding="utf-8")
    doc = CSVLoader().load(path)
    assert "name: Widget; price: 12.50" in doc.content
    assert "name: Gadget; price: 3.99" in doc.content
    assert doc.metadata["rows"] == 2


def test_tsv_uses_tab_delimiter(tmp_path):
    path = tmp_path / "t.tsv"
    path.write_text("a\tb\n1\t2\n", encoding="utf-8")
    doc = CSVLoader().load(path)
    assert "a: 1; b: 2" in doc.content


def test_html_strips_scripts_and_keeps_title(tmp_path):
    path = tmp_path / "page.html"
    path.write_text(
        "<html><head><title>Quarterly Report</title><script>alert(1)</script></head>"
        "<body><nav>menu</nav><p>Revenue grew 40%.</p></body></html>",
        encoding="utf-8",
    )
    doc = HTMLLoader().load(path)
    assert "Quarterly Report" in doc.content
    assert "Revenue grew 40%." in doc.content
    assert "alert" not in doc.content
    assert "menu" not in doc.content  # nav chrome removed


# --- JSON ------------------------------------------------------------------


def test_json_array_becomes_one_labeled_block_per_record(tmp_path):
    path = tmp_path / "people.json"
    path.write_text(
        json.dumps([{"name": "Ada", "role": "admin"}, {"name": "Bob", "role": "user"}]),
        encoding="utf-8",
    )
    doc = JSONLoader().load(path)
    assert "name: Ada; role: admin" in doc.content
    assert "name: Bob; role: user" in doc.content
    assert doc.metadata["records"] == 2


def test_json_nesting_survives_as_a_dotted_path(tmp_path):
    path = tmp_path / "nested.json"
    path.write_text(
        json.dumps({"user": {"name": "Ada"}, "tags": ["db", "graph"]}), encoding="utf-8"
    )
    doc = JSONLoader().load(path)
    assert "user.name: Ada" in doc.content
    assert "tags[0]: db" in doc.content
    assert "tags[1]: graph" in doc.content


def test_json_drops_nulls_and_empty_strings(tmp_path):
    """A sparse record should not spend its chunk on fields that aren't there."""
    path = tmp_path / "sparse.json"
    path.write_text(json.dumps({"name": "Ada", "note": None, "bio": ""}), encoding="utf-8")
    doc = JSONLoader().load(path)
    assert "name: Ada" in doc.content
    assert "note" not in doc.content
    assert "bio" not in doc.content


def test_json_keeps_false_and_zero(tmp_path):
    """Falsy is not absent — `active: False` is a fact worth retrieving."""
    path = tmp_path / "flags.json"
    path.write_text(json.dumps({"active": False, "count": 0}), encoding="utf-8")
    doc = JSONLoader().load(path)
    assert "active: False" in doc.content
    assert "count: 0" in doc.content


def test_jsonl_reads_line_by_line_and_survives_a_truncated_tail(tmp_path):
    """The format streams, so a file cut off mid-write is normal — and the
    good records above the break must not be lost with it."""
    path = tmp_path / "events.jsonl"
    lines = ['{"event": "signup"}', '{"event": "login"}', '{"event": "logo']
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    doc = JSONLoader().load(path)
    assert "event: signup" in doc.content
    assert "event: login" in doc.content
    assert doc.metadata["records"] == 2
    assert doc.metadata["skipped"] == 1  # the loss is recorded, not silent


def test_malformed_json_document_raises(tmp_path):
    """A whole-document parse failure is not partial data — it is a bad file."""
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(IngestionError):
        JSONLoader().load(path)


# --- XLSX ------------------------------------------------------------------


def test_xlsx_rows_are_labeled_and_carry_their_sheet(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "book.xlsx"
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "Products"
    sheet.append(["name", "price"])
    sheet.append(["Widget", 12.5])
    second = book.create_sheet("Staff")
    second.append(["name", "role"])
    second.append(["Ada", "admin"])
    book.save(path)

    doc = XLSXLoader().load(path)
    assert "[Products] name: Widget; price: 12.5" in doc.content
    assert "[Staff] name: Ada; role: admin" in doc.content
    assert doc.metadata["sheets"] == 2
    assert doc.metadata["rows"] == 2


def test_xlsx_names_columns_past_the_header(tmp_path):
    """A row wider than its header must not lose cells or raise."""
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "ragged.xlsx"
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(["name"])
    sheet.append(["Widget", "extra"])
    book.save(path)

    doc = XLSXLoader().load(path)
    assert "name: Widget" in doc.content
    assert "col2: extra" in doc.content


def test_xls_is_not_claimed():
    """The old binary format openpyxl cannot read stays an honest 415."""
    assert ".xls" not in supported_suffixes()
    assert ".xlsx" in supported_suffixes()


# --- PPTX ------------------------------------------------------------------


def test_pptx_slides_are_numbered_and_notes_included(tmp_path):
    pptx = pytest.importorskip("pptx")
    path = tmp_path / "deck.pptx"
    deck = pptx.Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[5])  # title only
    slide.shapes.title.text = "Quarterly Review"
    slide.notes_slide.notes_text_frame.text = "Revenue grew 40% year on year."
    deck.save(path)

    doc = PPTXLoader().load(path)
    assert "[Slide 1]" in doc.content
    assert "Quarterly Review" in doc.content
    assert "Notes: Revenue grew 40% year on year." in doc.content
    assert doc.metadata["slides"] == 1


# --- the registry ----------------------------------------------------------


def test_supported_suffixes_is_built_from_the_loaders(tmp_path):
    """The upload gate reads this, so a new loader must widen it for free."""
    for suffix in (".pdf", ".docx", ".pptx", ".xlsx", ".json", ".jsonl", ".tiff", ".rst"):
        assert suffix in supported_suffixes()
    assert ".zip" not in supported_suffixes()
