"""Chunks form a linked list, so a severed passage can be read back in context.

`Chunk.id` is a one-way hash of (doc_id, index), so nothing could walk from a
retrieved chunk to the text on either side of it. The chunkers already overlap
their text; these pin the part that makes the overlap *traversable*.
"""

import pytest

from graphrag.agent.tools import build_tools, collect_sources
from graphrag.core.types import Chunk, Document, RetrievedChunk
from graphrag.ingestion.chunking.base import Chunker


class _Fixed(Chunker):
    """Emits exactly the texts it is given, through the real `_emit`."""

    def __init__(self, texts):
        self._texts = texts

    def chunk(self, document: Document) -> list[Chunk]:
        return self._emit(document, self._texts)


def _doc() -> Document:
    return Document(source="report.pdf", content="unused", metadata={"origin": "test"})


# ── the links themselves ───────────────────────────────────────────────────


def test_each_chunk_points_at_its_neighbours():
    chunks = _Fixed(["one", "two", "three"]).chunk(_doc())
    ids = [c.id for c in chunks]

    assert [c.metadata["prev_id"] for c in chunks] == [None, ids[0], ids[1]]
    assert [c.metadata["next_id"] for c in chunks] == [ids[1], ids[2], None]


def test_the_ends_are_open():
    """A dangling link would send `chunk_window` looking for a node that was
    never written."""
    chunks = _Fixed(["only"]).chunk(_doc())
    assert chunks[0].metadata["prev_id"] is None
    assert chunks[0].metadata["next_id"] is None


def test_chunk_index_is_recorded():
    chunks = _Fixed(["a", "b", "c"]).chunk(_doc())
    assert [c.metadata["chunk_index"] for c in chunks] == [0, 1, 2]
    assert [c.index for c in chunks] == [0, 1, 2]


def test_links_skip_blank_texts_without_leaving_a_hole():
    """`_emit` drops whitespace-only pieces, so indices must be assigned after
    the filter — otherwise a link points at an id that was never stored."""
    chunks = _Fixed(["a", "   ", "b"]).chunk(_doc())
    assert len(chunks) == 2
    assert chunks[0].metadata["next_id"] == chunks[1].id
    assert chunks[1].metadata["prev_id"] == chunks[0].id


def test_document_metadata_is_not_shared_between_chunks():
    """The link fields are per-chunk. One shared dict would give every chunk
    the last chunk's neighbours."""
    chunks = _Fixed(["a", "b"]).chunk(_doc())
    assert chunks[0].metadata is not chunks[1].metadata
    assert chunks[0].metadata["origin"] == "test"


# ── the store contract ─────────────────────────────────────────────────────


class _Store:
    """Records what the pipeline asks of the graph store."""

    def __init__(self):
        self.linked: list[list[Chunk]] = []
        self.window_calls: list[tuple] = []
        self.window_result: list[RetrievedChunk] = []

    def link_chunk_sequence(self, chunks):
        self.linked.append(list(chunks))

    def chunk_window(self, chunk_ids, before=1, after=1):
        self.window_calls.append((list(chunk_ids), before, after))
        return self.window_result


def test_link_chunk_sequence_receives_the_chunks_in_order():
    store = _Store()
    chunks = _Fixed(["a", "b", "c"]).chunk(_doc())
    store.link_chunk_sequence(chunks)
    assert [c.index for c in store.linked[0]] == [0, 1, 2]


# ── the read_around tool ───────────────────────────────────────────────────


class _Ctx:
    vector = hybrid = embedder = None
    top_k = 8
    graph_hops = 2

    def __init__(self, store):
        self.graph = store


def _read_around(store):
    return {t.name: t for t in build_tools(_Ctx(store))}["read_around"].func


def test_read_around_asks_for_one_chunk_either_side():
    store = _Store()
    _read_around(store)("abc123")
    assert store.window_calls == [(["abc123"], 1, 1)]


def test_read_around_accepts_several_ids():
    store = _Store()
    _read_around(store)("abc123, def456")
    assert store.window_calls == [(["abc123", "def456"], 1, 1)]


@pytest.mark.parametrize("arg", ["", "   ", ","])
def test_read_around_with_no_usable_id_does_not_hit_the_store(arg):
    store = _Store()
    assert _read_around(store)(arg) == "No results."
    assert store.window_calls == []


def test_read_around_records_what_it_surfaced():
    """Neighbours become citable evidence like anything else a tool returns."""
    store = _Store()
    store.window_result = [
        RetrievedChunk(
            chunk_id="n1", text="the missing sentence", source="report.pdf",
            score=0.0, retriever="window",
        )
    ]
    with collect_sources() as sink:
        out = _read_around(store)("abc123")
        assert [c.chunk_id for c in sink.chunks] == ["n1"]
    assert "[source: report.pdf]" in out
    assert "the missing sentence" in out


def test_chunk_ids_are_exposed_so_the_tool_can_be_called():
    """`read_around` takes a chunk id, so results have to carry one — otherwise
    the tool costs schema tokens on every turn and can never be used."""
    from graphrag.agent.tools import _format

    out = _format(
        [
            RetrievedChunk(
                chunk_id="abc123", text="body", source="report.pdf",
                score=1.0, retriever="test",
            )
        ]
    )
    assert "[chunk: abc123]" in out
    # And it must not disturb citation parsing.
    from graphrag.agent.review.citations import extract_citations

    assert extract_citations(out) == {"report.pdf"}
