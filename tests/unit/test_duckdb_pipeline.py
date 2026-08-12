"""The ingest pipeline against the DuckDB vector provider.

The provider swap is only safe if the rest of the pipeline still behaves: chunk
nodes must still reach the graph (fulltext and MENTIONS need them), re-ingest
must still replace rather than duplicate, and retrieval must still come back
scoped to the right tenant. Graph and LLM pieces are stubbed; the vector store
is the real DuckDB one.
"""

from __future__ import annotations

import pytest

from graphrag.config.settings import Settings
from graphrag.core.types import Document
from graphrag.pipelines.ingest import IngestPipeline
from graphrag.storage.vector.duckdb_store import DuckDBVectorStore, close_all


@pytest.fixture(autouse=True)
def _close_connections():
    yield
    close_all()


class _Embedder:
    """Bag-of-characters vectors: deterministic, and similar texts land near
    each other, which is all retrieval ordering needs here."""

    dim = 8

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    @staticmethod
    def _vec(text: str):
        out = [0.0] * 8
        for ch in text.lower():
            if ch.isalpha():
                out[(ord(ch) - 97) % 8] += 1.0
        return out or [1.0] * 8


class _Chunker:
    def chunk(self, document: Document):
        from graphrag.core.types import Chunk

        lines = [ln for ln in document.content.splitlines() if ln.strip()]
        return [
            Chunk(doc_id=document.id, index=i, text=ln, source=document.source)
            for i, ln in enumerate(lines)
        ]


class _GraphStore:
    """Records what the pipeline sends it, so the test can assert the chunk
    nodes still arrive when vectors live outside Neo4j."""

    def __init__(self):
        self.chunks: list = []
        self.deleted: list[str] = []
        self.linked: list = []

    def setup(self):
        pass

    def delete_document(self, source):
        self.deleted.append(source)
        return 0

    def upsert_chunks(self, chunks):
        self.chunks.extend(chunks)

    def link_chunk_sequence(self, chunks):
        self.linked.append(list(chunks))


class _Tenant:
    def __init__(self, vector_store, graph_store, user_id="alice", corpus=None):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.user_id = user_id
        # Equal to `user_id` unless a shelf is named — which is exactly what the
        # real Tenant does for the default shelf.
        self.corpus = corpus if corpus is not None else user_id


class _Container:
    """Just the surface IngestPipeline touches."""

    def __init__(self, tmp_path, corpus="alice"):
        self.settings = Settings()
        self.settings.storage.vector.provider = "duckdb"
        self.settings.ingestion.extract_graph = False  # no LLM in this test
        self.embedder = _Embedder()
        self.chunker = _Chunker()
        self.ocr = None
        self.vector_store = DuckDBVectorStore(tmp_path, "neo4j", corpus)
        self.vector_store.setup(self.embedder.dim)
        self.graph_store = _GraphStore()
        self._tenant = _Tenant(self.vector_store, self.graph_store, corpus)

    def tenant(self, user_id=None, shelf=None):
        return self._tenant


def _write(tmp_path, name: str, body: str):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def test_ingest_writes_vectors_and_is_retrievable(tmp_path):
    c = _Container(tmp_path / "vec")
    doc = _write(tmp_path, "notes.txt", "alpha beta\ngamma delta\n")
    stats = IngestPipeline(c).run(doc)

    assert stats.documents == 1
    assert stats.chunks == 2
    assert c.vector_store.count() == 2
    hits = c.vector_store.query(c.embedder.embed_query("alpha beta"), k=1)
    assert hits and hits[0].text == "alpha beta"


def test_chunk_nodes_still_reach_the_graph(tmp_path):
    """Vectors moved out of Neo4j, but fulltext search and MENTIONS edges still
    depend on the chunk nodes being there."""
    c = _Container(tmp_path / "vec")
    IngestPipeline(c).run(_write(tmp_path, "notes.txt", "alpha\nbeta\n"))
    assert len(c.graph_store.chunks) == 2


def test_chunks_are_linked_under_a_non_neo4j_vector_provider(tmp_path):
    """The :NEXT chain is laid down by the pipeline, not by whichever store
    happened to write the nodes — so it exists under duckdb too. Linking inside
    `upsert_chunks` would have made the chain neo4j-only."""
    c = _Container(tmp_path / "vec")
    IngestPipeline(c).run(_write(tmp_path, "notes.txt", "alpha\nbeta\n"))
    assert [ch.index for ch in c.graph_store.linked[0]] == [0, 1]


def test_reingest_replaces_instead_of_duplicating(tmp_path):
    c = _Container(tmp_path / "vec")
    path = _write(tmp_path, "notes.txt", "one\ntwo\nthree\n")
    IngestPipeline(c).run(path)
    assert c.vector_store.count() == 3

    # A shortened document must not leave its old tail chunks behind.
    path.write_text("one\n", encoding="utf-8")
    IngestPipeline(c).run(path)
    assert c.vector_store.count() == 1
    assert c.graph_store.deleted  # the pipeline asked for the old rows first


def test_each_tenant_gets_its_own_database_file(tmp_path):
    root = tmp_path / "vec"
    alice, bob = _Container(root, "alice"), _Container(root, "bob")
    IngestPipeline(alice).run(_write(tmp_path, "a.txt", "alpha secret\n"))

    assert alice.vector_store.path != bob.vector_store.path
    assert alice.vector_store.path.exists() and bob.vector_store.path.exists()
    assert bob.vector_store.count() == 0
    assert bob.vector_store.query(bob.embedder.embed_query("alpha secret"), k=5) == []


def test_empty_document_is_skipped(tmp_path):
    c = _Container(tmp_path / "vec")
    stats = IngestPipeline(c).run(_write(tmp_path, "blank.txt", "   \n\n"))
    assert stats.documents == 0
    assert c.vector_store.count() == 0
