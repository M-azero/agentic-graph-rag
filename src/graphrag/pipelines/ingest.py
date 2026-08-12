"""Ingestion pipeline: file -> text -> chunks -> embeddings + knowledge graph.

Documents are written into a single user's isolated namespace. The embedder,
chunker, OCR, and extractor are shared across all users; only the target stores
are user-specific.

    load ── chunk ── embed ── store vectors (user's namespace)
                        └──── extract entities/relations ── store graph ── link chunks
                                        └── resolve duplicate entities
                                        └── rebuild community summaries

Re-ingesting a source is an *update*: its previous chunks are deleted first, so
a shortened document can't leave stale tail chunks behind (chunk ids are
(doc, index)-stable, which without the delete makes old indexes immortal).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from graphrag.container import Container, Tenant
from graphrag.core.logging import get_logger
from graphrag.ingestion.chunking.router import route
from graphrag.ingestion.enrich import build_communities, resolve_entities
from graphrag.ingestion.loaders import iter_documents

log = get_logger(__name__)


@dataclass
class IngestStats:
    documents: int = 0
    chunks: int = 0
    entities: int = 0
    relations: int = 0
    files: list[str] = field(default_factory=list)
    # Chunks whose extraction call failed outright (provider down, key denied).
    # The vectors for those chunks are stored and searchable; only their graph
    # structure is missing, which is why this is "partial" and not a failure.
    extraction_failures: int = 0

    @property
    def partial(self) -> bool:
        return self.extraction_failures > 0


class LimitExceededError(RuntimeError):
    """The ingest would push the user past a quota. Surfaced as a failed job
    with a message the UI can show, not a stack trace."""


class IngestPipeline:
    def __init__(self, container: Container, max_chunks: int | None = None) -> None:
        self._c = container
        self._max_chunks = max_chunks

    def _chunk(self, document) -> list:
        """Split one document, routing per document when that is enabled.

        The strategy lands in each chunk's metadata: a corpus chunked several
        ways is only debuggable if you can tell which document got which, and
        re-chunking one source means re-ingesting it (the writes below replace
        rather than patch).
        """
        cfg = self._c.settings.chunking
        if not cfg.route_per_document:
            return self._c.chunker.chunk(document)

        chunker, strategy = route(document, self._c.chunkers, cfg.strategy)
        chunks = chunker.chunk(document)
        for chunk in chunks:
            chunk.metadata["chunk_strategy"] = strategy
        if strategy != cfg.strategy:
            log.info("chunk_strategy_routed", source=document.source, strategy=strategy)
        return chunks

    def _check_capacity(self, tenant: Tenant, incoming: int) -> None:
        """Refuse a document that would exceed the chunk allowance.

        Checked per document rather than once up front: a folder ingest should
        store what fits and stop at the boundary, not fail wholesale.

        `count()` reports this *shelf's* chunks, not the account's, because a
        vector store only ever sees one corpus. The caller closes that gap by
        passing `max_chunks` already reduced by what the user's other shelves
        hold — see `api.routers.ingest._shelf_chunk_budget`.
        """
        if not self._max_chunks:
            return
        count = getattr(tenant.vector_store, "count", None)
        if count is None:
            return  # backend can't report a total; the file/storage caps still apply
        current = count()
        if current + incoming > self._max_chunks:
            raise LimitExceededError(
                f"Storage limit reached: {current} of {self._max_chunks} chunks used. "
                "Delete a document to free space."
            )

    def run(
        self, path: str | Path, user_id: str | None = None, shelf: str | None = None
    ) -> IngestStats:
        """Ingest into one user's shelf. `shelf` is the slug; empty or None is
        their default shelf, whose corpus is the bare tenant id."""
        c = self._c
        tenant = c.tenant(user_id, shelf)  # resolves + prepares the namespace
        stats = IngestStats()
        extract = c.settings.ingestion.extract_graph

        for document in iter_documents(
            path, ocr=c.ocr, min_text_chars=c.settings.ocr.min_text_chars
        ):
            if not document.content.strip():
                log.warning("empty_document", source=document.source)
                continue

            chunks = self._chunk(document)
            if not chunks:
                continue

            replaced = tenant.graph_store.delete_document(document.source)
            replaced += tenant.vector_store.delete_source(document.source)
            if replaced:
                log.info("reingest_replaced", source=document.source, removed=replaced)

            # After the replace, so re-ingesting an existing document is
            # measured against the space it is about to free.
            self._check_capacity(tenant, len(chunks))

            embeddings = c.embedder.embed_documents([ch.text for ch in chunks])
            for ch, vec in zip(chunks, embeddings, strict=True):
                ch.embedding = vec
            tenant.vector_store.upsert(chunks)
            if c.settings.storage.vector.provider != "neo4j":
                # Vectors live elsewhere, but fulltext search and MENTIONS
                # edges still need the chunk nodes in the graph.
                tenant.graph_store.upsert_chunks(chunks)
            # After both writers, because either one may have authored the
            # nodes: the Neo4j vector store writes them when vectors live in
            # the graph, `upsert_chunks` when they don't. Linking here is what
            # makes the :NEXT chain exist under every vector provider.
            tenant.graph_store.link_chunk_sequence(chunks)

            if extract:
                self._build_graph(tenant, chunks, stats)

            stats.documents += 1
            stats.chunks += len(chunks)
            stats.files.append(document.source)
            log.info(
                "ingested", user=tenant.user_id, corpus=tenant.corpus,
                source=document.source, chunks=len(chunks),
            )

        if extract and stats.documents:
            self._enrich(tenant)
        return stats

    def _build_graph(self, tenant: Tenant, chunks, stats: IngestStats) -> None:
        # Extraction is one LLM call per chunk — by far the slowest part of
        # ingest — so those calls run concurrently. Writes stay serial:
        # concurrent MERGEs on the same entity keys only fight for locks.
        workers = max(1, self._c.settings.ingestion.max_concurrency)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            extracted = list(pool.map(self._c.extractor.extract, [c.text for c in chunks]))

        for chunk, result in zip(chunks, extracted, strict=True):
            if result.failed:
                stats.extraction_failures += 1
                continue
            if not result.entities:
                continue
            tenant.graph_store.add_entities(result.entities)
            tenant.graph_store.add_relations(result.relations)
            tenant.graph_store.link_chunk_entities(
                chunk.id, [e.key for e in result.entities]
            )
            stats.entities += len(result.entities)
            stats.relations += len(result.relations)

    def _enrich(self, tenant: Tenant) -> None:
        cfg = self._c.settings.ingestion
        try:
            resolve_entities(tenant.graph_store, self._c.embedder, cfg.resolve_entities)
        except Exception as exc:  # enrichment must never fail the ingest
            log.warning("entity_resolution_failed", error=str(exc))
        try:
            build_communities(
                tenant.graph_store, self._c.embedder, self._c.extractor_llm,
                cfg.communities,
            )
        except Exception as exc:
            log.warning("community_build_failed", error=str(exc))
