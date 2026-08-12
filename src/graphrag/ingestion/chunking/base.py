"""Chunker interface: a Document -> ordered list of Chunks."""

from __future__ import annotations

import abc

from graphrag.core.types import Chunk, Document


class Chunker(abc.ABC):
    @abc.abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        ...

    @staticmethod
    def _emit(document: Document, texts: list[str]) -> list[Chunk]:
        chunks: list[Chunk] = [
            Chunk(
                doc_id=document.id,
                index=i,
                text=text,
                source=document.source,
                metadata=dict(document.metadata),
            )
            for i, text in enumerate(t for t in texts if t.strip())
        ]
        # Link each chunk to its neighbours, so a retrieved passage can be read
        # in context. Chunking severs a document at arbitrary boundaries and
        # `Chunk.id` is a one-way hash of (doc_id, index) — without recording
        # the neighbour ids here there is no way back from a chunk to the text
        # on either side of it, and an answer that straddles a boundary cannot
        # be repaired. The overlap the chunkers already apply gives the *text*
        # continuity; this is what makes it traversable.
        last = len(chunks) - 1
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["prev_id"] = chunks[i - 1].id if i > 0 else None
            chunk.metadata["next_id"] = chunks[i + 1].id if i < last else None
        return chunks
