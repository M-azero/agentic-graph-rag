"""A Redis-backed cache wrapper around any Embedder. Identical (model, text)
pairs are embedded once and reused — big savings on re-ingest and repeat queries.

The cache is an optimization and is treated as one: every Redis call is guarded,
because a cache that can fail the operation it was added to speed up is a
liability. This mattered most on ingest, where an unguarded `mget` turned a
momentary Redis blip into a failed job for a document the user had already
uploaded — while every other Redis touchpoint in the codebase degrades quietly.
"""

from __future__ import annotations

import hashlib
import json

from graphrag.core.logging import get_logger
from graphrag.embeddings.base import Embedder

log = get_logger(__name__)


class CachedEmbedder(Embedder):
    def __init__(self, inner: Embedder, redis_client, model: str, ttl: int) -> None:
        self._inner = inner
        self._redis = redis_client
        self._model = model
        self._ttl = ttl
        self.dim = inner.dim

    def _key(self, text: str) -> str:
        h = hashlib.sha1(f"{self._model}::{text}".encode()).hexdigest()
        return f"emb:{h}"

    def _get_many(self, keys: list[str]) -> list[list[float] | None]:
        """Cached vectors positionally, `None` for a miss — and all misses when
        Redis is unreachable, so the caller simply embeds everything."""
        if not keys:
            return []
        try:
            cached = self._redis.mget(keys)
        except Exception as exc:
            log.warning("embedding_cache_read_failed", error=str(exc))
            return [None] * len(keys)
        out: list[list[float] | None] = []
        for raw in cached:
            try:
                out.append(json.loads(raw) if raw else None)
            except (TypeError, ValueError):
                out.append(None)  # a corrupt entry is a miss, not a crash
        return out

    def _put_many(self, pairs: list[tuple[str, list[float]]]) -> None:
        if not pairs:
            return
        try:
            pipe = self._redis.pipeline()
            for key, vec in pairs:
                pipe.setex(key, self._ttl, json.dumps(vec))
            pipe.execute()
        except Exception as exc:
            log.warning("embedding_cache_write_failed", error=str(exc))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [self._key(t) for t in texts]
        results = self._get_many(keys)

        missing = [i for i, r in enumerate(results) if r is None]
        if missing:
            fresh = self._inner.embed_documents([texts[i] for i in missing])
            for idx, vec in zip(missing, fresh, strict=True):
                results[idx] = vec
            self._put_many([(keys[i], results[i]) for i in missing])
        # One vector per input, in order. `strict=True` above guarantees the
        # count; asserting it here as well because a short list silently
        # misaligns chunks with their embeddings downstream.
        assert len(results) == len(texts)
        return [r for r in results if r is not None]

    def embed_query(self, text: str) -> list[float]:
        key = self._key("q::" + text)
        hit = self._get_many([key])[0]
        if hit is not None:
            return hit
        vec = self._inner.embed_query(text)
        self._put_many([(key, vec)])
        return vec
