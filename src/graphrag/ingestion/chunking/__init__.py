"""Chunker factory. `chunking.strategy` picks recursive | token | semantic."""

from __future__ import annotations

from graphrag.config.settings import ChunkingCfg, EmbeddingCfg
from graphrag.core.errors import ConfigError
from graphrag.embeddings.base import Embedder
from graphrag.ingestion.chunking.base import Chunker
from graphrag.ingestion.chunking.recursive import RecursiveChunker
from graphrag.ingestion.chunking.semantic import SemanticChunker
from graphrag.ingestion.chunking.token import TokenChunker
from graphrag.ingestion.chunking.tokenizer import TokenCounter, load_hf_tokenizer


def _make(
    strategy: str, cfg: ChunkingCfg, tokenizer, counter, embedder: Embedder | None
) -> Chunker:
    if strategy == "token":
        return TokenChunker(tokenizer, cfg.max_tokens, cfg.overlap)
    if strategy == "recursive":
        return RecursiveChunker(counter, cfg.max_tokens, cfg.overlap, tokenizer)
    if strategy == "semantic":
        if embedder is None:
            raise ConfigError("Semantic chunking needs an embedder")
        return SemanticChunker(
            embedder,
            counter,
            cfg.semantic.threshold,
            cfg.max_tokens,
            cfg.semantic.min_chunk_tokens,
        )
    raise ConfigError(f"Unknown chunking strategy: {strategy}")


def build_chunker(
    cfg: ChunkingCfg, embed_cfg: EmbeddingCfg, embedder: Embedder | None = None
) -> Chunker:
    tokenizer = load_hf_tokenizer(embed_cfg.tokenizer or embed_cfg.model)
    counter = TokenCounter(tokenizer)
    return _make(cfg.strategy, cfg, tokenizer, counter, embedder)


def build_chunkers(
    cfg: ChunkingCfg, embed_cfg: EmbeddingCfg, embedder: Embedder | None = None
) -> dict[str, Chunker]:
    """Every strategy, sharing one tokenizer.

    The router picks per document, so building through `build_chunker` each time
    would re-resolve the tokenizer once per file — the expensive part of
    construction, and pure waste when the answer is always the same tokenizer.

    A strategy that cannot be built is skipped rather than raising: `semantic`
    needs an embedder, and a deployment without one should still get a router
    that chooses between the other two instead of failing every ingest.
    """
    tokenizer = load_hf_tokenizer(embed_cfg.tokenizer or embed_cfg.model)
    counter = TokenCounter(tokenizer)
    built: dict[str, Chunker] = {}
    for strategy in ("recursive", "token", "semantic"):
        try:
            built[strategy] = _make(strategy, cfg, tokenizer, counter, embedder)
        except ConfigError:
            continue
    return built


__all__ = ["Chunker", "build_chunker", "build_chunkers"]
