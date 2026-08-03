"""Rerankers order candidate chunks by true relevance to the query.

Three shapes, because no single backend covers every setup:
  cross_encoder  local Hugging Face model, scores all pairs in one batched pass
  ollama/…       any chat model, scoring pairs one at a time (see LLMReranker)
  cohere/voyage  hosted rerank endpoints
"""

from __future__ import annotations

import abc
import re
from concurrent.futures import ThreadPoolExecutor

from graphrag.config.settings import RerankCfg, Secrets
from graphrag.core.errors import ConfigError, ProviderError
from graphrag.core.logging import get_logger
from graphrag.core.messages import content_to_text
from graphrag.core.types import RetrievedChunk
from graphrag.llm.factory import build_chat_chain

log = get_logger(__name__)

_SCORE_RE = re.compile(r"\d+(?:\.\d+)?")
# Chat providers that can be driven as a generative reranker.
_LLM_PROVIDERS = (
    "ollama", "anthropic", "openai", "gemini", "deepseek", "qwen", "deepinfra",
)

# Stamped on every returned chunk. False means the scores are raw retrieval
# similarities rather than calibrated relevance, which `retrieval.min_relevance`
# must not be compared against — see `graphrag.api.routers.query`.
CALIBRATED = "rerank_calibrated"


def _mark(chunks: list[RetrievedChunk], calibrated: bool) -> list[RetrievedChunk]:
    for chunk in chunks:
        chunk.metadata[CALIBRATED] = calibrated
    return chunks


class Reranker(abc.ABC):
    # True when `rerank` returns scores on the 0-1 relevance scale the
    # closed-domain gate is calibrated against.
    calibrated: bool = True

    @abc.abstractmethod
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        ...


class NoOpReranker(Reranker):
    # Passes retrieval scores straight through. Those are cosine/RRF values on
    # their own scale, not relevance.
    calibrated = False

    def rerank(self, query, chunks, top_k):
        return _mark(chunks[:top_k], False)


class CrossEncoderReranker(Reranker):
    def __init__(self, model: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                "rerank.provider 'cross_encoder' runs the model in-process and needs an "
                "optional extra: pip install '.[local-models]'. To rerank without it, "
                "use provider 'ollama' (any chat model), 'cohere'/'voyage', or 'none'."
            ) from exc
        self._model = CrossEncoder(model)

    def rerank(self, query, chunks, top_k):
        if not chunks:
            return []
        scores = self._model.predict([(query, c.text) for c in chunks])
        ranked = sorted(zip(chunks, scores, strict=True), key=lambda cs: cs[1], reverse=True)
        return _mark([
            RetrievedChunk(
                chunk_id=c.chunk_id, text=c.text, source=c.source,
                score=float(s), retriever=c.retriever, metadata=c.metadata,
            )
            for c, s in ranked[:top_k]
        ], True)


class LLMReranker(Reranker):
    """Generative reranking — a chat model scores each (query, document) pair.

    Ollama exposes no cross-encoder endpoint (`/api/rerank` is not a route), so
    this is how a local reranker runs on a model you've already pulled. Scoring
    is pointwise: one call per candidate, issued concurrently to hide latency,
    which makes `retrieval.candidate_k` the knob that drives cost.

    Quality depends entirely on the model returning a bare number, so `prompt`
    is config-exposed — a model that ignores it scores nothing, and those chunks
    fall back to their retrieval order rather than being dropped.
    """

    def __init__(self, cfg: RerankCfg, secrets: Secrets) -> None:
        extra = dict(cfg.extra)
        if cfg.provider == "ollama":
            # A thinking model spends the whole token budget reasoning and
            # returns empty content — no score at all. Reasoning is also pure
            # latency for a scoring prompt (~8x measured), so off by default.
            extra.setdefault("reasoning", False)
        self.cfg = cfg
        self._llm = build_chat_chain(
            cfg.provider, cfg.model, secrets,
            fallbacks=[f for f in cfg.fallbacks if f.provider in _LLM_PROVIDERS],
            temperature=0.0, max_tokens=cfg.max_tokens, extra=extra,
        )

    def _score(self, query: str, text: str) -> float | None:
        try:
            reply = self._llm.invoke(self.cfg.prompt.format(query=query, document=text))
        except Exception as exc:
            log.warning("rerank_call_failed", error=str(exc), model=self.cfg.model)
            return None
        content = content_to_text(reply.content)
        match = _SCORE_RE.search(content)
        if match is None:
            log.warning("rerank_unparseable", model=self.cfg.model, reply=content[:120])
            return None
        return max(0.0, min(float(match.group()) / 10.0, 1.0))

    def rerank(self, query, chunks, top_k):
        if not chunks:
            return []
        with ThreadPoolExecutor(max_workers=max(1, self.cfg.concurrency)) as pool:
            scores = list(pool.map(lambda c: self._score(query, c.text), chunks))

        pairs = list(zip(chunks, scores, strict=True))
        ranked = sorted(
            ((c, s) for c, s in pairs if s is not None), key=lambda cs: cs[1], reverse=True
        )
        # Keep unscored chunks in retrieval order behind the scored ones; a
        # flaky reply shouldn't silently drop a candidate.
        unscored = [(c, c.score) for c, s in pairs if s is None]
        if unscored:
            log.warning("rerank_partial", scored=len(ranked), unscored=len(unscored))
        # Nothing scored at all means every call failed: the "scores" below are
        # raw retrieval values, so the gate must not read them as relevance.
        return _mark([
            RetrievedChunk(
                chunk_id=c.chunk_id, text=c.text, source=c.source,
                score=float(s), retriever=c.retriever, metadata=c.metadata,
            )
            for c, s in (ranked + unscored)[:top_k]
        ], bool(ranked))


class APIReranker(Reranker):
    """Cohere / Voyage rerank endpoints."""

    def __init__(self, provider: str, model: str, secrets: Secrets) -> None:
        self._provider = provider
        self._model = model
        self._secrets = secrets
        self._client = None

    def rerank(self, query, chunks, top_k):
        if not chunks:
            return []
        docs = [c.text for c in chunks]
        try:
            if self._provider == "cohere":
                import cohere

                # ClientV2 — the v1 client predates the rerank-v4.0 models.
                if self._client is None:
                    self._client = cohere.ClientV2(api_key=self._secrets.cohere_api_key)
                resp = self._client.rerank(
                    model=self._model, query=query, documents=docs, top_n=top_k
                )
                order = [(r.index, r.relevance_score) for r in resp.results]
            elif self._provider == "voyage":
                import voyageai

                client = voyageai.Client(api_key=self._secrets.voyage_api_key)
                resp = client.rerank(query, docs, model=self._model, top_k=top_k)
                order = [(r.index, r.relevance_score) for r in resp.results]
            else:
                raise ConfigError(f"Unknown rerank provider: {self._provider}")
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(f"Rerank provider '{self._provider}' package missing") from exc

        return _mark([
            RetrievedChunk(
                chunk_id=chunks[i].chunk_id, text=chunks[i].text, source=chunks[i].source,
                score=float(score), retriever=chunks[i].retriever, metadata=chunks[i].metadata,
            )
            for i, score in order
        ], True)


class FallbackReranker(Reranker):
    """Try each reranker in turn; the first that answers wins.

    Reranking is not optional here the way it looks: `retrieval.min_relevance`
    refuses any question whose top score falls below it, so losing the reranker
    does not merely worsen ordering — it decides whether questions get answered
    at all. Hence a chain, and hence the last resort being a no-op that reports
    itself as uncalibrated rather than one that quietly hands the gate a
    different scale.
    """

    def __init__(self, members: list[Reranker], labels: list[str]) -> None:
        self._members = members
        self._labels = labels

    @property
    def calibrated(self) -> bool:
        return any(m.calibrated for m in self._members)

    def rerank(self, query, chunks, top_k):
        if not chunks:
            return []
        for index, member in enumerate(self._members):
            try:
                return member.rerank(query, chunks, top_k)
            except Exception as exc:
                nxt = self._labels[index + 1] if index + 1 < len(self._labels) else "none"
                log.warning(
                    "rerank_failover", failed=self._labels[index], next=nxt, error=str(exc)
                )
        log.warning("rerank_unavailable", note="retrieval order only; relevance gate bypassed")
        return _mark(chunks[:top_k], False)


def _build_one(provider: str, model: str, cfg: RerankCfg, secrets: Secrets) -> Reranker:
    if provider == "cross_encoder":
        return CrossEncoderReranker(model)
    if provider in ("cohere", "voyage"):
        return APIReranker(provider, model, secrets)
    if provider in _LLM_PROVIDERS:
        return LLMReranker(cfg.model_copy(update={"provider": provider, "model": model}), secrets)
    raise ConfigError(f"Unknown rerank provider: {provider}")


def build_reranker(cfg: RerankCfg, secrets: Secrets) -> Reranker:
    if not cfg.enabled or cfg.provider == "none":
        return NoOpReranker()

    primary = _build_one(cfg.provider, cfg.model, cfg, secrets)
    if not cfg.fallbacks:
        return primary

    members = [primary]
    labels = [f"{cfg.provider}:{cfg.model}"]
    for fb in cfg.fallbacks:
        label = f"{fb.provider}:{fb.model}"
        if (fb.provider, fb.model) == (cfg.provider, cfg.model):
            continue
        try:
            members.append(_build_one(fb.provider, fb.model, cfg, secrets))
        except Exception as exc:
            # Building a reranker can fail for good reasons — cross_encoder
            # without the local-models extra, a provider whose package is
            # absent. Drop the link, keep the deployment.
            log.warning("rerank_fallback_skipped", model=label, reason=str(exc))
            continue
        labels.append(label)

    if len(members) == 1:
        return primary
    log.info("rerank_chain_ready", chain=" -> ".join(labels))
    return FallbackReranker(members, labels)
