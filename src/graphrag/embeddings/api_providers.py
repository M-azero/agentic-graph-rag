"""API-based embedders (OpenAI, Gemini, Voyage, Cohere, DeepInfra), wrapped
behind our `Embedder` interface via their LangChain integrations."""

from __future__ import annotations

from graphrag.config.settings import EmbeddingCfg, Secrets
from graphrag.core.errors import ProviderError
from graphrag.embeddings.base import Embedder

# Sensible defaults so we can report `dim` without a network round-trip.
# A model missing here still works — `dim` falls back to 1024 — but the vector
# store is created with that number, so getting it wrong costs a re-ingest.
# Set `embeddings.dimensions` explicitly for anything not listed.
_KNOWN_DIMS = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "voyage-3-large": 1024,
    "voyage-3": 1024,
    "embed-v4.0": 1536,
    "models/text-embedding-004": 768,
    # DeepInfra serves these under their upstream ids, so the same name means
    # the same weights — and therefore the same vector space — whether it is
    # reached through DeepInfra, Ollama or sentence-transformers.
    "BAAI/bge-m3": 1024,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-base-en-v1.5": 768,
    "Qwen/Qwen3-Embedding-0.6B": 1024,
    "Qwen/Qwen3-Embedding-4B": 2560,
    "Qwen/Qwen3-Embedding-8B": 4096,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
}


class LangChainEmbedder(Embedder):
    """Adapts any LangChain `Embeddings` object to our interface."""

    def __init__(self, backend, cfg: EmbeddingCfg) -> None:
        self._backend = backend
        self.dim = cfg.dimensions or _KNOWN_DIMS.get(cfg.model, 1024)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._backend.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._backend.embed_query(text)


def build_api_embedder(cfg: EmbeddingCfg, secrets: Secrets) -> Embedder:
    provider = cfg.provider
    try:
        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings

            backend = OpenAIEmbeddings(
                model=cfg.model,
                api_key=secrets.openai_api_key,
                dimensions=cfg.dimensions,
            )
        elif provider == "gemini":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            backend = GoogleGenerativeAIEmbeddings(
                model=cfg.model, google_api_key=secrets.google_api_key
            )
        elif provider == "voyage":
            from langchain_voyageai import VoyageAIEmbeddings

            backend = VoyageAIEmbeddings(model=cfg.model, api_key=secrets.voyage_api_key)
        elif provider == "cohere":
            # Native SDK, not the LangChain wrapper: embed-v4.0 needs
            # input_type and output_dimension, which the wrapper doesn't expose.
            from graphrag.embeddings.cohere_native import CohereEmbedder

            return CohereEmbedder(cfg, secrets.cohere_api_key)
        elif provider == "deepinfra":
            # OpenAI-compatible embeddings endpoint, so the OpenAI client works
            # unchanged against DeepInfra's open-weight models.
            from langchain_openai import OpenAIEmbeddings

            from graphrag.llm.factory import DEEPINFRA_BASE_URL

            backend = OpenAIEmbeddings(
                model=cfg.model,
                api_key=secrets.deepinfra_api_key,
                base_url=DEEPINFRA_BASE_URL,
                # OpenAI's client sends `dimensions` only for models that
                # support truncation; DeepInfra's open-weight models return
                # their native size, so ask for nothing and take what comes.
                check_embedding_ctx_length=False,
            )
        else:
            raise ProviderError(f"Unknown embedding provider: {provider}")
    except ImportError as exc:  # pragma: no cover
        raise ProviderError(
            f"Provider '{provider}' needs an extra package. Install with: pip install '.[extras]'"
        ) from exc

    return LangChainEmbedder(backend, cfg)
