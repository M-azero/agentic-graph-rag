"""The composition root.

`Container` holds the heavy, shared singletons — the embedding model, the
reranker model, the LLM client, the Neo4j driver, Redis. These are built once and
reused by every user.

`Tenant` is a lightweight, per-user view: it binds cheap store/retriever/agent
wrappers to that user's isolated namespace while reusing the container's shared
models. This is the memory optimization — N users cost N sets of small wrappers,
not N copies of the models.
"""

from __future__ import annotations

import re
import time
from collections import OrderedDict
from functools import cached_property

from graphrag.agent import AgentRunner, build_checkpointer
from graphrag.agent.review.graph import ReviewRunner
from graphrag.cache import get_redis
from graphrag.config import Secrets, Settings, load_settings
from graphrag.core.logging import configure_logging, get_logger
from graphrag.embeddings.base import Embedder
from graphrag.ingestion.chunking import build_chunker, build_chunkers
from graphrag.ingestion.extraction import LLMGraphExtractor
from graphrag.llm import build_chat_chain
from graphrag.ocr import build_ocr
from graphrag.retrieval import (
    GraphAugmentedRetriever,
    HybridRetriever,
    VectorRetriever,
    build_reranker,
)
from graphrag.retrieval.plan import RetrievalPlan
from graphrag.storage import build_graph_store, build_vector_store
from graphrag.storage.neo4j_client import driver_from_secrets, safe_ident

log = get_logger(__name__)

_USER_RE = re.compile(r"[^a-z0-9_-]+")

# Separates the tenant from the shelf inside a corpus name. A dot, specifically:
# `_USER_RE` strips it, so neither a sanitized tenant id nor a sanitized shelf
# slug can ever contain one. That makes `{tenant}.{slug}` unambiguous — exactly
# one dot in a shelf corpus, none in a default one — which a dash or underscore
# would not have been, since tenant ids may legitimately hold both.
SHELF_SEPARATOR = "."

# When Redis is down, don't re-attempt a connection on every access — but do
# recover without a restart once it's back.
_REDIS_RETRY_SECONDS = 30.0


def sanitize_user(user_id: str) -> str:
    """Normalize a user id into a safe namespace/database token."""
    clean = _USER_RE.sub("-", (user_id or "").strip().lower()).strip("-")
    return (clean or "default")[:48]


def sanitize_slug(slug: str | None) -> str:
    """Normalize a shelf slug. Empty means the tenant's default shelf.

    Shorter than a user id because it is only ever a suffix: the pair has to fit
    a DuckDB filename and a Neo4j property comfortably.
    """
    clean = _USER_RE.sub("-", (slug or "").strip().lower()).strip("-")
    return clean[:32]


def corpus_for(user: str, slug: str | None = None) -> str:
    """The storage namespace for one shelf of one tenant.

    The default shelf is deliberately the bare tenant id rather than something
    like `{tenant}.default`. Everything ingested before shelves existed lives
    under that name, so this is what makes those documents the contents of the
    default shelf instead of data no query can reach any more.
    """
    clean = sanitize_slug(slug)
    return f"{user}{SHELF_SEPARATOR}{clean}" if clean else user


class Tenant:
    """One user's isolated view of one shelf, built from the container's shared
    resources.

    `user_id` identifies the account; `corpus` identifies the shelf and is the
    real namespace — the two are equal only for the default shelf. Anything that
    must not leak between shelves (stores, retrievers, conversation memory keys)
    keys on `corpus`; anything that is about the person (logging, quotas) keys on
    `user_id`.
    """

    def __init__(self, container: Container, database: str, corpus: str, user_id: str) -> None:
        c = container
        s = c.settings
        self.user_id = user_id
        self.corpus = corpus
        self.database = database

        self.graph_store = build_graph_store(c.driver, database, corpus, s)
        self.vector_store = build_vector_store(c.driver, database, corpus, s)
        self.vector_retriever = VectorRetriever(c.embedder, self.vector_store)
        graph_aug = GraphAugmentedRetriever(self.graph_store, s.retrieval.graph_hops)
        self.hybrid_retriever = HybridRetriever(
            self.vector_retriever, graph_aug, self.graph_store, c.reranker,
            candidate_k=s.retrieval.candidate_k,
        )
        self.agent = AgentRunner(
            c.llm, self.vector_retriever, self.hybrid_retriever, self.graph_store,
            c.embedder,
            checkpointer=c.checkpointer,
            top_k=s.retrieval.top_k, graph_hops=s.retrieval.graph_hops,
            default_style=s.agent.default_style,
            default_preset=s.agent.default_preset,
            max_tool_iterations=s.agent.max_tool_iterations,
        )
        self._embed_dim = c.embedder.dim
        self._container = c

    @cached_property
    def reviewer(self) -> ReviewRunner:
        """The review loop over this tenant's agent.

        Built lazily: a tenant that never asks a reviewed question never
        compiles the graph, and `agent.review.enabled` is off by default.
        """
        s = self._container.settings
        return ReviewRunner(
            self.agent, self._container.llm, self.graph_store,
            max_rounds=s.agent.review.max_rounds,
            window_before=s.agent.review.window_before,
            window_after=s.agent.review.window_after,
            critic_free_tool_calls=s.agent.review.critic_free_tool_calls,
            critic_free_chars=s.agent.review.critic_free_chars,
            base_plan=RetrievalPlan(
                top_k=s.retrieval.top_k,
                candidate_k=s.retrieval.candidate_k,
                graph_hops=s.retrieval.graph_hops,
            ),
        )

    def setup(self) -> None:
        self.graph_store.setup()
        self.vector_store.setup(self._embed_dim)


class Container:
    def __init__(self, settings: Settings | None = None, secrets: Secrets | None = None) -> None:
        if settings is None or secrets is None:
            settings, secrets = load_settings()
        self.settings = settings
        self.secrets = secrets
        configure_logging(settings.app.log_level)
        self._tenants: OrderedDict[str, Tenant] = OrderedDict()
        self._ready_dbs: set[str] = set()
        self._chat_models: dict[tuple[str, str], object] = {}
        self._redis_client = None
        self._redis_checked_at = 0.0
        # The API flips this to True before serving so the checkpointer is built
        # async (its streaming needs the async saver). CLI/scripts stay sync.
        self.async_memory = False

    # -- shared infrastructure ------------------------------------------------
    @property
    def redis(self):
        """Shared Redis client, or None while unreachable. Reconnects lazily
        (at most every _REDIS_RETRY_SECONDS) so a blip at startup doesn't
        disable caching and quotas until the next restart."""
        if self._redis_client is not None:
            return self._redis_client
        now = time.monotonic()
        if now - self._redis_checked_at < _REDIS_RETRY_SECONDS:
            return None
        self._redis_checked_at = now
        try:
            client = get_redis(self.secrets.redis_url)
            client.ping()
            self._redis_client = client
            return client
        except Exception:
            return None

    @cached_property
    def driver(self):
        return driver_from_secrets(self.secrets)

    @cached_property
    def checkpointer(self):
        return self._build_checkpointer(self.settings.agent.memory_backend)

    def _build_checkpointer(self, backend: str, *, allow_redis: bool = True):
        return build_checkpointer(
            self.secrets.redis_url if allow_redis else None,
            self.settings.agent.memory,
            use_async=self.async_memory,
            redis_available=self.redis is not None,
            backend=backend,
            database_url=self.secrets.database_url,
        )

    def retry_checkpointer(self, failed_backend: str):
        """Rebuild agent memory on a different backend after `failed_backend`
        turned out to be unusable.

        The async savers connect lazily, so a broken backend is only discovered
        at setup time in the API lifespan — too late for `build_checkpointer`'s
        own fallback chain. Plain Redis with the Redis saver is the case that
        matters: it constructs happily and then fails every write.
        """
        other = "postgres" if failed_backend == "redis" else "redis"
        saver = self._build_checkpointer(other, allow_redis=(other == "redis"))
        self.__dict__["checkpointer"] = saver  # replace the cached_property
        return saver

    # -- shared models (loaded once, reused by all tenants) -------------------
    @cached_property
    def embedder(self) -> Embedder:
        cfg = self.settings.embeddings
        if cfg.provider == "sentence_transformers":
            from graphrag.embeddings.sentence_transformers import SentenceTransformerEmbedder

            base: Embedder = SentenceTransformerEmbedder(cfg)
        elif cfg.provider == "ollama":
            from graphrag.embeddings.ollama import OllamaEmbedder

            base = OllamaEmbedder(cfg, self.secrets.ollama_base_url)
        else:
            from graphrag.embeddings.api_providers import build_api_embedder

            base = build_api_embedder(cfg, self.secrets)
        if cfg.cache.enabled and self.redis is not None:
            from graphrag.embeddings.cache import CachedEmbedder

            return CachedEmbedder(base, self.redis, cfg.model, cfg.cache.ttl_seconds)
        return base

    def _failover(self) -> dict:
        """Breaker tuning, shared by every chain (chat, OCR, rerank, extraction)."""
        c = self.settings.llm
        return {
            "max_failures": c.failover_max_failures,
            "cooldown_seconds": c.failover_cooldown_seconds,
        }

    @cached_property
    def llm(self):
        c = self.settings.llm
        return build_chat_chain(
            c.provider, c.model, self.secrets,
            fallbacks=c.fallbacks,
            temperature=c.temperature, max_tokens=c.max_tokens, extra=c.extra,
            **self._failover(),
        )

    def chat_model(self, provider: str, model: str):
        """A chat model for an explicit, registry-validated (provider, model)
        pair, cached per pair. The default pair reuses `llm` so extraction and
        chat share one client. Provider `extra` kwargs (e.g. Anthropic
        thinking) apply only to the configured default — they are model-
        specific and must not leak onto overrides.

        A user-selected model gets the same fallback chain as the default: the
        alternative is that picking a model from the UI silently opts out of
        failover, so one dead provider breaks chat for whoever chose it."""
        c = self.settings.llm
        if (provider, model) == (c.provider, c.model):
            return self.llm
        key = (provider, model)
        if key not in self._chat_models:
            self._chat_models[key] = build_chat_chain(
                provider, model, self.secrets,
                fallbacks=c.fallbacks,
                temperature=c.temperature, max_tokens=c.max_tokens,
                **self._failover(),
            )
        return self._chat_models[key]

    @cached_property
    def extractor_llm(self):
        """The chat model extraction (and community summarization) runs on —
        the dedicated `ingestion.llm` when configured, else the main `llm`."""
        cfg = self.settings.ingestion.llm
        if cfg is None:
            return self.llm
        return build_chat_chain(
            cfg.provider, cfg.model, self.secrets,
            fallbacks=cfg.fallbacks,
            temperature=cfg.temperature, max_tokens=cfg.max_tokens, extra=cfg.extra,
            **self._failover(),
        )

    @cached_property
    def ocr(self):
        if not self.settings.ocr.enabled:
            return None
        return build_ocr(self.settings.ocr, self.secrets)

    @cached_property
    def chunker(self):
        return build_chunker(self.settings.chunking, self.settings.embeddings, self.embedder)

    @cached_property
    def chunkers(self) -> dict[str, object]:
        """Every strategy, for the per-document router. One tokenizer load."""
        return build_chunkers(
            self.settings.chunking, self.settings.embeddings, self.embedder
        )

    @cached_property
    def extractor(self) -> LLMGraphExtractor:
        return LLMGraphExtractor(self.extractor_llm)

    @cached_property
    def reranker(self):
        return build_reranker(self.settings.retrieval.rerank, self.secrets)

    @cached_property
    def guardrails(self):
        """The Guardrails safety client, shared across tenants. Always built (so
        the query path can call it unconditionally); its methods short-circuit
        to `allow` when `safety.enabled` is false, and it opens no socket until
        the first real check. The `GRAPHRAG_GUARDRAILS_URL` secret overrides the
        YAML `base_url` so one image can point at a host or a compose service."""
        from graphrag.safety import GuardrailsClient

        cfg = self.settings.safety
        if self.secrets.guardrails_url:
            cfg = cfg.model_copy(update={"base_url": self.secrets.guardrails_url})
        return GuardrailsClient(cfg, self.secrets.guardrails_api_key)

    # -- per-user tenants -----------------------------------------------------
    def _resolve_scope(self, user: str, shelf: str | None = None) -> tuple[str, str]:
        """Return (database, corpus) for a sanitized user id and shelf slug.

        Only the *corpus* varies per shelf; the database stays per user. Under
        `tenancy.per_tenant_database` that keeps a user's shelves inside their
        one Neo4j database — a database per shelf would multiply an
        Enterprise-only resource by however many subjects someone happens to
        keep, for isolation the corpus tag already provides.
        """
        t = self.settings.tenancy
        corpus = corpus_for(user, shelf)
        if t.per_tenant_database:
            return safe_ident(t.database_prefix + user.replace("-", "_")), corpus
        return self.settings.storage.graph.database, corpus

    def _ensure_database(self, database: str) -> None:
        if not self.settings.tenancy.per_tenant_database:
            return
        try:  # Enterprise-only; degrades gracefully on Community.
            with self.driver.session(database="system") as session:
                session.run(f"CREATE DATABASE {database} IF NOT EXISTS").consume()
        except Exception as exc:
            log.warning("per_tenant_database_unavailable", database=database, error=str(exc))

    def tenant(self, user_id: str | None = None, shelf: str | None = None) -> Tenant:
        """One user's view of one shelf, from cache when it's warm.

        Keyed on the corpus rather than the user: a user with three shelves
        holds three entries, which is the point — each carries its own stores,
        retrievers and compiled agent over a separate slice of the graph. They
        remain cheap wrappers around the container's shared models, so the LRU
        bound in `tenancy.max_active_tenants` is still counting wrappers.
        """
        if not self.settings.tenancy.enabled:
            user_id = None  # single-tenant mode: everyone shares default_user
        user = sanitize_user(user_id or self.settings.tenancy.default_user)
        database, corpus = self._resolve_scope(user, shelf)
        if corpus in self._tenants:
            self._tenants.move_to_end(corpus)
            return self._tenants[corpus]

        tenant = Tenant(self, database, corpus, user)

        # Create indexes once per database (constraints/indexes are DB-wide).
        if database not in self._ready_dbs:
            self._ensure_database(database)
            try:
                tenant.setup()
                self._ready_dbs.add(database)
            except Exception as exc:  # don't fail the request if Neo4j is briefly down
                log.warning("tenant_setup_deferred", user=user, error=str(exc))

        self._tenants[corpus] = tenant
        self._tenants.move_to_end(corpus)
        while len(self._tenants) > self.settings.tenancy.max_active_tenants:
            self._tenants.popitem(last=False)  # evict LRU (cheap wrappers only)
        return tenant

    def evict_tenant(self, user: str) -> int:
        """Drop every cached shelf belonging to `user`, returning how many.

        Used after a purge: the tenant objects hold store wrappers pointed at
        data that no longer exists, and for DuckDB an open file handle that has
        to be released before the file can be unlinked.
        """
        prefix = user + SHELF_SEPARATOR
        stale = [k for k in self._tenants if k == user or k.startswith(prefix)]
        for key in stale:
            self._tenants.pop(key, None)
        return len(stale)

    # -- lifecycle ------------------------------------------------------------
    def setup_storage(self) -> None:
        """Prepare the default user's namespace. Safe to call repeatedly."""
        self.tenant(self.settings.tenancy.default_user)
