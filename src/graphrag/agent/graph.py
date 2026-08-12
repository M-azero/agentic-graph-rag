"""The LangGraph agent. Wires the chat model + tools into a ReAct loop with
optional Redis-backed multi-turn memory, and exposes a blocking `run`, an async
`arun`, and a token-streaming `astream_events`."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from graphrag.agent.presets import answer_instruction, canonical_preset
from graphrag.agent.prompts import SYSTEM_PROMPT
from graphrag.agent.styles import canonical_style
from graphrag.agent.tools import SourceSink, ToolContext, build_tools, collect_sources
from graphrag.core.logging import get_logger
from graphrag.core.messages import content_to_text as _text
from graphrag.core.types import QueryResult, RetrievedChunk
from graphrag.retrieval.plan import RetrievalPlan, use_plan
from graphrag.usage.meter import use_meter

log = get_logger(__name__)

# Compiled agents held per tenant. The reachable space is bounded but no longer
# tiny — eleven presets times four styles times the models a user picks — so
# this is a real LRU rather than a formality. Still cheap to miss: eviction
# costs one recompile of the tools and the graph, nothing that touches a model.
# One preset dominates a given shelf in practice (it is the shelf's default), so
# the working set stays a handful.
_AGENT_CACHE_SIZE = 12


def _postgres_checkpointer(dsn: str, use_async: bool):
    """Durable memory in Postgres — the same database that holds accounts and
    chat history, which lets the deployment run plain Redis instead of
    redis-stack (only the Redis saver needed the RediSearch modules).

    The saver talks to psycopg directly, so it takes a libpq conninfo string,
    not a SQLAlchemy URL (`libpq_dsn` strips the `+driver` marker psycopg
    rejects). Connections are pooled and small: this pool shares Postgres'
    `max_connections` budget with the app's SQLAlchemy engine.
    """
    from graphrag.db.engine import libpq_dsn

    conninfo = libpq_dsn(dsn)
    kwargs = {"autocommit": True, "prepare_threshold": 0}
    if use_async:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool

        # `open=False`: opening a pool binds it to the running loop, and the
        # API's loop is not this one. The lifespan opens it and awaits setup().
        pool = AsyncConnectionPool(
            conninfo=conninfo, min_size=1, max_size=4, open=False,
            kwargs={**kwargs, "row_factory": dict_row},
        )
        saver = AsyncPostgresSaver(pool)
        saver._graphrag_pool = pool  # the lifespan opens/closes it
        log.info("agent_memory", backend="postgres-async")
        return saver

    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(
        conninfo=conninfo, min_size=1, max_size=2,
        kwargs={**kwargs, "row_factory": dict_row},
    )
    saver = PostgresSaver(pool)
    saver.setup()
    log.info("agent_memory", backend="postgres")
    return saver


def _redis_supports_checkpoints(redis_url: str) -> bool:
    """Whether this Redis can actually back the saver.

    A reachable Redis is not a usable one. The saver issues `FT.*` and `JSON.*`
    commands, which exist only with the RediSearch and ReJSON modules
    (redis-stack); plain Redis answers "unknown command" — but only at *write*
    time. So a saver built against it looks healthy, and then every checkpoint
    fails, one noisy traceback per turn, while the answers themselves still go
    out. This deployment deliberately runs plain Redis (durable memory lives in
    Postgres), so that path is reachable whenever Postgres is the one that is
    unavailable.

    Probing here turns a permanent write failure into a clean fallback at
    startup. Failure to probe counts as unsupported: the whole point is to stop
    selecting a backend we cannot confirm.
    """
    try:
        import redis

        client = redis.Redis.from_url(redis_url)
        try:
            names = {str(m.get("name", "")).lower() for m in client.module_list()}
        finally:
            client.close()
    except Exception as exc:
        log.warning("redis_module_probe_failed", error=str(exc))
        return False

    has_search = bool(names & {"search", "searchlight"})
    has_json = bool(names & {"json", "rejson"})
    if not (has_search and has_json):
        log.info("redis_lacks_checkpoint_modules", modules=sorted(names))
    return has_search and has_json


def _redis_checkpointer(redis_url: str, use_async: bool):
    """Durable memory in Redis. Needs the RediSearch modules (redis-stack) —
    plain Redis answers FT._LIST with "unknown command" and every checkpoint
    write fails."""
    if not _redis_supports_checkpoints(redis_url):
        # Raised, not returned: `build_checkpointer`'s loop treats an exception
        # as "try the next backend", which is exactly the wanted behaviour.
        raise RuntimeError(
            "Redis has no RediSearch/ReJSON modules; the checkpointer needs "
            "redis-stack. Use the postgres memory backend instead."
        )
    if use_async:
        from langgraph.checkpoint.redis import AsyncRedisSaver

        # `asetup()` is awaited in the API lifespan, on the loop that serves
        # requests — driving it here with asyncio.run() would bind the async
        # client to a loop that closes on the way out.
        saver = AsyncRedisSaver(redis_url=redis_url)
        log.info("agent_memory", backend="redis-async")
        return saver

    from langgraph.checkpoint.redis import RedisSaver

    saver = RedisSaver(redis_url=redis_url)
    saver.setup()
    log.info("agent_memory", backend="redis")
    return saver


def build_checkpointer(
    redis_url: str | None,
    enabled: bool,
    *,
    use_async: bool = False,
    redis_available: bool = True,
    backend: str = "redis",
    database_url: str | None = None,
):
    """Durable agent memory, falling back to in-process memory when no durable
    backend is usable. CLI (sync) and API (async) share one keyspace, so a
    thread started in one is visible to the other.

    The configured `backend` is tried first, then the other durable option.
    Trying both matters because the two stores have opposite prerequisites: the
    Redis saver needs redis-stack's RediSearch modules (plain Redis fails every
    write), while the Postgres saver needs a database URL. A deployment that
    runs plain Redis with Postgres available should get durable memory rather
    than silently losing conversations to an in-process saver.

    `use_async` picks the saver flavor. The API needs the async saver — its
    /query streams over `astream`, which needs `aget_tuple`; the sync saver
    inherits a base that raises NotImplementedError there. The CLI is the mirror
    image: it calls the sync `invoke`, which the async saver refuses. Neither
    flavor covers both, hence the flag.

    Falling back rather than failing is deliberate: memory is a feature, not a
    prerequisite. `redis_available` must be a real connectivity check, because
    the savers connect lazily — constructing one against a dead Redis
    "succeeds" and then every query fails at checkpoint time.
    """
    if not enabled:
        return None

    candidates: list[str] = [backend] + [b for b in ("postgres", "redis") if b != backend]
    for name in candidates:
        try:
            if name == "postgres" and database_url:
                return _postgres_checkpointer(database_url, use_async)
            if name == "redis" and redis_url and redis_available:
                return _redis_checkpointer(redis_url, use_async)
        except Exception as exc:
            # Postgres on Windows is usually the event loop: psycopg's async
            # mode refuses the ProactorEventLoop asyncio defaults to there.
            # Linux (and every container) uses a selector loop and is fine.
            log.warning(f"{name}_checkpointer_unavailable", error=str(exc))

    from langgraph.checkpoint.memory import MemorySaver

    log.info("agent_memory", backend="in-process")
    return MemorySaver()


class AgentSession:
    """One question in flight. Owns the per-query source collector."""

    def __init__(
        self,
        agent,
        question: str,
        config: dict,
        meter: Any = None,
        *,
        plan: RetrievalPlan | None = None,
        sink: SourceSink | None = None,
    ) -> None:
        self._agent = agent
        self._input = {"messages": [HumanMessage(content=question)]}
        self._config = config
        self._meter = meter
        self._plan = plan
        self._shared_sink = sink
        self._sink: SourceSink | None = None
        self._final: list[RetrievedChunk] = []
        self._final_labels: set[str] = set()

    @contextmanager
    def _collecting(self) -> Iterator[None]:
        """Bind the collector, the retrieval plan and the token meter for one run.

        All three are bound here rather than by the caller because this is the
        code that invokes the agent: LangGraph copies the context at submit time,
        so anything set after this point is invisible inside the tools.

        The meter is bound as well as passed in `config` because the config only
        reaches the graph's own model calls. Retrieval happens *inside* a tool,
        and under a generative reranker that is one model call per candidate —
        billable work the callbacks never saw.
        """
        with (
            collect_sources(self._shared_sink) as sink,
            use_plan(self._plan),
            use_meter(self._meter),
        ):
            self._sink = sink
            try:
                yield
            finally:
                # Keep the sources readable once the ContextVar is gone: the
                # API reads them off the session after the run returns.
                self._final = sink.chunks
                self._final_labels = sink.labels
                self._sink = None

    @property
    def sources(self) -> list[RetrievedChunk]:
        """Live while the query runs — the streaming path reads this between
        tokens — and still correct after it finishes."""
        return self._sink.chunks if self._sink is not None else self._final

    @property
    def source_labels(self) -> set[str]:
        """Non-chunk provenances the tools surfaced, same lifetime as `sources`."""
        return self._sink.labels if self._sink is not None else self._final_labels

    @property
    def usage(self) -> tuple[int, int]:
        """(input, output) tokens for this run — (0, 0) when unmetered."""
        return self._meter.totals if self._meter is not None else (0, 0)

    def _shape(self, result) -> QueryResult:
        messages = result["messages"]
        answer = next(
            (_text(m.content) for m in reversed(messages)
             if isinstance(m, AIMessage) and _text(m.content).strip()),
            "",
        )
        tool_calls = [
            {"tool": tc["name"], "args": tc.get("args", {})}
            for m in messages if isinstance(m, AIMessage)
            for tc in (m.tool_calls or [])
        ]
        tokens_in, tokens_out = self.usage
        return QueryResult(
            answer=answer, sources=self.sources, tool_calls=tool_calls,
            source_labels=sorted(self.source_labels),
            input_tokens=tokens_in, output_tokens=tokens_out,
        )

    def run(self) -> QueryResult:
        """Blocking run — CLI and scripts. Needs a sync-capable checkpointer."""
        with self._collecting():
            return self._shape(self._agent.invoke(self._input, self._config))

    async def arun(self) -> QueryResult:
        """Async run — the API's non-streaming path."""
        with self._collecting():
            return self._shape(await self._agent.ainvoke(self._input, self._config))

    async def astream_events(self) -> AsyncIterator[tuple[str, str]]:
        """Yield ("tool", name) when the model starts a tool call and
        ("token", text) for answer text. Text produced *before* a tool call
        (thinking out loud) is separated from the final answer with a blank
        line, so streamed and non-streamed outputs read the same."""
        emitted_text = False
        boundary_pending = False
        with self._collecting():
            async for msg, _meta in self._agent.astream(
                self._input, self._config, stream_mode="messages"
            ):
                if isinstance(msg, ToolMessage):
                    if emitted_text:
                        boundary_pending = True
                    continue
                if isinstance(msg, AIMessageChunk):
                    for tc in msg.tool_call_chunks or []:
                        if tc.get("name"):
                            yield "tool", tc["name"]
                    text = _text(msg.content)
                    if text:
                        if boundary_pending:
                            yield "token", "\n\n"
                            boundary_pending = False
                        emitted_text = True
                        yield "token", text


class AgentRunner:
    def __init__(
        self,
        model: BaseChatModel,
        vector,
        hybrid,
        graph,
        embedder,
        checkpointer=None,
        *,
        top_k: int = 8,
        graph_hops: int = 2,
        default_style: str = "detailed",
        default_preset: str = "general",
        max_tool_iterations: int = 6,
    ) -> None:
        self._model = model
        self._checkpointer = checkpointer
        self._default_style = default_style
        self._default_preset = default_preset
        # One tool iteration is two graph supersteps (agent -> tools), plus one
        # final answer step. This is what bounds a looping agent.
        self._recursion_limit = 2 * max(1, max_tool_iterations) + 1
        # Built once: every field is tenant-scoped and nothing in it varies per
        # question, so the tools closing over it can be compiled once too.
        self._ctx = ToolContext(
            vector=vector, hybrid=hybrid, graph=graph, embedder=embedder,
            top_k=top_k, graph_hops=graph_hops,
        )
        self._agents: OrderedDict[
            tuple[str, str, int | None, bool], tuple[Any, Any]
        ] = OrderedDict()

    def _agent_for(
        self,
        preset: str | None,
        style: str | None,
        model: BaseChatModel | None,
        remember: bool = True,
    ):
        """One compiled graph per (preset, style, model, remember), reused across
        questions.

        Compiling is not free — it rebuilds eight StructuredTools and the graph
        every time — and now that the source collector rides in a ContextVar
        there is nothing per-query left to bake in. An `AgentRunner` belongs to
        one `Tenant`, so a compiled agent is never shared across users.

        Keyed on `id(model)` because chat models are neither hashable nor
        cheaply nameable. That is sound here for two reasons: overrides come
        from `Container.chat_model`, which caches one instance per (provider,
        model) pair, and the cache entry holds its own reference to the model —
        so the id stays valid and cannot be recycled onto a different object
        while the entry lives.

        `remember=False` compiles a memoryless variant. A second research pass
        over the same thread would otherwise append the first pass's entire
        ReAct transcript to the user's conversation history *and* resend it on
        every later turn — paying twice for a detour the user never asked for.
        """
        # Both clamps land on their enum, so a caller cannot mint unbounded
        # cache keys out of request input — the preset arrives as a raw string
        # from the request exactly as the style does.
        resolved_style = canonical_style(style or self._default_style)
        resolved_preset = canonical_preset(preset or self._default_preset)
        key = (
            resolved_preset.value,
            resolved_style.value,
            id(model) if model is not None else None,
            remember,
        )
        cached = self._agents.get(key)
        if cached is not None:
            self._agents.move_to_end(key)
            return cached[1]

        # The answer instruction lives in the system prompt, not the human turn:
        # everything on the system side is ours, everything on the human side is
        # the user's question and nothing else. A preset is picked from a
        # server-side table by enum value, so this stays true of presets too.
        prompt = SYSTEM_PROMPT.format(
            style=answer_instruction(resolved_preset, resolved_style)
        )
        chat = model or self._model
        tools = build_tools(self._ctx)
        saver = self._checkpointer if remember else None
        try:
            agent = create_react_agent(chat, tools, prompt=prompt, checkpointer=saver)
        except TypeError:  # older langgraph uses state_modifier
            agent = create_react_agent(
                chat, tools, state_modifier=prompt, checkpointer=saver
            )

        # Value keeps `chat` alive alongside the agent — that is what makes the
        # id() key safe, so don't reduce this to storing the agent alone.
        self._agents[key] = (chat, agent)
        while len(self._agents) > _AGENT_CACHE_SIZE:
            self._agents.popitem(last=False)  # evict LRU
        return agent

    def session(
        self,
        question: str,
        style: str | None = None,
        thread_id: str = "default",
        model: BaseChatModel | None = None,
        meter: Any | None = None,
        plan: RetrievalPlan | None = None,
        sink: SourceSink | None = None,
        remember: bool = True,
        preset: str | None = None,
    ) -> AgentSession:
        """One question in flight.

        `preset` picks the job the assistant is doing (see `agent.presets`);
        `plan` sets how much to retrieve for this run; `sink` lets several runs
        accumulate into one citation list (a review loop's second pass must not
        lose the first pass's evidence); `remember=False` keeps a run out of the
        user's conversation history.
        """
        agent = self._agent_for(preset, style, model, remember)
        config: dict[str, Any] = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self._recursion_limit,
        }
        if meter is not None:
            # Scoped to this invocation, which is the point: the checkpointer
            # keeps the whole thread in state, so anything that counted tokens
            # from the final message list would re-charge every earlier turn.
            config["callbacks"] = [meter]
        return AgentSession(agent, question, config, meter, plan=plan, sink=sink)
