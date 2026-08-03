"""The LangGraph agent. Wires the chat model + tools into a ReAct loop with
optional Redis-backed multi-turn memory, and exposes a blocking `run`, an async
`arun`, and a token-streaming `astream_events`."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from graphrag.agent.prompts import SYSTEM_PROMPT
from graphrag.agent.styles import style_instruction
from graphrag.agent.tools import ToolContext, build_tools
from graphrag.core.logging import get_logger
from graphrag.core.messages import content_to_text as _text
from graphrag.core.types import QueryResult, RetrievedChunk

log = get_logger(__name__)


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


def _redis_checkpointer(redis_url: str, use_async: bool):
    """Durable memory in Redis. Needs the RediSearch modules (redis-stack) —
    plain Redis answers FT._LIST with "unknown command" and every checkpoint
    write fails."""
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
    """One question in flight. Holds the per-query source collector."""

    def __init__(
        self, agent, question: str, config: dict, ctx: ToolContext, meter: Any = None
    ) -> None:
        self._agent = agent
        self._input = {"messages": [HumanMessage(content=question)]}
        self._config = config
        self._ctx = ctx
        self._meter = meter

    @property
    def sources(self) -> list[RetrievedChunk]:
        return self._ctx.collected

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
            input_tokens=tokens_in, output_tokens=tokens_out,
        )

    def run(self) -> QueryResult:
        """Blocking run — CLI and scripts. Needs a sync-capable checkpointer."""
        return self._shape(self._agent.invoke(self._input, self._config))

    async def arun(self) -> QueryResult:
        """Async run — the API's non-streaming path."""
        return self._shape(await self._agent.ainvoke(self._input, self._config))

    async def astream_events(self) -> AsyncIterator[tuple[str, str]]:
        """Yield ("tool", name) when the model starts a tool call and
        ("token", text) for answer text. Text produced *before* a tool call
        (thinking out loud) is separated from the final answer with a blank
        line, so streamed and non-streamed outputs read the same."""
        emitted_text = False
        boundary_pending = False
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
        max_tool_iterations: int = 6,
    ) -> None:
        self._model = model
        self._vector = vector
        self._hybrid = hybrid
        self._graph = graph
        self._embedder = embedder
        self._checkpointer = checkpointer
        self._top_k = top_k
        self._graph_hops = graph_hops
        self._default_style = default_style
        # One tool iteration is two graph supersteps (agent -> tools), plus one
        # final answer step. This is what bounds a looping agent.
        self._recursion_limit = 2 * max(1, max_tool_iterations) + 1

    def _make_agent(
        self, ctx: ToolContext, style: str | None, model: BaseChatModel | None = None
    ):
        tools = build_tools(ctx)
        # The style instruction lives in the system prompt, not the human turn:
        # everything on the system side is ours, everything on the human side is
        # the user's question and nothing else. `style_instruction` clamps to
        # the AnswerStyle enum, so no free-form text can ride in on it.
        prompt = SYSTEM_PROMPT.format(style=style_instruction(style or self._default_style))
        chat = model or self._model
        try:
            return create_react_agent(
                chat, tools, prompt=prompt, checkpointer=self._checkpointer
            )
        except TypeError:  # older langgraph uses state_modifier
            return create_react_agent(
                chat, tools, state_modifier=prompt, checkpointer=self._checkpointer
            )

    def session(
        self,
        question: str,
        style: str | None = None,
        thread_id: str = "default",
        model: BaseChatModel | None = None,
        meter: Any | None = None,
    ) -> AgentSession:
        ctx = ToolContext(
            vector=self._vector,
            hybrid=self._hybrid,
            graph=self._graph,
            embedder=self._embedder,
            top_k=self._top_k,
            graph_hops=self._graph_hops,
        )
        agent = self._make_agent(ctx, style, model)
        config: dict[str, Any] = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self._recursion_limit,
        }
        if meter is not None:
            # Scoped to this invocation, which is the point: the checkpointer
            # keeps the whole thread in state, so anything that counted tokens
            # from the final message list would re-charge every earlier turn.
            config["callbacks"] = [meter]
        return AgentSession(agent, question, config, ctx, meter)
