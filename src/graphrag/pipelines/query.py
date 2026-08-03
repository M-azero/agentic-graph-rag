"""Query service: run a user's agent and shape its output for the API. Every
call is scoped to one user's namespace, and conversation memory is keyed per
user so threads never bleed across accounts."""

from __future__ import annotations

from collections.abc import AsyncIterator

from graphrag.agent import AgentSession
from graphrag.container import Container
from graphrag.core.types import QueryResult, RetrievedChunk
from graphrag.observability import query_trace
from graphrag.usage.meter import TokenMeter


class QueryService:
    def __init__(self, container: Container) -> None:
        self._c = container

    @property
    def settings(self):
        return self._c.settings

    def _session(
        self,
        question: str,
        style: str | None,
        thread_id: str,
        user_id: str | None,
        model=None,
        meter=None,
    ) -> AgentSession:
        tenant = self._c.tenant(user_id)
        # Namespace the memory thread with the user so conversations stay private.
        return tenant.agent.session(
            question, style=style, thread_id=f"{tenant.user_id}:{thread_id}",
            model=model, meter=meter,
        )

    def answer(
        self,
        question: str,
        style: str | None = None,
        thread_id: str = "default",
        user_id: str | None = None,
        model=None,
    ) -> QueryResult:
        """Blocking — for the CLI and scripts (sync checkpointer)."""
        return self._session(question, style, thread_id, user_id, model).run()

    async def aanswer(
        self,
        question: str,
        style: str | None = None,
        thread_id: str = "default",
        user_id: str | None = None,
        model=None,
    ) -> QueryResult:
        """Async — the API's non-streaming path (async checkpointer).

        Always metered: the returned `QueryResult` carries what the run cost, so
        no caller has to remember to ask for accounting.
        """
        # The trace is a no-op unless llmlens observability is enabled; when it
        # is, it roots the auto-traced LangChain spans under this user.
        with query_trace("agent_query", user_id=user_id):
            return await self._session(
                question, style, thread_id, user_id, model, meter=TokenMeter()
            ).arun()

    async def stream(
        self,
        question: str,
        style: str | None = None,
        thread_id: str = "default",
        user_id: str | None = None,
        model=None,
        meter=None,
    ) -> AsyncIterator[tuple[str, str, list[RetrievedChunk]]]:
        """Yield (kind, data, sources) — kind is "token" or "tool".

        A generator cannot return totals, so the caller supplies the `meter` and
        reads it once the stream is exhausted.
        """
        with query_trace("agent_query", user_id=user_id):
            session = self._session(question, style, thread_id, user_id, model, meter)
            async for kind, data in session.astream_events():
                yield kind, data, session.sources

    def search(self, query: str, k: int = 8, user_id: str | None = None) -> list[RetrievedChunk]:
        return self._c.tenant(user_id).hybrid_retriever.retrieve(query, k)
