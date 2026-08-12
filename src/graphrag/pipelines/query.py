"""Query service: run a user's agent and shape its output for the API. Every
call is scoped to one user's namespace, and conversation memory is keyed per
user so threads never bleed across accounts."""

from __future__ import annotations

from collections.abc import AsyncIterator

from graphrag.agent import AgentSession
from graphrag.container import Container
from graphrag.core.types import QueryResult, RetrievedChunk
from graphrag.observability import query_trace
from graphrag.usage.meter import TokenMeter, use_meter


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
        shelf: str | None = None,
        preset: str | None = None,
    ) -> AgentSession:
        tenant = self._c.tenant(user_id, shelf)
        # Namespace the memory thread with the corpus, so conversations stay
        # private *and* stay on their shelf. For the default shelf the corpus is
        # the tenant id, so threads that predate shelves keep their existing key
        # and their history with it.
        return tenant.agent.session(
            question, style=style, preset=preset,
            thread_id=f"{tenant.corpus}:{thread_id}",
            model=model, meter=meter,
        )

    def answer(
        self,
        question: str,
        style: str | None = None,
        thread_id: str = "default",
        user_id: str | None = None,
        model=None,
        shelf: str | None = None,
        preset: str | None = None,
    ) -> QueryResult:
        """Blocking — for the CLI and scripts (sync checkpointer)."""
        return self._session(
            question, style, thread_id, user_id, model, shelf=shelf, preset=preset
        ).run()

    async def aanswer(
        self,
        question: str,
        style: str | None = None,
        thread_id: str = "default",
        user_id: str | None = None,
        model=None,
        meter: TokenMeter | None = None,
        shelf: str | None = None,
        preset: str | None = None,
    ) -> QueryResult:
        """Async — the API's non-streaming path (async checkpointer).

        Always metered: the returned `QueryResult` carries what the run cost, so
        no caller has to remember to ask for accounting.

        A caller that already spent tokens on this request — the closed-domain
        probe reranks before the agent runs — passes its own meter, so the whole
        request lands on one bill instead of two, one of which nobody reads.
        """
        # The trace is a no-op unless llmlens observability is enabled; when it
        # is, it roots the auto-traced LangChain spans under this user.
        with query_trace("agent_query", user_id=user_id):
            return await self._session(
                question, style, thread_id, user_id, model,
                meter=meter or TokenMeter(), shelf=shelf, preset=preset,
            ).arun()

    async def stream(
        self,
        question: str,
        style: str | None = None,
        thread_id: str = "default",
        user_id: str | None = None,
        model=None,
        meter=None,
        shelf: str | None = None,
        preset: str | None = None,
    ) -> AsyncIterator[tuple[str, str, list[RetrievedChunk], list[str]]]:
        """Yield (kind, data, sources, source_labels) — kind is "token" or "tool".

        A generator cannot return totals, so the caller supplies the `meter` and
        reads it once the stream is exhausted.

        The labels ride along for the same reason they do on `QueryResult`:
        without them, an answer grounded in the knowledge graph reads as citing
        a source that was never retrieved.
        """
        with query_trace("agent_query", user_id=user_id):
            session = self._session(
                question, style, thread_id, user_id, model, meter,
                shelf=shelf, preset=preset,
            )
            async for kind, data in session.astream_events():
                yield kind, data, session.sources, sorted(session.source_labels)

    @property
    def review_enabled(self) -> bool:
        return self._c.settings.agent.review.enabled

    async def areview(
        self,
        question: str,
        style: str | None = None,
        thread_id: str = "default",
        user_id: str | None = None,
        model=None,
        meter: TokenMeter | None = None,
        shelf: str | None = None,
        preset: str | None = None,
    ) -> QueryResult:
        """Answer with review: research, check the citations, repair or
        re-retrieve, then return.

        Returns a `QueryResult` like `aanswer`, with `output_tokens` already
        covering the critic and reviser — they share this request's meter. The
        caller bills the returned totals and needs to know nothing about how
        many model calls went into producing them.
        """
        tenant = self._c.tenant(user_id, shelf)
        meter = meter or TokenMeter()
        with query_trace("agent_review", user_id=user_id):
            outcome = await tenant.reviewer.arun(
                question,
                style=style,
                preset=preset,
                thread_id=f"{tenant.corpus}:{thread_id}",
                user_id=tenant.user_id,
                model=model,
                meter=meter,
            )
        tokens_in, tokens_out = meter.totals
        return QueryResult(
            answer=outcome.answer,
            sources=outcome.sources,
            tool_calls=outcome.tool_calls,
            source_labels=outcome.labels,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
        )

    def search(
        self,
        query: str,
        k: int = 8,
        user_id: str | None = None,
        meter: TokenMeter | None = None,
        shelf: str | None = None,
    ) -> list[RetrievedChunk]:
        """Raw retrieval. Blocking — callers on the event loop must hand it to a
        thread (`asyncio.to_thread`), which carries the bound context with it.

        `meter` is honoured because retrieval is not free under every
        configuration: a generative reranker spends one model call per candidate,
        so an unmetered search is an unmetered bill.
        """
        with use_meter(meter):
            return self._c.tenant(user_id, shelf).hybrid_retriever.retrieve(query, k)
