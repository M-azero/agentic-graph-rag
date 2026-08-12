"""Ask questions. `/query` runs the agent (streaming by default); `/compare`
is a convenience that phrases a side-by-side comparison. Both are scoped to the
current user and metered against their limits."""

from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sse_starlette.sse import EventSourceResponse

from graphrag.agent.prompts import CLOSED_DOMAIN_REFUSAL
from graphrag.agent.review.citations import (
    CitationReport,
    available_sources,
    order_by_citation,
    source_key,
    verify_citations,
)
from graphrag.api.deps import AuthUser, get_container, get_db, get_query_service
from graphrag.api.schemas import (
    CompareRequest,
    QueryRequest,
    QueryResponse,
    SafetyInfo,
    Source,
    ToolCall,
)
from graphrag.api.streaming import sse_answer, sse_message, sse_refusal
from graphrag.container import Container
from graphrag.core.logging import get_logger
from graphrag.db.engine import session_scope
from graphrag.db.models import Message, Thread
from graphrag.limits import enforce_message_limits
from graphrag.llm.registry import resolve_model
from graphrag.pipelines import QueryService
from graphrag.retrieval.reranker import CALIBRATED
from graphrag.shelves import shelf_by_id, shelf_for_request
from graphrag.usage import TokenMeter, estimate_tokens, record_answer_tokens

router = APIRouter(tags=["query"])
log = get_logger(__name__)

# Bound what the output guard sees: the groundedness check only needs the
# retrieved evidence, and forwarding the whole corpus would be slow and could
# trip the guard's own input-size caps.
_MAX_GUARD_DOCS = 8
_MAX_DOC_CHARS = 4000

_REFUSAL = "I can't help with that request."


def _context_docs(sources) -> list[dict[str, str]]:
    """Retrieved chunks in the guard's `context_docs` shape (enables the
    output-direction groundedness / hallucination check)."""
    return [
        {"id": c.chunk_id, "text": c.text[:_MAX_DOC_CHARS], "source": c.source}
        for c in sources[:_MAX_GUARD_DOCS]
    ]


def _safety_info(verdict, stage: str) -> SafetyInfo | None:
    """Surface a block/flag/redaction to the client; None when the guard allowed."""
    if verdict.blocked:
        action = "block"
    elif verdict.modified:
        action = "redacted"
    elif verdict.flagged:
        action = "flag"
    else:
        return None
    return SafetyInfo(action=action, stage=stage, reasons=list(verdict.reasons))


def _response(
    answer: str,
    sources,
    tool_calls,
    safety: SafetyInfo | None = None,
    cited: frozenset[str] = frozenset(),
) -> QueryResponse:
    return QueryResponse(
        answer=answer,
        sources=[
            Source.from_chunk(c, cited=source_key(c.source) in cited) for c in sources
        ],
        tool_calls=[ToolCall(**tc) for tc in tool_calls],
        safety=safety,
    )


def _review_citations(answer: str, result) -> tuple[list, CitationReport]:
    """Check a finished answer's citations. No model call — a regex and two set
    operations, so this runs on every answer.

    It makes two corrections to what the agent handed back:

    - **A refusal keeps no sources.** The model refuses precisely when nothing
      retrieved covers the question, so returning the chunks it rejected invites
      the reader to believe the answer came from them. Until now a model-emitted
      refusal shipped with a full `sources` array.
    - **Cited sources sort first** and are marked, so a client can separate the
      evidence the answer used from what retrieval merely surfaced.
    """
    report = verify_citations(
        answer, available_sources(result.sources, result.source_labels)
    )
    if report.refusal:
        return [], report
    if report.fabricated:
        # Advisory for now: the answer still ships. Repairing it is the review
        # loop's job, and blocking on a regex would turn a working answer into
        # a refusal the moment a source name confuses the parser.
        log.warning(
            "citations_fabricated",
            cited=sorted(report.fabricated),
            available=len(report.available),
        )
    return order_by_citation(result.sources, report), report


def _gate_applies(probe) -> bool:
    """May `retrieval.min_relevance` be compared against these scores?

    The threshold is calibrated on the reranker's relevance scale. When every
    reranker in the chain is down the scores are raw retrieval similarities on
    a different scale entirely, and comparing them would refuse whole
    conversations — or admit everything — for a reason no log would explain.
    So a degraded reranker suspends the gate: answering with citations from the
    best available chunks is the better failure, and the retrieval itself is
    unaffected.
    """
    if probe[0].metadata.get(CALIBRATED, True):
        return True
    log.warning("relevance_gate_bypassed", reason="reranker degraded")
    return False


async def _probe(
    service: QueryService, question: str, tenant: str, meter: TokenMeter,
    shelf: str | None = None,
):
    """The closed-domain gate's retrieval, off the event loop.

    `service.search` is fully synchronous — an embedding round trip, three Neo4j
    queries and a rerank — and the API runs a single uvicorn worker (the DuckDB
    provider requires it). Calling it inline froze the whole process for the
    duration, including token flushes for other users' in-flight streams.

    `asyncio.to_thread` copies the caller's context, so the meter bound around
    this call is still visible inside the retriever's own thread pools.
    """
    return await asyncio.to_thread(
        service.search, question, user_id=tenant, meter=meter, shelf=shelf
    )


def _pick_model(request: Request, container: Container, requested: str | None):
    """Resolve a requested model id against both allowlists, or None for the
    default.

    Two lists, and the second was previously read by nobody: `llm.allowed` in
    the profile, then whatever an admin narrowed it to in the console. Storing
    that narrowing without enforcing it meant "disable a model" removed it from
    the picker and left it fully callable over the API.
    """
    if not requested:
        return None
    return resolve_model(
        requested, container.settings, getattr(request.app.state, "enabled_models", None)
    )


def _chat_model(request: Request, container: Container, requested: str | None):
    """A provider client for a validated (provider, model), or None for the
    default. A raw request string never reaches a provider client."""
    chosen = _pick_model(request, container, requested)
    return container.chat_model(chosen.provider, chosen.model) if chosen else None


async def _book(
    request: Request, container: Container, user: AuthUser, meter: TokenMeter, meta: dict
) -> None:
    """Charge whatever this request has spent so far to the caller."""
    tokens_in, tokens_out = meter.totals
    await record_answer_tokens(
        getattr(request.app.state, "usage", None), container.redis,
        tenant_id=user.tenant_id, account_id=user.user_id,
        tokens=tokens_out, input_tokens=tokens_in, meta=meta,
    )


async def _owned_thread(
    db, user: AuthUser, thread_id: str
) -> tuple[str | None, uuid.UUID | None]:
    """Confirm the caller owns this conversation; return its id and its shelf.

    Returns `(None, None)` when there is nothing to check against (no database,
    or a dev-mode identity), which leaves the old free-form thread ids working.
    """
    if db is None or not thread_id or thread_id == "default":
        return None, None
    try:
        owner = uuid.UUID(str(user.user_id))
        tid = uuid.UUID(thread_id)
    except (ValueError, AttributeError, TypeError):
        return None, None
    async with session_scope(db) as s:
        found = (
            await s.execute(
                select(Thread.id, Thread.shelf_id).where(
                    Thread.id == tid, Thread.user_id == owner, Thread.deleted_at.is_(None)
                )
            )
        ).one_or_none()
    if found is None:
        raise HTTPException(status_code=404, detail="No such conversation.")
    return str(found[0]), found[1]


async def _shelf_for(
    db, user: AuthUser, req, thread_id: str | None, thread_shelf: uuid.UUID | None
):
    """Which shelf this question searches.

    An existing conversation decides, and the request's `shelf_id` is only read
    when there is no conversation yet. Honouring a stale `shelf_id` — the picker
    moved after the thread started — would search a corpus this thread's history
    was never grounded in, while the agent's memory (keyed on the *thread's*
    corpus) stayed where it was: the answer would cite documents the transcript
    above it has never mentioned.

    The branch is on `thread_id`, not on `thread_shelf`. A thread pinned to the
    default shelf has `shelf_id IS NULL` — as does every thread predating
    shelves — so branching on the shelf would hand exactly those conversations
    back to whatever the request asked for, which is the case this exists to
    prevent. NULL means "the default shelf", not "no opinion".
    """
    if thread_id is not None:
        return await shelf_by_id(db, _owner_uuid(user), thread_shelf)
    return await shelf_for_request(db, user, req.shelf_id)


def _owner_uuid(user: AuthUser) -> uuid.UUID | None:
    try:
        return uuid.UUID(str(user.user_id))
    except (ValueError, AttributeError, TypeError):
        return None


async def _save_turn(
    db, thread_id: str | None, question: str, answer: str, sources, model: str
) -> None:
    """Persist one exchange. Never raises: the answer already reached the user,
    and losing a transcript row must not turn a served request into an error."""
    if db is None or not thread_id:
        return
    from graphrag.core.logging import get_logger

    try:
        async with session_scope(db) as s:
            s.add(Message(thread_id=uuid.UUID(thread_id), role="user", content=question))
            s.add(
                Message(
                    thread_id=uuid.UUID(thread_id),
                    role="assistant",
                    content=answer,
                    sources=[Source.from_chunk(c).model_dump() for c in sources],
                    model=model or None,
                )
            )
            thread = (
                await s.execute(select(Thread).where(Thread.id == uuid.UUID(thread_id)))
            ).scalar_one_or_none()
            if thread is not None and thread.title == "New chat":
                # Name the conversation after its opening question, so the
                # sidebar is scannable without the user renaming anything.
                thread.title = question.strip()[:60] or "New chat"
    except Exception as exc:
        get_logger(__name__).warning("transcript_save_failed", error=str(exc))


@router.post("/query", response_model=QueryResponse | None)
async def query(
    req: QueryRequest,
    request: Request,
    service: QueryService = Depends(get_query_service),
    container: Container = Depends(get_container),
    user: AuthUser = Depends(enforce_message_limits),
    db=Depends(get_db),
):
    thread_id, thread_shelf = await _owned_thread(db, user, req.thread_id)
    shelf = await _shelf_for(db, user, req, thread_id, thread_shelf)
    # The shelf's own preset is the fallback, not an override: the composer
    # seeds itself from it, so a request that names one has had the user change
    # it deliberately for this question.
    preset = req.preset or shelf.preset
    stream = req.stream if req.stream is not None else container.settings.api.stream
    chosen = _pick_model(request, container, req.model)
    model = container.chat_model(chosen.provider, chosen.model) if chosen else None
    model_name = chosen.model if chosen else container.settings.llm.model

    # One meter for the whole request. The probe below reranks before the agent
    # runs, and under a generative reranker that is real spend — a meter created
    # later inside `aanswer` would never see it.
    meter = TokenMeter()
    guard = container.guardrails

    # Guardrails input check — before the model runs. A block short-circuits the
    # whole request: no agent, no retrieval, no tokens spent. No-op when
    # safety.enabled is false (guard.check_input returns an `allow`).
    if guard.enabled:
        v_in = await guard.check_input(req.question)
        if v_in.blocked:
            refusal = v_in.refusal_message or _REFUSAL
            await _save_turn(db, thread_id, req.question, refusal, [], model_name)
            if stream:
                return EventSourceResponse(sse_refusal(refusal))
            return _response(refusal, [], [], _safety_info(v_in, "input"))

    # Closed-domain gate: only answer when the knowledge base actually covers the
    # question. One probe retrieval; if nothing clears retrieval.min_relevance, we
    # refuse here — an off-topic question gets an honest "not in the KB" instead of
    # a general-knowledge answer. min_relevance = 0 disables the gate.
    min_rel = container.settings.retrieval.min_relevance
    if min_rel > 0:
        probe = await _probe(service, req.question, user.tenant_id, meter, shelf.slug)
        if not probe or (_gate_applies(probe) and probe[0].score < min_rel):
            # The probe still cost tokens under a generative reranker, and a
            # refused question is exactly when a caller would otherwise retry
            # for free.
            await _book(request, container, user, meter, {"style": req.style,
                                                          "preset": preset,
                                                          "refused": "off_topic"})
            await _save_turn(db, thread_id, req.question, CLOSED_DOMAIN_REFUSAL, [], model_name)
            if stream:
                return EventSourceResponse(sse_message(CLOSED_DOMAIN_REFUSAL))
            return _response(CLOSED_DOMAIN_REFUSAL, [], [])

    if stream:
        async def _out_guard(answer, sources):
            return await guard.check_output(
                req.question, answer, docs=_context_docs(sources)
            )

        return EventSourceResponse(
            sse_answer(
                service, req.question, req.style, req.thread_id, user.tenant_id,
                redis_client=container.redis, model=model,
                shelf=shelf.slug, preset=preset,
                recorder=getattr(request.app.state, "usage", None),
                account_id=user.user_id,
                # The same meter the probe already charged to, so the stream
                # bills one total rather than forgetting the retrieval half.
                meter=meter,
                output_guard=_out_guard if guard.enabled else None,
                on_complete=lambda answer, sources: _save_turn(
                    db, thread_id, req.question, answer, sources, model_name
                ),
            )
        )
    # Async, not sync-in-threadpool: the API's checkpointer is the async saver,
    # which refuses sync `.invoke()`.
    #
    # The reviewed path is only reachable here, on the non-streaming branch:
    # once tokens are on the wire an answer cannot be withdrawn or repaired,
    # which is the same reason the output guard degrades to advisory when
    # streaming (see `api/streaming.py`).
    answer_for = service.areview if service.review_enabled else service.aanswer
    result = await answer_for(
        req.question, style=req.style, thread_id=req.thread_id,
        user_id=user.tenant_id, model=model, meter=meter,
        shelf=shelf.slug, preset=preset,
    )
    answer, sources, tool_calls = result.answer, result.sources, result.tool_calls
    # Read off the run, not the final text: the agent's tool loop resends the
    # whole prompt every turn, so the visible answer is a small fraction of what
    # the question actually cost. Captured before the guard can swap the text —
    # the model has already run, and billing the refusal that replaces a blocked
    # answer would make the most expensive requests the cheapest ones.
    billable_in = result.input_tokens
    billable = result.output_tokens or estimate_tokens(answer)

    # Citation review (deterministic, no tokens): drops the sources behind a
    # refusal and sorts the ones the answer cited to the front.
    sources, citations = _review_citations(answer, result)

    # Guardrails output check — the non-streaming path can enforce fully: block
    # (withhold the answer) or redact (swap in the sanitized, PII-clean text).
    # The verdict rides back on the response so the UI can show why.
    safety = None
    if guard.enabled:
        v_out = await guard.check_output(
            req.question, answer, docs=_context_docs(sources)
        )
        if v_out.blocked:
            answer, sources, tool_calls = (v_out.refusal_message or _REFUSAL), [], []
            citations = None
        elif v_out.modified and v_out.sanitized_output is not None:
            answer = v_out.sanitized_output
        safety = _safety_info(v_out, "output")

    await record_answer_tokens(
        getattr(request.app.state, "usage", None), container.redis,
        tenant_id=user.tenant_id, account_id=user.user_id,
        tokens=billable, input_tokens=billable_in,
        meta={"style": req.style, "preset": preset, "stream": False},
    )
    await _save_turn(db, thread_id, req.question, answer, sources, model_name)
    return _response(
        answer, sources, tool_calls, safety,
        cited=citations.cited if citations else frozenset(),
    )


@router.post("/compare", response_model=QueryResponse)
async def compare(
    req: CompareRequest,
    request: Request,
    service: QueryService = Depends(get_query_service),
    container: Container = Depends(get_container),
    user: AuthUser = Depends(enforce_message_limits),
    db=Depends(get_db),
):
    subjects = ", ".join(req.subjects)
    aspects = ("along these aspects: " + ", ".join(req.aspects)) if req.aspects else ""
    question = f"Compare {subjects} {aspects}. Present the comparison as a table."

    thread_id, thread_shelf = await _owned_thread(db, user, req.thread_id)
    shelf = await _shelf_for(db, user, req, thread_id, thread_shelf)
    preset = req.preset or shelf.preset
    meter = TokenMeter()
    guard = container.guardrails
    if guard.enabled:
        # Screen the composed question: subjects AND aspects are user-supplied,
        # and either one can carry an injection.
        v_in = await guard.check_input(question)
        if v_in.blocked:
            return _response(
                v_in.refusal_message or _REFUSAL, [], [], _safety_info(v_in, "input")
            )

    min_rel = container.settings.retrieval.min_relevance
    if min_rel > 0:
        probe = await _probe(service, question, user.tenant_id, meter, shelf.slug)
        if not probe or (_gate_applies(probe) and probe[0].score < min_rel):
            await _book(
                request, container, user, meter,
                {"style": req.style, "preset": preset,
                 "endpoint": "compare", "refused": "off_topic"},
            )
            return _response(CLOSED_DOMAIN_REFUSAL, [], [])

    result = await service.aanswer(
        question, style=req.style, thread_id=req.thread_id, user_id=user.tenant_id,
        model=_chat_model(request, container, req.model), meter=meter,
        shelf=shelf.slug, preset=preset,
    )
    answer, sources, tool_calls = result.answer, result.sources, result.tool_calls
    # Before the guard can replace the text; see the note in `query` above.
    billable_in = result.input_tokens
    billable = result.output_tokens or estimate_tokens(answer)

    sources, citations = _review_citations(answer, result)

    safety = None
    if guard.enabled:
        v_out = await guard.check_output(question, answer, docs=_context_docs(sources))
        if v_out.blocked:
            answer, sources, tool_calls = (v_out.refusal_message or _REFUSAL), [], []
            citations = None
        elif v_out.modified and v_out.sanitized_output is not None:
            answer = v_out.sanitized_output
        safety = _safety_info(v_out, "output")

    # /compare never streams, so it was the other half of the unmetered path —
    # and it composes a table over several subjects, making it the *more*
    # expensive of the two per call.
    await record_answer_tokens(
        getattr(request.app.state, "usage", None), container.redis,
        tenant_id=user.tenant_id, account_id=user.user_id,
        tokens=billable, input_tokens=billable_in,
        meta={"style": req.style, "preset": preset,
              "stream": False, "endpoint": "compare"},
    )
    return _response(
        answer, sources, tool_calls, safety,
        cited=citations.cited if citations else frozenset(),
    )
