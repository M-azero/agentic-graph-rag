"""The pipeline inspector's data: what each block of a real run received and
returned.

`GET /trace/pipelines` serves the diagram — nodes and edges built from
`graphrag.trace.steps`, so the picture cannot drift from the instrumentation.
`POST /trace/query` runs one real question with a tracer bound and streams the
steps as they land. `GET /trace/{id}` reads back a finished run, which is how
an ingest — running in a background task or an arq worker, where nothing can be
streamed from — reaches the page.

Two things this endpoint is deliberately strict about:

* **A traced run is a real run.** It goes through `enforce_message_limits` and
  records its tokens exactly as `/query` does. Without that it would be a free
  query: same model calls, same cost, no quota.
* **The answer is emitted only after the output guard has run.** Steps stream;
  the answer does not. That is what keeps block/redact enforceable here, and it
  is why streaming *steps* does not reopen the problem that moved the chat UI
  off SSE in the first place (see the header of `frontend/src/api.ts`).
"""

from __future__ import annotations

import asyncio
import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from graphrag.agent.prompts import CLOSED_DOMAIN_REFUSAL
from graphrag.agent.review.citations import source_key
from graphrag.api.deps import AuthUser, get_container, get_current_user, get_db, get_query_service

# The query path's own helpers, imported rather than reimplemented. A traced run
# has to make the *same* decisions as an untraced one — the same guard, the same
# gate, the same citation review — and a second copy of that logic here would
# drift on the first change to either, which would make the inspector a picture
# of a pipeline that no longer exists.
from graphrag.api.routers.query import (
    _REFUSAL,
    _bill_step,
    _gate,
    _guard_input,
    _guard_output,
    _pick_model,
    _review_citations,
    _safety_info,
)
from graphrag.api.schemas import Source, TraceRunRequest
from graphrag.container import Container
from graphrag.core.logging import get_logger
from graphrag.core.redact import safe_detail
from graphrag.limits import enforce_message_limits
from graphrag.pipelines import QueryService
from graphrag.shelves import shelf_for_request
from graphrag.trace import Tracer, use_tracer
from graphrag.trace import steps as trace_steps
from graphrag.usage import TokenMeter, estimate_tokens, record_answer_tokens

router = APIRouter(tags=["trace"])
log = get_logger(__name__)

# How often the stream checks the tracer for new steps. The run is doing real
# I/O for seconds at a time, so this only decides how promptly a block lights
# up; a tighter tick would spend wakeups to no visible effect.
_TICK_SECONDS = 0.12


def _store(request: Request):
    store = getattr(request.app.state, "trace_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Tracing is not available")
    return store


def _require_enabled(container: Container) -> None:
    """404, not 403: with tracing off the surface should not advertise itself."""
    if not container.settings.trace.enabled:
        raise HTTPException(status_code=404, detail="Not Found")


def _new_tracer(container: Container, pipeline: str, owner: str) -> Tracer:
    cfg = container.settings.trace
    return Tracer(
        uuid.uuid4().hex[:12],
        pipeline,
        owner=owner,
        include_text=cfg.include_text,
        max_steps=cfg.max_steps,
        max_preview_chars=cfg.max_preview_chars,
    )


# -- the diagram ---------------------------------------------------------------


@router.get("/trace/pipelines")
def pipelines(
    container: Container = Depends(get_container),
    user: AuthUser = Depends(get_current_user),
) -> dict:
    """Nodes and edges for every pipeline the inspector can draw.

    Authenticated, though it carries no user data: it is a map of the system's
    internals, naming every stage and the file it lives in. Production turns
    `docs_enabled` off so the API contract is not published to anyone who types
    the URL; serving this openly would hand out the same class of information
    through a different door.
    """
    _require_enabled(container)
    return {"pipelines": trace_steps.all_graphs()}


# -- reading -------------------------------------------------------------------
#
# Declared before `/trace/{trace_id}`: FastAPI matches in declaration order, and
# a path parameter would otherwise swallow these two.


@router.get("/trace/runs")
def recent(
    request: Request,
    container: Container = Depends(get_container),
    user: AuthUser = Depends(get_current_user),
) -> dict:
    _require_enabled(container)
    return {"runs": _store(request).recent(user.tenant_id)}


@router.get("/trace/{trace_id}")
def read(
    trace_id: str,
    request: Request,
    container: Container = Depends(get_container),
    user: AuthUser = Depends(get_current_user),
) -> dict:
    """One finished run.

    Scoped to the caller's tenant: a trace holds their question and excerpts of
    their documents. Someone else's is reported as missing rather than
    forbidden, so the response does not confirm which ids are real.
    """
    _require_enabled(container)
    record = _store(request).get(trace_id, owner=user.tenant_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown trace id")
    record.pop("owner", None)
    return record


# -- running -------------------------------------------------------------------


@router.post("/trace/query")
async def trace_query(
    req: TraceRunRequest,
    request: Request,
    service: QueryService = Depends(get_query_service),
    container: Container = Depends(get_container),
    user: AuthUser = Depends(enforce_message_limits),
    db=Depends(get_db),
):
    """Run one real question and stream its steps.

    Events: `step` (repeatedly, whole steps rather than deltas so a client that
    missed one cannot get out of sync), then `answer`, then `done`. An `error`
    event carries a scrubbed message if the run fails.
    """
    _require_enabled(container)
    store = _store(request)
    shelf = await shelf_for_request(db, user, req.shelf_id)
    preset = req.preset or shelf.preset
    chosen = _pick_model(request, container, req.model)
    model = container.chat_model(chosen.provider, chosen.model) if chosen else None
    model_name = chosen.model if chosen else container.settings.llm.model

    meter = TokenMeter()
    guard = container.guardrails
    reviewed = service.review_enabled
    tracer = _new_tracer(
        container,
        trace_steps.REVIEW if reviewed else trace_steps.QUERY,
        user.tenant_id,
    )

    async def run() -> dict:
        """The same sequence `/query` runs on its non-streaming branch."""
        v_in = await _guard_input(guard, req.question)
        if v_in is not None and v_in.blocked:
            return {
                "answer": v_in.refusal_message or _REFUSAL,
                "sources": [],
                "safety": _safety_info(v_in, "input"),
                "outcome": "blocked",
            }

        if not await _gate(
            service, req.question, container, user.tenant_id, meter, shelf.slug
        ):
            _bill_step(meter, refused="off_topic")
            return {
                "answer": CLOSED_DOMAIN_REFUSAL,
                "sources": [],
                "safety": None,
                "outcome": "off_topic",
            }

        answer_for = service.areview if reviewed else service.aanswer
        result = await answer_for(
            req.question,
            style=req.style,
            # A throwaway thread, so inspecting a run never lands in a real
            # conversation's memory or its transcript.
            thread_id=f"trace-{tracer.trace_id}",
            user_id=user.tenant_id,
            model=model,
            meter=meter,
            shelf=shelf.slug,
            preset=preset,
        )
        answer, sources = result.answer, result.sources
        billable_in = result.input_tokens
        billable = result.output_tokens or estimate_tokens(answer)

        sources, citations = _review_citations(answer, result)

        safety = None
        v_out = await _guard_output(guard, req.question, answer, sources)
        if v_out is not None:
            if v_out.blocked:
                answer, sources, citations = (v_out.refusal_message or _REFUSAL), [], None
            elif v_out.modified and v_out.sanitized_output is not None:
                answer = v_out.sanitized_output
            safety = _safety_info(v_out, "output")

        _bill_step(meter, style=req.style, preset=preset, model=model_name)
        await record_answer_tokens(
            getattr(request.app.state, "usage", None),
            container.redis,
            tenant_id=user.tenant_id,
            account_id=user.user_id,
            tokens=billable,
            input_tokens=billable_in,
            meta={"style": req.style, "preset": preset, "traced": True},
        )
        cited = citations.cited if citations else frozenset()
        return {
            "answer": answer,
            "sources": [
                Source.from_chunk(c, cited=source_key(c.source) in cited).model_dump()
                for c in sources
            ],
            "safety": safety.model_dump() if safety else None,
            "outcome": "ok",
        }

    async def events():
        # Bound here so the task created inside inherits it: asyncio copies the
        # context at task creation, which is the same reason `use_plan` has to be
        # entered by whatever invokes the agent.
        with use_tracer(tracer):
            task = asyncio.create_task(run())
            try:
                while not task.done():
                    for s in tracer.drain():
                        yield {"event": "step", "data": json.dumps(s.to_dict())}
                    await asyncio.sleep(_TICK_SECONDS)
                for s in tracer.drain():
                    yield {"event": "step", "data": json.dumps(s.to_dict())}

                result = await task
            except asyncio.CancelledError:
                # The client went away mid-run. Stop the work rather than
                # letting an abandoned query keep spending.
                task.cancel()
                raise
            except Exception as exc:
                log.exception("trace_run_failed", error=str(exc))
                tracer.finalize(status="error", title=req.question)
                store.save(tracer.to_dict())
                yield {"event": "error", "data": safe_detail(exc)}
                yield {"event": "done", "data": "[DONE]"}
                return

            tracer.finalize(
                status="ok", title=req.question, outcome=result["outcome"],
                model=model_name, shelf=shelf.slug, preset=preset,
            )
            store.save(tracer.to_dict())
            yield {
                "event": "answer",
                "data": json.dumps({**result, "trace_id": tracer.trace_id}),
            }
            yield {"event": "done", "data": "[DONE]"}

    return EventSourceResponse(events())
