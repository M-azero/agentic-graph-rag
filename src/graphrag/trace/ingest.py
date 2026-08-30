"""Run an ingest with a tracer bound, and keep the trace.

Both ingest paths — the in-process background task and the arq worker — call
`IngestPipeline.run` inside `asyncio.to_thread`, so both get this one wrapper
rather than a copy each. The trace is keyed by the **job id**, which is what
lets the inspector page read an ingest it could never have streamed: the work
may be happening in another process entirely.

Never raises on account of tracing. An ingest that succeeded must not be
reported as failed because a trace could not be written.
"""

from __future__ import annotations

from typing import Any

from graphrag.core.logging import get_logger
from graphrag.trace import steps as trace_steps
from graphrag.trace.recorder import ERROR, OK, Tracer, use_tracer
from graphrag.trace.store import TraceStore

log = get_logger(__name__)


def run_traced(
    pipeline: Any,
    path: str,
    user_id: str | None,
    shelf: str | None,
    *,
    job_id: str,
    settings: Any,
    trace_store: TraceStore | None,
    title: str | None = None,
) -> Any:
    """`pipeline.run(path, user_id, shelf)`, traced. Blocking — call it off the
    event loop, exactly as the untraced call already is."""
    cfg = getattr(settings, "trace", None)
    if trace_store is None or cfg is None or not cfg.enabled:
        return pipeline.run(path, user_id=user_id, shelf=shelf)

    tracer = Tracer(
        job_id,
        trace_steps.INGEST,
        owner=user_id,
        include_text=cfg.include_text,
        max_steps=cfg.max_steps,
        max_preview_chars=cfg.max_preview_chars,
    )
    try:
        with use_tracer(tracer):
            stats = pipeline.run(path, user_id=user_id, shelf=shelf)
    except Exception:
        # A failed ingest is exactly when the trace is worth having: it shows
        # which block stopped and what it had at the time.
        tracer.finalize(status=ERROR, title=title or _title(path))
        _persist(tracer, trace_store)
        raise
    tracer.finalize(
        status=OK,
        title=title or _title(path),
        documents=stats.documents,
        chunks=stats.chunks,
        entities=stats.entities,
        relations=stats.relations,
        extraction_failures=stats.extraction_failures,
    )
    _persist(tracer, trace_store)
    return stats


def _title(path: str) -> str:
    """Fallback for a caller with no display name — the server-side path ingest,
    where the on-disk name *is* the name the operator asked for."""
    return str(path).replace("\\", "/").rsplit("/", 1)[-1] or str(path)


def _persist(tracer: Tracer, store: TraceStore) -> None:
    try:
        store.save(tracer.to_dict())
    except Exception as exc:  # a lost trace must never fail a good ingest
        log.warning("ingest_trace_save_failed", job=tracer.trace_id, error=str(exc))


__all__ = ["run_traced"]
