"""Per-run step recorder: what the pipeline inspector draws.

Not to be confused with `graphrag.observability`, which is the llmlens
integration and ships spans to an external server. This one records the steps of
*one* run, in-process, so a UI can show a reader what each block received and
what it returned.

It is the fourth member of a family that already exists here — `SourceSink`
(`agent/tools.py`), `RetrievalPlan` (`retrieval/plan.py`) and `TokenMeter`
(`usage/meter.py`) — and it is bound the same way, for the same reason: the work
being traced happens inside a compiled graph that outlives the query, several
thread pools deep. Every fan-out point already copies the context
(`HybridRetriever.retrieve`, `LLMReranker.rerank`, `asyncio.to_thread`), so a
tracer bound where the run starts is visible everywhere the run reaches.

The property that makes this safe to leave in the hot path: **with no tracer
bound, `step()` costs one ContextVar read and returns a handle whose methods do
nothing.** The untraced path — which is every normal query — is unchanged.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any

from graphrag.core.logging import get_logger
from graphrag.core.redact import redact_secrets, safe_detail
from graphrag.trace import steps as registry

log = get_logger(__name__)

RUNNING = "running"
OK = "ok"
ERROR = "error"
SKIPPED = "skipped"

# Defaults, overridden per-tracer from `settings.trace`. They exist so a tracer
# built without config (tests, scripts) is still bounded.
DEFAULT_MAX_STEPS = 200
DEFAULT_MAX_PREVIEW_CHARS = 1200


@dataclass
class TraceStep:
    """One block of the pipeline, as it actually ran."""

    id: str
    name: str
    label: str
    kind: str
    parent_id: str | None = None
    status: str = RUNNING
    started_ms: float = 0.0
    ended_ms: float | None = None
    # Capped, redacted previews. `input` is what went in, `output` what came
    # out, `meta` the counts and identifiers worth showing beside them.
    input: dict = field(default_factory=dict)
    output: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def duration_ms(self) -> float | None:
        if self.ended_ms is None:
            return None
        return round(self.ended_ms - self.started_ms, 2)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["duration_ms"] = self.duration_ms
        return data


class _NullStep:
    """What `step()` yields when nothing is being traced. Every method is a
    no-op, so instrumentation reads identically on both paths and callers never
    branch on whether a tracer exists."""

    __slots__ = ()

    id: str | None = None

    def output(self, **_kw: Any) -> None:
        return None

    def meta(self, **_kw: Any) -> None:
        return None

    def note(self, *_a: Any, **_kw: Any) -> None:
        return None

    def skip(self, *_a: Any, **_kw: Any) -> None:
        return None


NULL_STEP = _NullStep()


class _LiveStep:
    """A handle on a step that is currently running. Mutating methods take the
    tracer's lock, because the three retrieval legs and the rerank pool all
    write while the agent thread reads."""

    __slots__ = ("_step", "_tracer")

    def __init__(self, tracer: Tracer, step: TraceStep) -> None:
        self._tracer = tracer
        self._step = step

    @property
    def id(self) -> str:
        return self._step.id

    def output(self, **kw: Any) -> None:
        self._tracer._merge(self._step, "output", kw)

    def meta(self, **kw: Any) -> None:
        self._tracer._merge(self._step, "meta", kw)

    def note(self, **kw: Any) -> None:
        """Alias for `meta`, for call sites where 'note' reads better."""
        self._tracer._merge(self._step, "meta", kw)

    def skip(self, reason: str = "") -> None:
        """Mark a stage that was configured off or short-circuited. Distinct
        from an error, and distinct from never having been reached."""
        with self._tracer._lock:
            self._step.status = SKIPPED
            if reason:
                self._step.meta["reason"] = reason


class Tracer:
    """The steps of one run, in the order they completed.

    Thread-safe by construction. `SourceSink` next door carries the same lock
    for the same reason: LangGraph runs sync tools on executor threads and the
    hybrid retriever fans out to three, so appends genuinely race.
    """

    def __init__(
        self,
        trace_id: str,
        pipeline: str = registry.QUERY,
        *,
        owner: str | None = None,
        include_text: bool = True,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_preview_chars: int = DEFAULT_MAX_PREVIEW_CHARS,
    ) -> None:
        self.trace_id = trace_id
        self.pipeline = pipeline
        self.owner = owner
        self.include_text = include_text
        self.max_steps = max(1, max_steps)
        self.max_preview_chars = max(80, max_preview_chars)
        self.started_at = time.time()
        self.status = RUNNING
        self.title = ""
        self._steps: list[TraceStep] = []
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._truncated = False
        # Where a consumer has read up to. The SSE endpoint drains from here
        # rather than re-sending the whole list on every tick.
        self._drained = 0

    # -- construction ---------------------------------------------------------

    def _spec(self, name: str) -> registry.StepSpec | None:
        return registry.SPECS.get(name)

    def begin(self, name: str, label: str | None, kind: str | None, payload: dict) -> Any:
        """Open a step. Returns a `_LiveStep`, or `NULL_STEP` once the cap is hit."""
        spec = self._spec(name)
        if spec is None:
            # An unregistered name is a bug in the instrumentation, not a reason
            # to fail a query. Log once-ish and record it under its raw name so
            # it still shows up in the timeline.
            log.warning("trace_unknown_step", step=name)
        with self._lock:
            if len(self._steps) >= self.max_steps:
                if not self._truncated:
                    self._truncated = True
                    self._steps.append(
                        TraceStep(
                            id="truncated",
                            name=registry.TRUNCATED,
                            label="Truncated",
                            kind="meta",
                            status=OK,
                            started_ms=self._now(),
                            ended_ms=self._now(),
                            meta={"max_steps": self.max_steps},
                        )
                    )
                return NULL_STEP
            step = TraceStep(
                id=f"s{next(self._ids)}",
                name=name,
                label=label or (spec.label if spec else name),
                kind=kind or (spec.kind if spec else "other"),
                parent_id=_CURRENT.get(),
                started_ms=self._now(),
                input=self._clean(payload),
            )
            self._steps.append(step)
        return _LiveStep(self, step)

    def finish(self, live: Any, error: BaseException | None = None) -> None:
        if not isinstance(live, _LiveStep):
            return
        with self._lock:
            step = live._step
            step.ended_ms = self._now()
            if error is not None:
                step.status = ERROR
                step.error = safe_detail(error)
            elif step.status == RUNNING:
                step.status = OK

    def _merge(self, step: TraceStep, slot: str, values: dict) -> None:
        cleaned = self._clean(values)
        with self._lock:
            getattr(step, slot).update(cleaned)

    def _now(self) -> float:
        return round((time.time() - self.started_at) * 1000, 2)

    # -- payload hygiene ------------------------------------------------------

    def _clean(self, payload: dict | None) -> dict:
        """Cap, redact, and flatten one payload.

        Everything here reaches a browser. Strings are truncated to
        `max_preview_chars` and run through `redact_secrets` — the same last
        line of defence the error path uses, because a step's input can be a
        prompt and a prompt can quote a config value. With `include_text` off,
        long strings are replaced by their length so the shape of the run is
        still visible without the documents themselves.
        """
        if not payload:
            return {}
        return {str(k): self._clean_value(v) for k, v in payload.items()}

    def _clean_value(self, value: Any, depth: int = 0) -> Any:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return self._clean_text(value)
        if depth >= 3:
            return f"<{type(value).__name__}>"
        if isinstance(value, dict):
            return {
                str(k): self._clean_value(v, depth + 1)
                for k, v in list(value.items())[:24]
            }
        if isinstance(value, (list, tuple, set)):
            items = list(value)[:24]
            return [self._clean_value(v, depth + 1) for v in items]
        return self._clean_text(str(value))

    def _clean_text(self, text: str) -> str:
        if not self.include_text and len(text) > 120:
            return f"<{len(text)} chars withheld: trace.include_text is off>"
        out = redact_secrets(text)
        if len(out) > self.max_preview_chars:
            return out[: self.max_preview_chars].rstrip() + " …[truncated]"
        return out

    # -- reading --------------------------------------------------------------

    @property
    def steps(self) -> list[TraceStep]:
        """A snapshot — safe to read while the run is still going."""
        with self._lock:
            return list(self._steps)

    def drain(self) -> list[TraceStep]:
        """Steps added or completed since the last drain, for the live stream.

        Returns whole steps rather than deltas: a step is small, and a consumer
        that re-renders from the latest copy cannot get out of sync with one
        that missed an update. A step still running is re-sent on the next tick
        so the UI can show it finishing.
        """
        with self._lock:
            fresh = self._steps[self._drained :]
            # Only advance past steps that are finished; a running one stays in
            # the window until it lands.
            settled = 0
            for step in fresh:
                if step.status == RUNNING:
                    break
                settled += 1
            self._drained += settled
            return list(fresh)

    def finalize(self, status: str = OK, title: str = "", **meta: Any) -> None:
        with self._lock:
            self.status = status
            if title:
                self.title = self._clean_text(title)
            self._meta = self._clean(meta)

    def to_dict(self) -> dict:
        with self._lock:
            steps = [s.to_dict() for s in self._steps]
            return {
                "trace_id": self.trace_id,
                "pipeline": self.pipeline,
                "owner": self.owner,
                "status": self.status,
                "title": self.title,
                "started_at": self.started_at,
                "duration_ms": round((time.time() - self.started_at) * 1000, 2),
                "steps": steps,
                "meta": getattr(self, "_meta", {}),
            }

    def public(self) -> dict:
        """What the API returns. `owner` is an internal ACL, not a field to echo
        — same rule as `JobStatus.public`."""
        data = self.to_dict()
        data.pop("owner", None)
        return data


# -- binding -------------------------------------------------------------------

_TRACER: ContextVar[Tracer | None] = ContextVar("graphrag_tracer", default=None)
# The step a new step nests under. Held separately from the tracer so each
# concurrent branch carries its own parent: the three retrieval legs run under
# copies of the context, so each sees the tool call that spawned it rather than
# whichever sibling happened to start last.
_CURRENT: ContextVar[str | None] = ContextVar("graphrag_trace_parent", default=None)


def active_tracer() -> Tracer | None:
    """The tracer for the run in flight, or None outside a traced run."""
    return _TRACER.get()


@contextmanager
def use_tracer(tracer: Tracer | None) -> Iterator[Tracer | None]:
    """Bind a tracer for one run.

    Must be entered by whatever *invokes* the pipeline, exactly like
    `use_plan` and `use_meter`: LangGraph copies the context at submit time, so
    anything bound after that point is invisible inside the tools.
    """
    token = _TRACER.set(tracer)
    try:
        yield tracer
    finally:
        _TRACER.reset(token)


@contextmanager
def step(
    name: str,
    *,
    label: str | None = None,
    kind: str | None = None,
    **payload: Any,
) -> Iterator[Any]:
    """Record one block of the pipeline.

        with step(steps.RETRIEVAL_FUSE, lists=3) as s:
            fused = reciprocal_rank_fusion(lists)
            s.output(candidates=len(fused))

    Outside a traced run this yields `NULL_STEP` and does nothing else, so the
    cost on the normal path is one ContextVar read.

    An exception is recorded as `status="error"` with a scrubbed message and
    re-raised — tracing observes the pipeline, it never swallows its failures.
    """
    tracer = _TRACER.get()
    if tracer is None:
        yield NULL_STEP
        return

    live = tracer.begin(name, label, kind, payload)
    if live is NULL_STEP:  # cap reached
        yield NULL_STEP
        return

    token = _CURRENT.set(live.id)
    try:
        yield live
    except BaseException as exc:
        tracer.finish(live, exc)
        raise
    else:
        tracer.finish(live)
    finally:
        _CURRENT.reset(token)


__all__ = [
    "ERROR", "OK", "RUNNING", "SKIPPED",
    "TraceStep", "Tracer", "active_tracer", "step", "use_tracer",
]
