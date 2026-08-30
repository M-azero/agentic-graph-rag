"""Pipeline step tracing — what the inspector page draws.

`graphrag.observability` is the llmlens integration (spans shipped to an
external server). This package is the in-request step recorder that feeds the
UI: see `recorder.py` for why it binds through a ContextVar.
"""

from graphrag.trace.recorder import (
    ERROR,
    OK,
    RUNNING,
    SKIPPED,
    Tracer,
    TraceStep,
    active_tracer,
    step,
    use_tracer,
)
from graphrag.trace.store import TraceStore

__all__ = [
    "ERROR",
    "OK",
    "RUNNING",
    "SKIPPED",
    "TraceStep",
    "TraceStore",
    "Tracer",
    "active_tracer",
    "step",
    "use_tracer",
]
