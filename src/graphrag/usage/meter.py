"""Per-run token meter.

Counts what one question actually cost, prompt side included. Input is the
larger half of a RAG bill — every agent turn resends the system prompt, the
conversation so far and the retrieved chunks — so an output-only meter watches
the cheaper number and calls it the bill.

Two design points carry the correctness:

* **It is a callback, not a walk over the final state.** LangGraph checkpoints
  the thread, so `result["messages"]` is the whole conversation, not this turn.
  Summing usage across it would re-charge every previous answer, growing the
  bill quadratically over a long chat. Callbacks fire only for calls made
  during this run.
* **Provider numbers win; the estimate is the floor.** OpenAI-compatible
  endpoints report exact usage on a normal call, but a *streamed* call only
  reports it when `stream_options.include_usage` was requested, and not every
  endpoint accepts that parameter. Rather than send a flag that could 400 the
  main chat path, the meter estimates from the prompt text it already sees and
  keeps the estimate only when the provider says nothing. Unmetered is the one
  outcome worth ruling out.

* **It travels in a ContextVar, beside the source sink and the retrieval plan.**
  The agent's own calls get the meter through LangGraph's config, but the
  reranker is not part of the graph — it is called from inside a tool, several
  thread pools deep, and under a generative provider it makes one model call per
  candidate. Threading a `config=` argument down through `HybridRetriever` and
  the `Reranker` interface would put metering in four signatures that have
  nothing else to do with it; `use_meter()` binds it once where the run starts.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler

from graphrag.core.logging import get_logger
from graphrag.usage.recorder import estimate_tokens

log = get_logger(__name__)


class TokenMeter(AsyncCallbackHandler):
    """Accumulates prompt/completion tokens across every model call in one run.

    Thread-safe, which is not optional here: the hybrid retriever runs its three
    legs concurrently and the generative reranker scores candidates on a pool of
    four, so several `on_llm_end` callbacks land at once. `self.x += n` is a
    load-add-store, and losing one of those silently undercounts a bill.
    """

    # LangChain skips handlers that raise; being explicit that a metering fault
    # must never surface as a failed answer.
    raise_error = False

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self.reported = False  # True once any provider gave real usage
        self._estimated_input = 0
        self._estimated_output = 0
        self._lock = threading.Lock()

    # -- prompt side ----------------------------------------------------------
    async def on_chat_model_start(
        self, serialized: dict, messages: list[list[Any]], **kwargs: Any
    ) -> None:
        """Every model call, including the tool-loop turns that resend context."""
        estimate = sum(
            estimate_tokens(_text_of(m)) for batch in messages for m in batch
        )
        with self._lock:
            self.calls += 1
            self._estimated_input += estimate

    async def on_llm_start(
        self, serialized: dict, prompts: list[str], **kwargs: Any
    ) -> None:
        """Completion-style models, for the same reason."""
        estimate = sum(estimate_tokens(p) for p in prompts)
        with self._lock:
            self.calls += 1
            self._estimated_input += estimate

    # -- the authoritative numbers, when the provider sends them --------------
    async def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        got_input = got_output = 0
        for generation in getattr(response, "generations", []) or []:
            for gen in generation:
                usage = _usage_of(gen)
                if not usage:
                    continue
                got_input += int(usage.get("input_tokens") or 0)
                got_output += int(usage.get("output_tokens") or 0)

        # Some providers only put usage in llm_output, not on the message.
        if not (got_input or got_output):
            raw = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
            got_input = int(raw.get("prompt_tokens") or 0)
            got_output = int(raw.get("completion_tokens") or 0)

        if got_input or got_output:
            with self._lock:
                self.reported = True
                self.input_tokens += got_input
                self.output_tokens += got_output
        else:
            # Streamed call with no usage block: keep this turn's estimate.
            estimate = estimate_tokens(_response_text(response))
            with self._lock:
                self._estimated_output += estimate

    # -- what the caller books ------------------------------------------------
    @property
    def totals(self) -> tuple[int, int]:
        """(input, output), never both zero after a real model call.

        A run mixing reported and unreported calls takes the larger of the two
        views rather than adding them, since the estimate covers calls the
        provider may already have counted.
        """
        with self._lock:
            return (
                max(self.input_tokens, self._estimated_input),
                max(self.output_tokens, self._estimated_output),
            )


_METER: ContextVar[TokenMeter | None] = ContextVar("graphrag_token_meter", default=None)


def active_meter() -> TokenMeter | None:
    """The meter bound to the work in flight, or None outside a metered run.

    None is a real answer, not a failure: the CLI, scripts and tests call the
    retrievers directly and have nothing to bill.
    """
    return _METER.get()


def meter_config(base: dict | None = None) -> dict:
    """A LangChain `config` carrying the bound meter, for components outside the
    agent graph. Returns `base` unchanged when nothing is bound.

    Sync `.invoke()` is fine: LangChain's synchronous callback manager runs
    coroutine handlers itself, so an `AsyncCallbackHandler` still fires for the
    reranker's blocking calls.
    """
    meter = _METER.get()
    if meter is None:
        return dict(base or {})
    config = dict(base or {})
    config["callbacks"] = [*config.get("callbacks", []), meter]
    return config


@contextmanager
def use_meter(meter: TokenMeter | None) -> Iterator[TokenMeter | None]:
    """Bind a meter for the duration of one request's model work.

    Must be entered by whatever *drives* the work, not by a node inside it:
    LangGraph copies the context at submit time, so anything bound later is
    invisible to the tools.
    """
    token = _METER.set(meter)
    try:
        yield meter
    finally:
        _METER.reset(token)


def _text_of(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Multimodal parts: only the text carries a token cost we can estimate.
        # An image's cost is provider-specific and is left to the reported
        # number rather than guessed at.
        return " ".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return str(content or "")


def _usage_of(generation: Any) -> dict | None:
    message = getattr(generation, "message", None)
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict) and usage:
        return usage
    info = getattr(generation, "generation_info", None) or {}
    raw = info.get("usage") or info.get("token_usage")
    if isinstance(raw, dict) and raw:
        return {
            "input_tokens": raw.get("prompt_tokens") or raw.get("input_tokens"),
            "output_tokens": raw.get("completion_tokens") or raw.get("output_tokens"),
        }
    return None


def _response_text(response: Any) -> str:
    parts: list[str] = []
    for generation in getattr(response, "generations", []) or []:
        for gen in generation:
            parts.append(getattr(gen, "text", "") or _text_of(getattr(gen, "message", "")))
    return " ".join(parts)
