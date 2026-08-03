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
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler

from graphrag.core.logging import get_logger
from graphrag.usage.recorder import estimate_tokens

log = get_logger(__name__)


class TokenMeter(AsyncCallbackHandler):
    """Accumulates prompt/completion tokens across every model call in one run."""

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

    # -- prompt side ----------------------------------------------------------
    async def on_chat_model_start(
        self, serialized: dict, messages: list[list[Any]], **kwargs: Any
    ) -> None:
        """Every model call, including the tool-loop turns that resend context."""
        self.calls += 1
        self._estimated_input += sum(
            estimate_tokens(_text_of(m)) for batch in messages for m in batch
        )

    async def on_llm_start(
        self, serialized: dict, prompts: list[str], **kwargs: Any
    ) -> None:
        """Completion-style models, for the same reason."""
        self.calls += 1
        self._estimated_input += sum(estimate_tokens(p) for p in prompts)

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
            self.reported = True
            self.input_tokens += got_input
            self.output_tokens += got_output
        else:
            # Streamed call with no usage block: keep this turn's estimate.
            self._estimated_output += estimate_tokens(_response_text(response))

    # -- what the caller books ------------------------------------------------
    @property
    def totals(self) -> tuple[int, int]:
        """(input, output), never both zero after a real model call.

        A run mixing reported and unreported calls takes the larger of the two
        views rather than adding them, since the estimate covers calls the
        provider may already have counted.
        """
        return (
            max(self.input_tokens, self._estimated_input),
            max(self.output_tokens, self._estimated_output),
        )


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
