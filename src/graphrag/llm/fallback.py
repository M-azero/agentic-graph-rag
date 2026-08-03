"""Provider failover: an ordered chain of chat models presented as one model.

A single API key going bad should degrade the deployment, not stop it. This
wraps N configured (provider, model) pairs so a call that fails on the first
one is retried on the next, transparently to every caller.

Three things this has to get right, none of them obvious:

**Never replay a stream.** If the primary dies *after* emitting tokens, falling
over would re-send the answer from the beginning and the user would watch it
print twice. So a stream may only fail over before its first chunk; once any
token has reached the caller the error propagates.

**Stop calling a provider that is reliably dead.** A denied API key fails in
~200 ms *every single call*. Without a circuit breaker a dead provider is a
permanent latency tax on every request. After `max_failures` consecutive
failures a member is taken out of rotation for `cooldown_seconds`, so it costs
one probe per cooldown instead of one round-trip per request. Breaker state is
shared across `bind_tools()` derivatives — the agent rebuilds its model on every
request, and a breaker that reset that often would never trip at all.

**Don't fail over when the next provider will fail identically.** A prompt over
the context limit, or one the provider's own safety filter refused, is a
property of the request, not of the provider's availability. Retrying costs a
second call to reach the same answer, so those errors propagate immediately.

This must subclass `BaseChatModel` rather than use LangChain's generic
`Runnable.with_fallbacks`: `create_react_agent` type-checks for `BaseChatModel`
and calls `.bind_tools()` on it, which `RunnableWithFallbacks` does not expose.
`_stream`/`_astream` must both be overridden for the same reason token streaming
works at all — `BaseChatModel._should_stream` decides by checking whether this
class overrides them.

Callbacks are deliberately *not* forwarded to the wrapped members. LangGraph's
`stream_mode="messages"` emits a token for every `on_llm_new_token`, and
langchain-core already fires that for the chunks yielded here; passing the
manager down as well would emit each token twice.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from prometheus_client import Counter
from pydantic import ConfigDict

from graphrag.core.errors import ProviderError
from graphrag.core.logging import get_logger

log = get_logger(__name__)

_FAILOVER = Counter(
    "graphrag_llm_failover_total",
    "Chat-model calls that fell through to a backup provider",
    ["from_model", "to_model", "reason"],
)
_BREAKER = Counter(
    "graphrag_llm_breaker_open_total",
    "Times a provider was taken out of rotation after repeated failures",
    ["model"],
)

# Failure reasons, kept to a closed set — these become Prometheus label values,
# and raw provider messages would be unbounded cardinality.
_TIMEOUT, _RATE_LIMIT, _AUTH, _SERVER, _CONNECTION, _OTHER = (
    "timeout", "rate_limit", "auth", "server_error", "connection", "other",
)

# Errors that are a property of the *request*, not of the provider. Sending the
# same prompt to the next model in the chain reproduces them, so failing over
# just spends a second call to arrive at the same place.
_PERMANENT = (
    "context length", "context_length", "maximum context", "too many tokens",
    "string too long", "content filter", "content_filter", "safety",
    "blocked", "recitation", "invalid schema",
)

_REASON_MARKERS = (
    (_TIMEOUT, ("timeout", "timed out", "deadline exceeded")),
    (_RATE_LIMIT, ("rate limit", "rate_limit", "429", "quota", "resource_exhausted",
                   "too many requests")),
    (_AUTH, ("permission_denied", "unauthenticated", "invalid api key", "invalid_api_key",
             "api key not valid", "401", "403", "unauthorized", "forbidden",
             "billing", "dunning", "insufficient")),
    (_SERVER, ("500", "502", "503", "504", "internal server error", "bad gateway",
               "service unavailable", "overloaded")),
    (_CONNECTION, ("connection", "connect", "network", "dns", "unreachable", "ssl")),
)


def classify(exc: BaseException) -> tuple[bool, str]:
    """Return (should_fail_over, reason) for a provider exception.

    Unrecognized errors fail over. Between the two ways of being wrong, trying
    the next provider costs a round-trip; refusing to costs the user their
    answer — and the breaker bounds the wasted calls either way.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(marker in text for marker in _PERMANENT):
        return False, "permanent"

    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        if status == 429:
            return True, _RATE_LIMIT
        if status in (401, 403):
            return True, _AUTH
        if status >= 500:
            return True, _SERVER
        if status == 404:
            return True, _OTHER
        if 400 <= status < 500:
            # A 4xx the provider considers our fault and no marker explained.
            # The next model would very likely reject it the same way.
            return False, "permanent"

    for reason, markers in _REASON_MARKERS:
        if any(marker in text for marker in markers):
            return True, reason
    return True, _OTHER


class Health:
    """Failure counters and cooldowns for one chain.

    Lives outside the pydantic model and is passed by reference so every
    `bind_tools()` derivative shares it. Without that, the agent — which rebinds
    tools on every single request — would hand each request a chain with a clean
    slate and no breaker would ever reach its threshold.
    """

    def __init__(self, size: int, max_failures: int, cooldown_seconds: float) -> None:
        self.size = size
        self.max_failures = max_failures
        self.cooldown_seconds = cooldown_seconds
        self._failures: dict[int, int] = {}
        self._open_until: dict[int, float] = {}

    def order(self) -> list[int]:
        """Member indices to try: healthy ones first, then any in cooldown.

        Cooled-down members stay at the back rather than being removed. If every
        provider has tripped, a long-shot call still beats a guaranteed failure.
        """
        now = time.monotonic()
        healthy = [i for i in range(self.size) if self._open_until.get(i, 0.0) <= now]
        tripped = [i for i in range(self.size) if self._open_until.get(i, 0.0) > now]
        return healthy + tripped

    def is_open(self, index: int) -> bool:
        return self._open_until.get(index, 0.0) > time.monotonic()

    def succeeded(self, index: int) -> None:
        self._failures.pop(index, None)
        self._open_until.pop(index, None)

    def failed(self, index: int, label: str) -> None:
        count = self._failures.get(index, 0) + 1
        self._failures[index] = count
        if count >= self.max_failures and not self.is_open(index):
            self._open_until[index] = time.monotonic() + self.cooldown_seconds
            _BREAKER.labels(label).inc()
            log.warning(
                "llm_breaker_open",
                model=label,
                failures=count,
                cooldown_s=self.cooldown_seconds,
            )


def _as_chunk(message: Any) -> ChatGenerationChunk:
    if isinstance(message, ChatGenerationChunk):
        return message
    return ChatGenerationChunk(message=message)


def _as_result(message: Any) -> ChatResult:
    if not isinstance(message, BaseMessage):
        message = AIMessage(content=str(message))
    return ChatResult(generations=[ChatGeneration(message=message)])


class FallbackChatModel(BaseChatModel):
    """An ordered chain of chat models that behaves like a single one.

    `members` are already-constructed models (or tool-bound runnables); `labels`
    are their `provider:model` names, used for logs and metric labels.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    members: list[Any]
    labels: list[str]
    health: Any

    @property
    def _llm_type(self) -> str:
        return "graphrag-fallback"

    @property
    def primary_label(self) -> str:
        return self.labels[0] if self.labels else "none"

    def _kwargs(self, stop: list[str] | None, kwargs: dict) -> dict:
        # Only pass `stop` when there is one: some providers reject an explicit
        # stop=None, and every provider treats "absent" as "no stop sequences".
        return {**kwargs, "stop": stop} if stop else dict(kwargs)

    def _exhausted(self, errors: list[tuple[str, BaseException]]) -> ProviderError:
        detail = "; ".join(f"{label} -> {type(exc).__name__}: {exc}" for label, exc in errors)
        return ProviderError(f"Every chat provider in the chain failed: {detail}")

    def _note_failover(self, from_index: int, reason: str, order: list[int]) -> None:
        position = order.index(from_index)
        nxt = order[position + 1] if position + 1 < len(order) else None
        _FAILOVER.labels(
            self.labels[from_index], self.labels[nxt] if nxt is not None else "none", reason
        ).inc()
        log.warning(
            "llm_failover",
            failed=self.labels[from_index],
            next=self.labels[nxt] if nxt is not None else "none",
            reason=reason,
        )

    # -- non-streaming --------------------------------------------------------

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        call = self._kwargs(stop, kwargs)
        errors: list[tuple[str, BaseException]] = []
        order = self.health.order()
        for index in order:
            try:
                message = self.members[index].invoke(messages, **call)
            except Exception as exc:
                errors.append((self.labels[index], exc))
                self.health.failed(index, self.labels[index])
                fail_over, reason = classify(exc)
                if not fail_over:
                    raise
                self._note_failover(index, reason, order)
                continue
            self.health.succeeded(index)
            return _as_result(message)
        raise self._exhausted(errors)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        call = self._kwargs(stop, kwargs)
        errors: list[tuple[str, BaseException]] = []
        order = self.health.order()
        for index in order:
            try:
                message = await self.members[index].ainvoke(messages, **call)
            except Exception as exc:
                errors.append((self.labels[index], exc))
                self.health.failed(index, self.labels[index])
                fail_over, reason = classify(exc)
                if not fail_over:
                    raise
                self._note_failover(index, reason, order)
                continue
            self.health.succeeded(index)
            return _as_result(message)
        raise self._exhausted(errors)

    # -- streaming ------------------------------------------------------------

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        call = self._kwargs(stop, kwargs)
        errors: list[tuple[str, BaseException]] = []
        order = self.health.order()
        for index in order:
            emitted = False
            try:
                for chunk in self.members[index].stream(messages, **call):
                    emitted = True
                    yield _as_chunk(chunk)
            except Exception as exc:
                self.health.failed(index, self.labels[index])
                if emitted:
                    # Tokens are already on their way to the caller. Starting
                    # another provider here would print the answer twice.
                    log.warning(
                        "llm_stream_failed_midway", model=self.labels[index], error=str(exc)
                    )
                    raise
                errors.append((self.labels[index], exc))
                fail_over, reason = classify(exc)
                if not fail_over:
                    raise
                self._note_failover(index, reason, order)
                continue
            self.health.succeeded(index)
            return
        raise self._exhausted(errors)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        call = self._kwargs(stop, kwargs)
        errors: list[tuple[str, BaseException]] = []
        order = self.health.order()
        for index in order:
            emitted = False
            try:
                async for chunk in self.members[index].astream(messages, **call):
                    emitted = True
                    yield _as_chunk(chunk)
            except Exception as exc:
                self.health.failed(index, self.labels[index])
                if emitted:
                    log.warning(
                        "llm_stream_failed_midway", model=self.labels[index], error=str(exc)
                    )
                    raise
                errors.append((self.labels[index], exc))
                fail_over, reason = classify(exc)
                if not fail_over:
                    raise
                self._note_failover(index, reason, order)
                continue
            self.health.succeeded(index)
            return
        raise self._exhausted(errors)

    # -- tool binding ---------------------------------------------------------

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> FallbackChatModel:
        """Bind tools to every member and keep them one model.

        `health` is passed through, not rebuilt: the agent rebinds on every
        request, so a fresh breaker here would forget a provider was dead
        between one question and the next.
        """
        return FallbackChatModel(
            members=[m.bind_tools(tools, **kwargs) for m in self.members],
            labels=list(self.labels),
            health=self.health,
        )
