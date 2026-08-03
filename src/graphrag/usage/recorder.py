"""Usage accounting.

Two destinations, on purpose. Redis counters are what the limit checks read, so
they must be cheap and current. The `usage_events` table is the durable record
the admin charts read — it survives a cache flush and can be aggregated over
arbitrary time ranges, which fixed-window counters cannot.

Recording is best-effort and never blocks a response: usage is billing-adjacent
bookkeeping, not something worth failing an answer the user already received.
"""

from __future__ import annotations

import contextlib
import uuid
from typing import TYPE_CHECKING

from graphrag.core.logging import get_logger
from graphrag.db.engine import session_scope
from graphrag.db.models import UsageEvent

if TYPE_CHECKING:
    # Type-only: importing this at runtime pulls in graphrag.limits.__init__,
    # which imports the FastAPI deps, which import the app — and the query
    # pipeline now imports this module, closing the loop.
    from graphrag.limits.service import LimitService

log = get_logger(__name__)

MESSAGE = "message"
TOKENS_IN = "tokens_in"
TOKENS_OUT = "tokens_out"
UPLOAD = "upload"
INGEST_CHUNKS = "ingest_chunks"

# Kinds that count against the token quota. Both sides cost money, and in RAG
# the prompt is the bigger half — every agent turn resends the system prompt,
# the conversation and the retrieved chunks. Recorded as separate events so the
# admin charts can still tell them apart (they are priced differently, often by
# 4-5x), but summed into one ceiling so a quota means "spend", not "half of it".
_BILLABLE = (TOKENS_IN, TOKENS_OUT)


class UsageRecorder:
    def __init__(self, factory=None, limits: LimitService | None = None) -> None:
        self._factory = factory
        self._limits = limits

    async def record(
        self, user_id: str, kind: str, amount: int = 1, meta: dict | None = None
    ) -> None:
        if amount <= 0:
            return
        if self._limits is not None and kind in _BILLABLE:
            self._limits.record_tokens(user_id, amount)

        if self._factory is None:
            return
        try:
            # Dev-mode identities are namespace strings, not account rows;
            # there is nothing to reference and nothing to bill.
            user_uuid = uuid.UUID(str(user_id))
        except (ValueError, AttributeError, TypeError):
            return
        try:
            async with session_scope(self._factory) as s:
                s.add(
                    UsageEvent(
                        user_id=user_uuid, kind=kind, amount=amount, meta=meta or {}
                    )
                )
        except Exception as exc:
            log.warning("usage_record_failed", kind=kind, error=str(exc))


def record_usage(redis_client, user_id: str | None, tokens: int) -> None:
    """Legacy Redis-only token counter, kept so the older /usage report and
    deployments without a database keep working."""
    if redis_client is None or tokens <= 0:
        return
    with contextlib.suppress(Exception):
        redis_client.hincrby("graphrag:usage:tokens", user_id or "default", tokens)


def estimate_tokens(text: str) -> int:
    """Approximate output tokens for an answer that wasn't streamed.

    The streaming path counts real chunks, one per `token` event. A
    non-streamed answer arrives whole, with no per-token signal and no usage
    metadata on `QueryResult`, so it has to be estimated from the text — and
    *some* estimate is the point: an uncounted answer costs the caller nothing,
    which turns `stream: false` into an unmetered path.

    Two buckets, because one ratio is wrong for half the world: ~4 characters
    per token holds for ASCII, while Arabic, Hebrew and CJK pack closer to 2.
    Charging those scripts at the ASCII rate would hand them roughly double the
    real budget. Deliberately a heuristic — quotas are a spend ceiling, not an
    invoice, and both paths only ever need to be comparable to each other.
    """
    if not text:
        return 0
    ascii_chars = sum(1 for c in text if c.isascii())
    wide_chars = len(text) - ascii_chars
    return max(1, round(ascii_chars / 4 + wide_chars / 2))


async def record_answer_tokens(
    recorder: UsageRecorder | None,
    redis_client,
    *,
    tenant_id: str | None,
    account_id: str | None,
    tokens: int,
    input_tokens: int = 0,
    meta: dict | None = None,
) -> None:
    """The one way to book a run's tokens, from every path that produces an answer.

    `tokens` is the completion side, `input_tokens` the prompt side. They are
    stored as separate events — prompt and completion are priced differently,
    often by 4-5x, so an admin chart that merged them could not explain a bill —
    and summed into the single quota counter, so the ceiling bounds real spend
    rather than the cheaper half of it.

    Both destinations belong together: the Redis counter is what `check_tokens`
    reads to enforce, and the `usage_events` row is the durable record. A path
    that does one and not the other either enforces without a record or records
    without enforcing — the second is how `stream: false` spent tokens for free.
    """
    total = max(0, tokens) + max(0, input_tokens)
    if total <= 0:
        return
    record_usage(redis_client, tenant_id, total)
    if recorder is None or not account_id:
        return
    if input_tokens > 0:
        await recorder.record(account_id, TOKENS_IN, input_tokens, meta or {})
    if tokens > 0:
        await recorder.record(account_id, TOKENS_OUT, tokens, meta or {})
