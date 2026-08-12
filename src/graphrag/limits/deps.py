"""FastAPI dependencies that enforce per-user limits.

A breach is a 429 whose body names the limit, what it is, and how much has been
used — enough for the UI to render "12 of 12 messages today, resets in 4 hours"
rather than a bare "too many requests". `Retry-After` is set so well-behaved
clients back off on their own.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request

from graphrag.api.deps import AuthUser, get_current_user
from graphrag.limits.service import LimitBreach, Limits, LimitService
from graphrag.usage.recorder import MESSAGE


def get_limits(request: Request) -> LimitService:
    return request.app.state.limits


def _reject(breach: LimitBreach) -> HTTPException:
    headers = {"Retry-After": str(breach.retry_after)} if breach.retry_after else None
    return HTTPException(status_code=429, detail=breach.as_detail(), headers=headers)


async def enforce_message_limits(
    request: Request,
    user: AuthUser = Depends(get_current_user),
    limits: LimitService = Depends(get_limits),
) -> AuthUser:
    """Gate a chat request on message rate and remaining token budget.

    The message counter is incremented here rather than after the answer: the
    expensive part is the model call, so an attempt has to cost quota even if
    the answer later fails, or a failing request becomes a free retry loop.

    Two counters, deliberately, and they are not redundant. The Redis window is
    what enforces the rate; the `usage_events` row is the durable record the
    admin charts read. Only the first existed, which is why every user showed
    zero messages in the console however much they chatted.
    """
    effective = await limits.effective(user.user_id)
    breach = limits.check_messages(user.user_id, effective) or limits.check_tokens(
        user.user_id, effective
    )
    if breach is not None:
        raise _reject(breach)
    limits.record_message(user.user_id)

    recorder = getattr(request.app.state, "usage", None)
    if recorder is not None:
        await recorder.record(user.user_id, MESSAGE, 1)
    return user


async def effective_limits(
    user: AuthUser = Depends(get_current_user),
    limits: LimitService = Depends(get_limits),
) -> Limits:
    return await limits.effective(user.user_id)


def reject_with(breach: LimitBreach) -> HTTPException:
    """For routers that do their own slow checks (uploads, threads)."""
    return _reject(breach)
