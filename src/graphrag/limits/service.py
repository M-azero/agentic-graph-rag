"""Per-user limits: what they are, and whether this request fits inside them.

Limits come in two speeds, and they are stored accordingly.

**Fast counters** (messages per minute/day, tokens per day/month) live in Redis
as fixed-window counters. They are checked on every chat request, so they must
not cost a database round trip. Fixed windows rather than sliding: a sliding
window needs a sorted set per user and an O(log n) trim per request to make a
boundary burst slightly smoother, which is not a trade worth making here.

**Slow checks** (file count, storage, chunk rows, threads) read the real thing —
Postgres or the tenant's DuckDB file. They run only on the endpoints that
create those objects, where one query is noise next to the upload itself, and
reading the truth means a failed ingest or a deleted file gives the quota back
automatically.

Effective limits are the global defaults with the user's overrides layered on
top, cached briefly in Redis and dropped when an admin edits either.

If Redis is unreachable the fast counters fail *open* and log. The alternative
is locking every user out of chat because a cache is down, which is a worse
outage than a few unmetered minutes; the slow checks, which guard durable
resources, keep working regardless because they read Postgres.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from graphrag.core.logging import get_logger
from graphrag.db.engine import session_scope
from graphrag.db.models import LIMIT_COLUMNS, GlobalLimit, UserLimit

log = get_logger(__name__)

_CACHE_PREFIX = "graphrag:limits:"
_CACHE_TTL = 300

# Windows are fixed and keyed by the clock, so every process agrees on the
# boundary without coordinating.
_WINDOWS = {
    "minute": ("%Y%m%d%H%M", 120),
    "day": ("%Y%m%d", 172_800),
    "month": ("%Y%m", 3_024_000),
}


@dataclass(frozen=True)
class Limits:
    messages_per_minute: int
    messages_per_day: int
    tokens_per_day: int
    tokens_per_month: int
    max_files: int
    max_file_mb: int
    max_storage_mb: int
    max_chunks: int
    max_threads: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class LimitBreach:
    """Which limit stopped this request, and what to tell the caller."""

    limit: str
    used: int
    maximum: int
    retry_after: int = 0

    @property
    def message(self) -> str:
        pretty = self.limit.replace("_", " ")
        if self.retry_after:
            return f"You've reached your limit of {self.maximum} {pretty}. Try again later."
        return f"You've reached your limit of {self.maximum} {pretty}."

    def as_detail(self) -> dict:
        return {
            "code": "limit_exceeded",
            "limit": self.limit,
            "used": self.used,
            "max": self.maximum,
            "retry_after": self.retry_after,
            "message": self.message,
        }


def _window_key(kind: str, user_id: str, window: str) -> str:
    fmt, _ttl = _WINDOWS[window]
    stamp = datetime.now(UTC).strftime(fmt)
    return f"rl:{kind}:{window}:{user_id}:{stamp}"


def _seconds_left(window: str) -> int:
    """How long until the current window rolls over — the Retry-After hint."""
    now = datetime.now(UTC)
    if window == "minute":
        return 60 - now.second
    if window == "day":
        return (24 - now.hour) * 3600 - now.minute * 60 - now.second
    days_left = 31 - now.day
    return max(60, days_left * 86400)


class LimitService:
    def __init__(
        self, factory: async_sessionmaker[AsyncSession] | None, redis_client=None
    ) -> None:
        self._factory = factory
        self._redis = redis_client

    # -- effective limits -----------------------------------------------------
    async def effective(self, user_id: str) -> Limits:
        cached = self._cache_get(user_id)
        if cached is not None:
            return cached

        values: dict[str, int] = {}
        if self._factory is not None:
            async with session_scope(self._factory) as s:
                glob = (await s.execute(select(GlobalLimit))).scalars().first()
                if glob is not None:
                    values = {c: getattr(glob, c) for c in LIMIT_COLUMNS}
                override = (
                    await s.execute(select(UserLimit).where(UserLimit.user_id == user_id))
                ).scalar_one_or_none()
                if override is not None:
                    # NULL means "inherit", so only set columns override.
                    for column in LIMIT_COLUMNS:
                        value = getattr(override, column)
                        if value is not None:
                            values[column] = value
        if not values:
            values = _DEFAULTS.copy()

        limits = Limits(**values)
        self._cache_put(user_id, limits)
        return limits

    def invalidate(self, user_id: str | None = None) -> None:
        """Drop cached limits after an admin edit. Without a user id, drop all
        of them — that's what a change to the global defaults means."""
        if self._redis is None:
            return
        with contextlib.suppress(Exception):
            if user_id:
                self._redis.delete(_CACHE_PREFIX + user_id)
                return
            for key in self._redis.scan_iter(match=_CACHE_PREFIX + "*", count=500):
                self._redis.delete(key)

    # -- fast counters --------------------------------------------------------
    def check_messages(self, user_id: str, limits: Limits) -> LimitBreach | None:
        """Would one more message exceed the per-minute or per-day allowance?"""
        for window, maximum in (
            ("minute", limits.messages_per_minute),
            ("day", limits.messages_per_day),
        ):
            used = self._peek("msg", user_id, window)
            if used is not None and used >= maximum:
                return LimitBreach(
                    f"messages_per_{window}", used, maximum, _seconds_left(window)
                )
        return None

    def record_message(self, user_id: str) -> None:
        for window in ("minute", "day"):
            self._bump("msg", user_id, window, 1)

    def check_tokens(self, user_id: str, limits: Limits) -> LimitBreach | None:
        """Tokens are post-paid — the cost isn't known until the answer is
        streamed — so the gate is "is there budget left", not "does this fit"."""
        for window, maximum in (
            ("day", limits.tokens_per_day),
            ("month", limits.tokens_per_month),
        ):
            used = self._peek("tok", user_id, window)
            if used is not None and used >= maximum:
                return LimitBreach(
                    f"tokens_per_{window}", used, maximum, _seconds_left(window)
                )
        return None

    def record_tokens(self, user_id: str, tokens: int) -> None:
        if tokens <= 0:
            return
        for window in ("day", "month"):
            self._bump("tok", user_id, window, tokens)

    async def usage_snapshot(self, user_id: str) -> dict[str, int]:
        """Current counters, for the account page's meters."""
        return {
            "messages_this_minute": self._peek("msg", user_id, "minute") or 0,
            "messages_today": self._peek("msg", user_id, "day") or 0,
            "tokens_today": self._peek("tok", user_id, "day") or 0,
            "tokens_this_month": self._peek("tok", user_id, "month") or 0,
        }

    # -- redis helpers --------------------------------------------------------
    def _peek(self, kind: str, user_id: str, window: str) -> int | None:
        """Current count, or None when Redis is unreachable (fail open)."""
        if self._redis is None:
            return None
        try:
            raw = self._redis.get(_window_key(kind, user_id, window))
            return int(raw) if raw else 0
        except Exception as exc:
            log.warning("limit_counter_unavailable", error=str(exc))
            return None

    def _bump(self, kind: str, user_id: str, window: str, amount: int) -> None:
        if self._redis is None:
            return
        key = _window_key(kind, user_id, window)
        _fmt, ttl = _WINDOWS[window]
        with contextlib.suppress(Exception):
            pipe = self._redis.pipeline()
            pipe.incrby(key, amount)
            # Only on creation: refreshing the TTL every request would keep a
            # busy user's window alive indefinitely and it would never reset.
            pipe.expire(key, ttl, nx=True)
            pipe.execute()

    def _cache_get(self, user_id: str) -> Limits | None:
        if self._redis is None:
            return None
        with contextlib.suppress(Exception):
            raw = self._redis.get(_CACHE_PREFIX + user_id)
            if raw:
                return Limits(**json.loads(raw))
        return None

    def _cache_put(self, user_id: str, limits: Limits) -> None:
        if self._redis is None:
            return
        with contextlib.suppress(Exception):
            self._redis.setex(
                _CACHE_PREFIX + user_id, _CACHE_TTL, json.dumps(limits.as_dict())
            )


# Used only when the database has no limits row at all (e.g. auth disabled).
# Mirrors the server_default values in migrations 0001 and 0003 — the token
# ceilings were raised in 0003 to cover the extra model calls answer review
# makes, so changing either one here needs a matching migration.
_DEFAULTS: dict[str, int] = {
    "messages_per_minute": 6,
    "messages_per_day": 100,
    "tokens_per_day": 300_000,
    "tokens_per_month": 4_000_000,
    "max_files": 10,
    "max_file_mb": 15,
    "max_storage_mb": 100,
    "max_chunks": 20_000,
    "max_threads": 25,
}
