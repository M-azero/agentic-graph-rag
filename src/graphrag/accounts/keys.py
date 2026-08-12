"""API keys, stored in Postgres.

Same scheme the Redis KeyStore used — `grk_` prefix, SHA-256 hash, shown once —
so existing clients and the `graphrag apikey` CLI keep working; only the
storage moved somewhere that survives a cache flush.

Resolution is on the hot path for every programmatic request, so verified
lookups are cached in Redis briefly. The cache is keyed by the key's hash and
holds only the tenant identity, never the key itself.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from graphrag.auth import generate_api_key, hash_key
from graphrag.db.engine import session_scope
from graphrag.db.models import APIKey, User

_CACHE_PREFIX = "graphrag:apikey:"
_CACHE_TTL = 60  # seconds; short enough that a revoke takes effect promptly


@dataclass(frozen=True)
class KeyOwner:
    user_id: str
    tenant_id: str
    role: str
    status: str
    email: str = ""


class PgKeyStore:
    def __init__(
        self, factory: async_sessionmaker[AsyncSession] | None, redis_client=None
    ) -> None:
        self._factory = factory
        self._redis = redis_client

    async def create_key(self, user_id: str, label: str = "") -> tuple[int, str]:
        """Mint a key for a user; return `(id, plaintext)`.

        The id comes from the inserted row rather than from re-listing the
        user's keys afterwards and taking the newest. That read-back raced: two
        keys created in the same second sort ambiguously by `created_at`, so the
        caller could be handed one key's plaintext alongside another key's id —
        and then revoke the wrong one.
        """
        key = generate_api_key()
        async with session_scope(self._factory) as s:
            row = APIKey(user_id=user_id, key_hash=hash_key(key), label=label[:64])
            s.add(row)
            await s.flush()  # assign the primary key inside the transaction
            key_id = row.id
        return key_id, key

    @property
    def available(self) -> bool:
        return self._factory is not None

    async def resolve(self, key: str) -> KeyOwner | None:
        """Return the owner of a live key, or None.

        A revoked key, or one whose owner is not active, resolves to None —
        suspending an account must cut off its API keys too, not just its
        browser sessions. So does any key at all when there is no database to
        check it against: this runs on every programmatic request, and raising
        there turns "accounts are not configured" into a 500 on every call.
        """
        if not key or self._factory is None:
            return None
        h = hash_key(key)
        cached = self._cache_get(h)
        if cached is not None:
            return cached

        async with session_scope(self._factory) as s:
            row = (
                await s.execute(
                    select(APIKey, User)
                    .join(User, User.id == APIKey.user_id)
                    .where(APIKey.key_hash == h, APIKey.revoked_at.is_(None))
                )
            ).first()
            if row is None:
                return None
            api_key, user = row
            if user.status != "active":
                return None
            # Throttled: one write per key per cache window, not per request.
            await s.execute(
                update(APIKey)
                .where(APIKey.id == api_key.id)
                .values(last_used_at=datetime.now(UTC))
            )
            owner = KeyOwner(
                str(user.id), user.tenant_id, user.role, user.status, user.email
            )

        self._cache_put(h, owner)
        return owner

    async def revoke_user(self, user_id: str) -> int:
        """Revoke every key a user holds. Returns how many were live."""
        async with session_scope(self._factory) as s:
            hashes = (
                await s.execute(
                    select(APIKey.key_hash).where(
                        APIKey.user_id == user_id, APIKey.revoked_at.is_(None)
                    )
                )
            ).scalars().all()
            if hashes:
                await s.execute(
                    update(APIKey)
                    .where(APIKey.user_id == user_id, APIKey.revoked_at.is_(None))
                    .values(revoked_at=datetime.now(UTC))
                )
        for h in hashes:
            self._cache_drop(h)
        return len(hashes)

    async def revoke_one(self, user_id: str, key_id: int) -> bool:
        async with session_scope(self._factory) as s:
            row = (
                await s.execute(
                    select(APIKey).where(
                        APIKey.id == key_id,
                        APIKey.user_id == user_id,
                        APIKey.revoked_at.is_(None),
                    )
                )
            ).scalar_one_or_none()
            if row is None:
                return False
            row.revoked_at = datetime.now(UTC)
            key_hash = row.key_hash
        self._cache_drop(key_hash)
        return True

    async def list_keys(self, user_id: str) -> list[APIKey]:
        async with session_scope(self._factory) as s:
            return list(
                (
                    await s.execute(
                        select(APIKey)
                        .where(APIKey.user_id == user_id, APIKey.revoked_at.is_(None))
                        .order_by(APIKey.created_at.desc())
                    )
                ).scalars().all()
            )

    # -- cache ----------------------------------------------------------------
    def _cache_get(self, key_hash: str) -> KeyOwner | None:
        if self._redis is None:
            return None
        with contextlib.suppress(Exception):
            raw = self._redis.get(_CACHE_PREFIX + key_hash)
            if raw:
                user_id, tenant_id, role, status, email = raw.split("|", 4)
                return KeyOwner(user_id, tenant_id, role, status, email)
        return None

    def _cache_put(self, key_hash: str, owner: KeyOwner) -> None:
        if self._redis is None:
            return
        with contextlib.suppress(Exception):
            self._redis.setex(
                _CACHE_PREFIX + key_hash,
                _CACHE_TTL,
                f"{owner.user_id}|{owner.tenant_id}|{owner.role}|"
                f"{owner.status}|{owner.email}",
            )

    def _cache_drop(self, key_hash: str) -> None:
        if self._redis is None:
            return
        with contextlib.suppress(Exception):
            self._redis.delete(_CACHE_PREFIX + key_hash)
