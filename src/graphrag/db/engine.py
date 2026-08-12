"""Async database engine and session helpers.

One engine per process, created in the API lifespan and disposed on shutdown.
The pool is deliberately small, and stays small on a bigger box: a large pool
buys nothing but Postgres backends competing for the same cores, and
`max_connections` is a shared budget with the checkpointer's own pool. The API
also runs a single uvicorn worker (the DuckDB vector provider takes an
exclusive lock per tenant file), so one process holds this whole pool.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from graphrag.core.errors import ConfigError
from graphrag.core.logging import get_logger

log = get_logger(__name__)

_ASYNC_PREFIX = "postgresql+asyncpg://"


def normalize_dsn(url: str) -> str:
    """Accept the plain `postgresql://` form people paste from hosting panels
    and point it at the async driver."""
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        url = _ASYNC_PREFIX + url[len("postgresql://") :]
    return url


def sync_dsn(url: str) -> str:
    """The psycopg (sync) SQLAlchemy form — what Alembic runs migrations on."""
    url = normalize_dsn(url)
    return "postgresql+psycopg://" + url[len(_ASYNC_PREFIX) :]


def libpq_dsn(url: str) -> str:
    """Plain `postgresql://` for libpq.

    The LangGraph Postgres checkpointer hands its connection string straight to
    psycopg, which parses libpq conninfo and rejects SQLAlchemy's `+driver`
    marker outright ('missing "=" ...'). So the driver suffix has to come back
    off before the saver sees it.
    """
    url = normalize_dsn(url)
    return "postgresql://" + url[len(_ASYNC_PREFIX) :]


def build_engine(database_url: str | None, *, echo: bool = False) -> AsyncEngine:
    if not database_url:
        raise ConfigError(
            "GRAPHRAG_DATABASE_URL is not set. The production profile stores "
            "accounts, limits, and chat history in Postgres."
        )
    return create_async_engine(
        normalize_dsn(database_url),
        echo=echo,
        pool_size=5,
        max_overflow=5,
        pool_pre_ping=True,   # a recycled connection after a Postgres restart
        pool_recycle=1800,
    )


def build_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    # expire_on_commit=False so response models can read attributes off an
    # object after its transaction closed.
    return async_sessionmaker(engine, expire_on_commit=False)


@asynccontextmanager
async def session_scope(
    factory: async_sessionmaker[AsyncSession] | None,
) -> AsyncIterator[AsyncSession]:
    """A transaction that commits on success and rolls back on error.

    A `None` factory means the deployment has no database. That used to reach
    `None()` and raise `TypeError: 'NoneType' object is not callable`, which
    every caller surfaced as a 500 — a bare stack trace where the honest answer
    is "this feature needs Postgres". Callers that can degrade should check
    before calling; this is the backstop that makes the ones that cannot fail
    legibly.
    """
    if factory is None:
        raise ConfigError(
            "No database is configured. Set GRAPHRAG_DATABASE_URL — accounts, "
            "limits, usage and chat history all live in Postgres."
        )
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
