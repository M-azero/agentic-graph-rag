"""Fixtures for tests that need real services.

Run them with a live Postgres (and Redis, optionally):

    docker compose up -d postgres redis neo4j
    GRAPHRAG_TEST_DATABASE_URL=postgresql://graphrag:change-me@localhost:5432/graphrag \
      pytest -m integration

Each test gets a schema-fresh database: tables are created from the models and
dropped afterwards, so a failed run cannot poison the next one.

**Run these with `GRAPHRAG_PROFILE=api`** (CI has no `.env`, so it gets that by
default). A developer `.env` that sets `GRAPHRAG_PROFILE=production` is read by
`Container()` here too, and the production profile pins `cookie_secure: "true"`
— httpx then refuses to send the session cookie back over the `http://test`
base URL and roughly half of these fail with a baffling 401.
"""

from __future__ import annotations

import asyncio
import os
import selectors
import sys

import pytest
import pytest_asyncio

from graphrag.db.engine import build_engine, build_sessionmaker
from graphrag.db.models import Base, GlobalLimit

DSN_ENV = "GRAPHRAG_TEST_DATABASE_URL"

# These fixtures drop every table between runs, so the database they point at
# must be disposable. Requiring the name to say so is a cheap guard against the
# afternoon someone exports a real DSN and loses their accounts table.
_TEST_MARKERS = ("test", "tmp", "scratch", "ci")


def _dsn() -> str | None:
    return os.getenv(DSN_ENV) or os.getenv("GRAPHRAG_DATABASE_URL")


def _database_name(dsn: str) -> str:
    return dsn.rsplit("/", 1)[-1].split("?")[0].lower()


def _checked_dsn() -> str | None:
    dsn = _dsn()
    if dsn is None:
        return None
    name = _database_name(dsn)
    if not any(marker in name for marker in _TEST_MARKERS):
        pytest.fail(
            f"Refusing to run destructive tests against database {name!r}: these "
            f"fixtures drop every table. Point {DSN_ENV} at a database whose name "
            f"contains one of {_TEST_MARKERS}.",
            pytrace=False,
        )
    return dsn


requires_db = pytest.mark.skipif(
    _dsn() is None, reason=f"set {DSN_ENV} to run database integration tests"
)


@pytest.fixture(scope="session")
def event_loop():
    """A selector loop: psycopg's async mode refuses Windows' default
    ProactorEventLoop, and these tests exercise the psycopg-backed
    checkpointer alongside asyncpg."""
    if sys.platform == "win32":
        loop = asyncio.SelectorEventLoop(selectors.SelectSelector())
    else:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def engine():
    dsn = _checked_dsn()
    if dsn is None:
        pytest.skip(f"set {DSN_ENV}")
    eng = build_engine(dsn)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await eng.dispose()


@pytest_asyncio.fixture
async def db(engine):
    factory = build_sessionmaker(engine)
    # The limits singleton is created by migration 0001; metadata.create_all
    # does not run migrations, so seed it here.
    async with factory() as s:
        s.add(GlobalLimit(id=1))
        await s.commit()
    return factory


class CapturingSender:
    """Email sender that keeps messages in memory so a test can read the code
    that would have been mailed."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str]] = []

    async def send(self, to: str, subject: str, text: str) -> bool:
        self.sent.append((to, subject, text))
        return True

    def last_code(self, to: str | None = None) -> str:
        """The six-digit code from the most recent message."""
        import re

        for recipient, _subject, body in reversed(self.sent):
            if to is None or recipient == to:
                match = re.search(r"\b(\d{6})\b", body)
                if match:
                    return match.group(1)
        raise AssertionError(f"no verification code was sent to {to}")


@pytest.fixture
def email_sender() -> CapturingSender:
    return CapturingSender()


def relax_auth_limits(container) -> None:
    """Turn off the per-IP `/auth/*` throttles for a test app.

    These tests drive signup and login dozens of times from one address, which
    is exactly what the throttle exists to stop. Worse, when a local Redis is
    reachable the limiter is Redis-backed, so the buckets are shared by every
    app instance in the run *and* survive between runs — a suite that passed
    once would then fail for the next ten minutes.

    The throttle itself is covered by `tests/unit/test_auth_ratelimit.py` (which
    address is counted) and `tests/integration/test_auth_throttle.py` (that the
    limit fires), so nothing is lost by disabling it here.
    """
    limits = container.settings.auth.rate_limits
    for field in type(limits).model_fields:
        setattr(limits, field, "")
