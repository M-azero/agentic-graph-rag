"""Agent memory picks a backend that actually works, or none at all.

The trap: both savers connect lazily, so an unusable backend constructs
happily and only fails later — the Redis saver against plain Redis returns a
healthy-looking object and then fails every checkpoint write with "unknown
command 'JSON.SET'", one traceback per turn, while answers keep going out.
Selection therefore has to *prove* a backend before choosing it.
"""

import pytest

from graphrag.agent import graph as agent_graph
from graphrag.agent.graph import _redis_supports_checkpoints, build_checkpointer

REDIS_STACK = [{"name": "search"}, {"name": "ReJSON"}, {"name": "bf"}]
PLAIN_REDIS: list[dict] = []


class _Client:
    def __init__(self, modules, fail=False):
        self._modules, self._fail = modules, fail
        self.closed = False

    def module_list(self):
        if self._fail:
            raise ConnectionError("redis down")
        return self._modules

    def close(self):
        self.closed = True


@pytest.fixture
def fake_redis(monkeypatch):
    """Patch `redis.Redis.from_url` for the duration of one test."""

    def install(modules, fail=False):
        client = _Client(modules, fail)
        import redis

        monkeypatch.setattr(redis.Redis, "from_url", staticmethod(lambda *a, **k: client))
        return client

    return install


# ── the probe ──────────────────────────────────────────────────────────────


def test_redis_stack_is_supported(fake_redis):
    fake_redis(REDIS_STACK)
    assert _redis_supports_checkpoints("redis://x") is True


def test_plain_redis_is_not_supported(fake_redis):
    """The deployment ships plain redis:7-alpine on purpose."""
    fake_redis(PLAIN_REDIS)
    assert _redis_supports_checkpoints("redis://x") is False


@pytest.mark.parametrize(
    "modules",
    [
        [{"name": "search"}],                    # no ReJSON
        [{"name": "ReJSON"}],                    # no RediSearch
        [{"name": "timeseries"}, {"name": "bf"}],
    ],
)
def test_partial_module_sets_are_not_supported(fake_redis, modules):
    """The saver needs both FT.* and JSON.* — either alone still fails writes."""
    fake_redis(modules)
    assert _redis_supports_checkpoints("redis://x") is False


def test_module_names_are_matched_case_insensitively(fake_redis):
    """Redis reports 'ReJSON', not 'rejson'."""
    fake_redis([{"name": "SearchLight"}, {"name": "ReJSON"}])
    assert _redis_supports_checkpoints("redis://x") is True


def test_a_failed_probe_counts_as_unsupported(fake_redis):
    """Not being able to confirm a backend is a reason not to pick it."""
    fake_redis(REDIS_STACK, fail=True)
    assert _redis_supports_checkpoints("redis://x") is False


def test_the_probe_closes_its_client(fake_redis):
    client = fake_redis(REDIS_STACK)
    _redis_supports_checkpoints("redis://x")
    assert client.closed


# ── selection ──────────────────────────────────────────────────────────────


def test_plain_redis_falls_back_instead_of_writing_forever(fake_redis, monkeypatch):
    """The regression: previously this returned a Redis saver that failed every
    single checkpoint write while looking perfectly healthy."""
    fake_redis(PLAIN_REDIS)
    saver = build_checkpointer(
        "redis://x", enabled=True, backend="redis", redis_available=True,
        database_url=None,
    )
    from langgraph.checkpoint.memory import MemorySaver

    assert isinstance(saver, MemorySaver)


def test_postgres_is_preferred_when_configured(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        agent_graph, "_postgres_checkpointer", lambda dsn, use_async: sentinel
    )
    saver = build_checkpointer(
        "redis://x", enabled=True, backend="postgres", redis_available=True,
        database_url="postgresql+asyncpg://u:p@h/db",
    )
    assert saver is sentinel


def test_redis_backend_falls_through_to_postgres(fake_redis, monkeypatch):
    """Configured backend first, then the other durable option — losing
    conversations to an in-process saver when a real one is available would be
    the worse outcome."""
    fake_redis(PLAIN_REDIS)
    sentinel = object()
    monkeypatch.setattr(
        agent_graph, "_postgres_checkpointer", lambda dsn, use_async: sentinel
    )
    saver = build_checkpointer(
        "redis://x", enabled=True, backend="redis", redis_available=True,
        database_url="postgresql+asyncpg://u:p@h/db",
    )
    assert saver is sentinel


def test_memory_disabled_returns_nothing():
    assert build_checkpointer("redis://x", enabled=False) is None
