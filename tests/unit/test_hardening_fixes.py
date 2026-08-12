"""The small ones. Each was a real defect, none was severe on its own.

Grouped because they share nothing but size: a purge that missed a file, a
cache that could fail the thing it accelerates, an unbounded dict, a
returned-id race, and a warning that did not fire for the password the stack
actually ships with.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from graphrag.config.settings import (
    Settings,
    StorageCfg,
    TenancyCfg,
    VectorStoreCfg,
)

# -- purge: the DuckDB file under per-tenant databases ------------------------

def _container(per_tenant: bool, tmp_path: Path):
    from graphrag.container import Container

    settings = Settings(
        storage=StorageCfg(
            vector=VectorStoreCfg(provider="duckdb", duckdb_dir=str(tmp_path))
        ),
        tenancy=TenancyCfg(per_tenant_database=per_tenant),
    )
    from graphrag.config.settings import Secrets

    return Container(settings, Secrets())


def _seed(root: Path, database: str, tenant: str) -> Path:
    path = root / database / f"{tenant}.duckdb"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"vectors")
    return path


def test_purge_removes_the_vector_file_in_the_shared_database(tmp_path):
    from graphrag.accounts.purge import _drop_vector_store

    container = _container(False, tmp_path)
    path = _seed(tmp_path, "neo4j", "alice-1234")

    assert _drop_vector_store(container, "alice-1234") is True
    assert not path.exists()


def test_purge_removes_the_vector_file_under_a_per_tenant_database(tmp_path):
    """The bug: the path was built from `storage.graph.database`, but the store
    was created beneath `_resolve_scope`'s `u_<tenant>`. The purge reported
    `vectors_removed: False` with no error and left the data on disk."""
    from graphrag.accounts.purge import _drop_vector_store

    container = _container(True, tmp_path)
    database, _corpus = container._resolve_scope("alice-1234")
    path = _seed(tmp_path, database, "alice-1234")
    assert path.exists()

    assert _drop_vector_store(container, "alice-1234") is True
    assert not path.exists()


def test_purge_reports_false_when_there_was_nothing_to_remove(tmp_path):
    from graphrag.accounts.purge import _drop_vector_store

    assert _drop_vector_store(_container(False, tmp_path), "never-existed") is False


# -- the embedding cache must not fail an ingest ------------------------------

class _Inner:
    def __init__(self) -> None:
        self.dim = 3
        self.doc_calls: list[list[str]] = []

    def embed_documents(self, texts):
        self.doc_calls.append(list(texts))
        return [[float(len(t)), 0.0, 0.0] for t in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]


class _BrokenRedis:
    def mget(self, keys):
        raise ConnectionError("redis went away")

    def pipeline(self):
        raise ConnectionError("redis went away")

    def get(self, key):
        raise ConnectionError("redis went away")

    def setex(self, *a):
        raise ConnectionError("redis went away")


def test_a_dead_cache_does_not_fail_the_embedding():
    """An ingest used to die here — on a document the user had already
    uploaded — because a cache read was unguarded."""
    from graphrag.embeddings.cache import CachedEmbedder

    inner = _Inner()
    cached = CachedEmbedder(inner, _BrokenRedis(), "m", 60)

    assert cached.embed_documents(["a", "bb"]) == [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    assert cached.embed_query("q") == [1.0, 0.0, 0.0]
    assert inner.doc_calls == [["a", "bb"]]


def test_a_corrupt_cache_entry_is_a_miss():
    from graphrag.embeddings.cache import CachedEmbedder

    class _Garbage(_BrokenRedis):
        def mget(self, keys):
            return ["not json"] * len(keys)

        def pipeline(self):
            return SimpleNamespace(setex=lambda *a: None, execute=lambda: None)

    inner = _Inner()
    cached = CachedEmbedder(inner, _Garbage(), "m", 60)
    assert cached.embed_documents(["a"]) == [[1.0, 0.0, 0.0]]


def test_every_input_gets_exactly_one_vector():
    """Misalignment here pairs a chunk with another chunk's embedding, which is
    silent and poisons retrieval rather than raising."""
    from graphrag.embeddings.cache import CachedEmbedder

    stored: dict[str, str] = {}

    class _Half:
        """Caches only the first text, so the second run is a partial hit."""

        def mget(self, keys):
            return [stored.get(k) for k in keys]

        def pipeline(self):
            calls: list[tuple] = []
            return SimpleNamespace(
                setex=lambda k, _ttl, v: calls.append((k, v)),
                execute=lambda: stored.update(dict(calls[:1])),
            )

    inner = _Inner()
    cached = CachedEmbedder(inner, _Half(), "m", 60)
    first = cached.embed_documents(["a", "bb", "ccc"])
    second = cached.embed_documents(["a", "bb", "ccc"])
    assert first == second
    assert len(second) == 3


# -- the in-process job store is bounded --------------------------------------

def test_the_memory_job_store_evicts_instead_of_growing():
    from graphrag.jobs import _MEM_MAX, JobStatus, JobStore

    store = JobStore(None)
    for i in range(_MEM_MAX + 50):
        store.set(JobStatus(f"job-{i}", status="done", owner="t"))

    assert len(store._mem) == _MEM_MAX
    assert store.get("job-0", owner="t") is None          # oldest evicted
    assert store.get(f"job-{_MEM_MAX + 49}", owner="t") is not None


def test_eviction_does_not_weaken_the_ownership_check():
    from graphrag.jobs import JobStatus, JobStore

    store = JobStore(None)
    store.set(JobStatus("j1", status="done", owner="tenant-a"))
    assert store.get("j1", owner="tenant-b") is None
    assert store.get("j1", owner="tenant-a") is not None


# -- the weak-password warning covers what ships ------------------------------

@pytest.mark.parametrize(
    "password", ["please-change-me", "", "12345678", "change-me"]
)
def test_shipped_default_passwords_are_recognised_as_weak(password):
    """`12345678` is what docker-compose.yml substitutes into NEO4J_AUTH when
    the operator never set one — the exact case the warning existed for, and
    the one it did not match."""
    from graphrag.api.app import _WEAK_PASSWORDS

    assert password in _WEAK_PASSWORDS


def test_a_real_password_is_not_flagged():
    from graphrag.api.app import _WEAK_PASSWORDS

    assert "Tz9!qv3Lm2wPx7Kd" not in _WEAK_PASSWORDS
