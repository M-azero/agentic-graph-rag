"""Deleting a user, everywhere they exist.

A user's data is spread across four stores — Postgres rows, a Neo4j corpus, a
DuckDB file, and uploaded files on disk — and "delete my account" has to mean
all of them. Each step is independent and best-effort: a Neo4j outage must not
leave the account half-deleted with no way to finish, so failures are collected
and reported rather than aborting the rest.

Postgres goes last. Its cascades are what make the account disappear, and while
those rows still exist the purge can be retried; once they're gone the tenant
id needed to find the other stores is gone too.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import delete, select

from graphrag.core.logging import get_logger
from graphrag.db.engine import session_scope
from graphrag.db.models import File, Shelf, User

log = get_logger(__name__)


@dataclass
class PurgeReport:
    tenant_id: str = ""
    graph_nodes: int = 0
    files_removed: int = 0
    vectors_removed: bool = False
    rows_removed: bool = False
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "tenant_id": self.tenant_id,
            "graph_nodes": self.graph_nodes,
            "files_removed": self.files_removed,
            "vectors_removed": self.vectors_removed,
            "rows_removed": self.rows_removed,
            "errors": self.errors,
        }


async def purge_user(db, container, user_id: str, *, keep_account: bool = False) -> PurgeReport:
    """Remove a user's data. With `keep_account`, wipe their content but leave
    the login intact — the "reset this user" the admin panel offers."""
    report = PurgeReport()
    try:
        owner = uuid.UUID(str(user_id))
    except (ValueError, AttributeError, TypeError):
        report.errors.append("not an account id")
        return report

    async with session_scope(db) as s:
        user = (await s.execute(select(User).where(User.id == owner))).scalar_one_or_none()
        if user is None:
            report.errors.append("no such user")
            return report
        tenant_id = user.tenant_id
        paths = [
            row.path
            for row in (
                await s.execute(select(File).where(File.user_id == owner))
            ).scalars().all()
        ]
        # Every shelf, plus the default one. The default's slug is "" and it may
        # have no row at all (an account that predates migration 0004), so it is
        # added unconditionally rather than read — and a set keeps the two from
        # purging the same corpus twice.
        slugs = set(
            (
                await s.execute(select(Shelf.slug).where(Shelf.user_id == owner))
            ).scalars().all()
        )
        slugs.add("")
    report.tenant_id = tenant_id

    # Uploaded files on disk.
    for path in paths:
        try:
            Path(path).unlink(missing_ok=True)
            report.files_removed += 1
        except OSError as exc:
            report.errors.append(f"file {path}: {exc}")

    # The graph and vectors of every shelf. Built from the store factories
    # directly rather than via `container.tenant()`: that would construct the
    # embedder, reranker and agent too, so deleting a user's data would fail
    # whenever an embedding provider happened to be unreachable.
    #
    # Each shelf is its own corpus and its own DuckDB file, so each needs its
    # own purge — and each is independently best-effort. One shelf failing must
    # not strand the rest, which is the same reason the four stores below are
    # separate steps rather than one transaction.
    for slug in sorted(slugs):
        try:
            from graphrag.storage import build_graph_store

            database, corpus = container._resolve_scope(tenant_id, slug)
            graph_store = build_graph_store(
                container.driver, database, corpus, container.settings
            )
            report.graph_nodes += graph_store.purge_corpus()
        except Exception as exc:
            log.warning(
                "purge_graph_failed", tenant=tenant_id, shelf=slug, error=str(exc)
            )
            report.errors.append(f"graph[{slug or 'default'}]: {exc}")

        try:
            # `or` rather than `and`: any shelf whose file was removed counts,
            # and a shelf that never had one must not clear the flag.
            report.vectors_removed = (
                _drop_vector_store(container, tenant_id, slug) or report.vectors_removed
            )
        except Exception as exc:
            log.warning(
                "purge_vectors_failed", tenant=tenant_id, shelf=slug, error=str(exc)
            )
            report.errors.append(f"vectors[{slug or 'default'}]: {exc}")

    # Evict every cached shelf so a later request rebuilds from nothing.
    container.evict_tenant(tenant_id)

    async with session_scope(db) as s:
        if keep_account:
            await s.execute(delete(File).where(File.user_id == owner))
        else:
            # Cascades take threads, messages, sessions, keys, usage and limits.
            await s.execute(delete(User).where(User.id == owner))
        report.rows_removed = True

    log.info(
        "user_purged", user=str(owner), tenant=tenant_id,
        kept_account=keep_account, errors=len(report.errors),
    )
    return report


def _drop_vector_store(container, tenant_id: str, shelf: str = "") -> bool:
    """Delete one shelf's vector data. For DuckDB that is a file, and the
    handle has to be closed first or Windows refuses to unlink it."""
    cfg = container.settings.storage.vector
    if cfg.provider == "neo4j":
        # Vectors live on the chunk nodes the graph purge already removed.
        return True
    if cfg.provider != "duckdb":
        return False  # `local` npz files are cleaned up with the data directory

    from graphrag.storage.vector.duckdb_store import close_file

    # `_resolve_scope`, not `storage.graph.database`: under
    # `tenancy.per_tenant_database` the store was built beneath `u_<tenant>/`,
    # so the flat name pointed at a path that never existed — the purge reported
    # `vectors_removed: False` with no error and the deleted user's vectors
    # stayed on disk. The corpus it returns is also what names the file, which
    # is what makes this correct per shelf rather than only for the default one.
    database, corpus = container._resolve_scope(tenant_id, shelf)
    path = Path(cfg.duckdb_dir) / database / f"{corpus}.duckdb"
    close_file(path)
    existed = path.exists()
    path.unlink(missing_ok=True)
    # DuckDB may leave a write-ahead log beside the database.
    Path(str(path) + ".wal").unlink(missing_ok=True)
    return existed
