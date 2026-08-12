"""Shelves: one account, several separate knowledge bases.

Uploading a maths textbook and a programming manual into one corpus does not
just mix the search results — entity resolution merges the "function" of each
into a single node and graph traversal walks from integration into closures. A
shelf gives each subject its own corpus, which is the isolation boundary the
storage layer already enforces per tenant.

Deleting a shelf is the one destructive operation here, and it is the reason
`DELETE` does more than remove a row: the row is a *name* for a corpus and a
DuckDB file, so dropping it alone would leave that storage live and unreachable
forever. See `delete_shelf`.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete, func, select

from graphrag.agent.presets import canonical_preset, preset_options
from graphrag.api.deps import AuthUser, get_container, get_current_user, get_db
from graphrag.api.schemas import (
    PresetOption,
    ShelfCreate,
    ShelfDeleted,
    ShelfInfo,
    ShelfList,
    ShelfUpdate,
)
from graphrag.container import Container
from graphrag.core.logging import get_logger
from graphrag.db.engine import session_scope
from graphrag.db.models import File, Shelf
from graphrag.shelves import (
    MAX_SHELVES,
    ShelfRef,
    account_uuid,
    ensure_default,
    unique_slug,
)

router = APIRouter(prefix="/shelves", tags=["shelves"])
log = get_logger(__name__)


def _require_db(db):
    if db is None:
        raise HTTPException(
            status_code=503,
            detail="Shelves need a database. Set GRAPHRAG_DATABASE_URL.",
        )
    return db


def _require_account(user: AuthUser) -> uuid.UUID:
    owner = account_uuid(user)
    if owner is None:
        raise HTTPException(
            status_code=400, detail="Shelves require a real account."
        )
    return owner


def _shape(row: Shelf, files: int = 0) -> ShelfInfo:
    return ShelfInfo(
        id=str(row.id),
        name=row.name,
        slug=row.slug,
        # Clamped on the way out as well as on the way in: a preset removed in a
        # later release would otherwise leave the picker with a value it has no
        # option for, which renders as an empty select.
        preset=str(canonical_preset(row.preset)),
        is_default=row.is_default,
        files=files,
    )


def _shape_ref(ref: ShelfRef, files: int = 0) -> ShelfInfo:
    return ShelfInfo(
        id=str(ref.id) if ref.id else None,
        name=ref.name,
        slug=ref.slug,
        preset=str(canonical_preset(ref.preset)),
        is_default=ref.is_default,
        files=files,
    )


async def _file_counts(s, owner: uuid.UUID) -> dict[uuid.UUID | None, int]:
    """Documents per shelf, with NULL folded in as the default shelf's."""
    rows = await s.execute(
        select(File.shelf_id, func.count())
        .where(File.user_id == owner)
        .group_by(File.shelf_id)
    )
    return {shelf_id: int(n) for shelf_id, n in rows}


@router.get("/presets", response_model=list[PresetOption])
async def list_presets() -> list[PresetOption]:
    """The job presets this build offers.

    Declared before `/{shelf_id}` so the literal path wins the match, and served
    from `agent.presets` so the UI never carries its own copy of the list —
    adding a preset there is all it takes to offer it here.
    """
    return [
        PresetOption(
            id=str(p.id), label=p.label, emoji=p.emoji, description=p.description
        )
        for p in preset_options()
    ]


@router.get("", response_model=ShelfList)
async def list_shelves(
    user: AuthUser = Depends(get_current_user),
    db=Depends(get_db),
) -> ShelfList:
    """Every shelf, default first. An account with no rows yet still gets its
    default shelf back — that shelf is implicit, so the picker never has to
    render an empty list."""
    owner = account_uuid(user)
    if db is None or owner is None:
        return ShelfList(shelves=[_shape_ref(ShelfRef.default())], max_shelves=MAX_SHELVES)

    async with session_scope(db) as s:
        rows = (
            await s.execute(
                select(Shelf)
                .where(Shelf.user_id == owner)
                .order_by(Shelf.is_default.desc(), Shelf.created_at)
            )
        ).scalars().all()
        counts = await _file_counts(s, owner)

    if not rows:
        # No rows, so every file this user has is on the implicit default shelf.
        return ShelfList(
            shelves=[_shape_ref(ShelfRef.default(), sum(counts.values()))],
            max_shelves=MAX_SHELVES,
        )

    shelves = []
    for row in rows:
        files = counts.get(row.id, 0)
        if row.is_default:
            files += counts.get(None, 0)  # NULL shelf_id is the default shelf
        shelves.append(_shape(row, files))
    return ShelfList(shelves=shelves, max_shelves=MAX_SHELVES)


@router.post("", response_model=ShelfInfo, status_code=201)
async def create_shelf(
    payload: ShelfCreate,
    user: AuthUser = Depends(get_current_user),
    db=Depends(get_db),
) -> ShelfInfo:
    """Add a shelf. Its slug — and therefore its corpus — is derived here and
    then fixed for the shelf's lifetime."""
    owner = _require_account(user)
    _require_db(db)

    # So the picker shows "My documents" alongside the new shelf rather than the
    # new shelf alone; an account that predates shelves has no default row yet.
    await ensure_default(db, owner)

    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="A shelf needs a name.")

    async with session_scope(db) as s:
        count = (
            await s.execute(
                select(func.count()).select_from(Shelf).where(Shelf.user_id == owner)
            )
        ).scalar_one()
        if count >= MAX_SHELVES:
            raise HTTPException(
                status_code=400,
                detail=f"You can have at most {MAX_SHELVES} shelves. "
                       "Delete one to make room.",
            )
        shelf = Shelf(
            user_id=owner,
            name=name[:80],
            slug=await unique_slug(s, owner, name),
            preset=str(canonical_preset(payload.preset)),
            is_default=False,
        )
        s.add(shelf)
        await s.flush()
        log.info("shelf_created", user=str(owner), slug=shelf.slug)
        return _shape(shelf)


@router.patch("/{shelf_id}", response_model=ShelfInfo)
async def update_shelf(
    shelf_id: str,
    payload: ShelfUpdate,
    user: AuthUser = Depends(get_current_user),
    db=Depends(get_db),
) -> ShelfInfo:
    """Rename a shelf, or change the preset it opens with. The slug is not
    touched — see `ShelfUpdate`."""
    owner = _require_account(user)
    _require_db(db)
    async with session_scope(db) as s:
        shelf = await _owned(s, shelf_id, owner)
        if payload.name is not None:
            name = payload.name.strip()
            if not name:
                raise HTTPException(status_code=400, detail="A shelf needs a name.")
            shelf.name = name[:80]
        if payload.preset is not None:
            shelf.preset = str(canonical_preset(payload.preset))
        return _shape(shelf)


@router.delete("/{shelf_id}", response_model=ShelfDeleted)
async def delete_shelf(
    shelf_id: str,
    user: AuthUser = Depends(get_current_user),
    container: Container = Depends(get_container),
    db=Depends(get_db),
) -> ShelfDeleted:
    """Delete a shelf and everything on it.

    Order matters. The corpus is purged *before* the row, because the row is the
    only record of the slug the corpus is named after — drop it first and the
    Neo4j nodes and the DuckDB file survive with nothing left that can address
    them. If the purge fails the row stays and the delete can be retried, which
    is the same trade `accounts.purge` makes for the same reason.

    The default shelf cannot be deleted: its corpus is the bare tenant id, so
    "delete this shelf" would mean "delete everything ingested before shelves
    existed" — that is what purging an account is for, not this.
    """
    owner = _require_account(user)
    _require_db(db)

    async with session_scope(db) as s:
        shelf = await _owned(s, shelf_id, owner)
        if shelf.is_default:
            raise HTTPException(
                status_code=400,
                detail="The default shelf can't be deleted. Delete its documents instead.",
            )
        slug, key = shelf.slug, shelf.id
        paths = (
            await s.execute(select(File.path).where(File.shelf_id == key))
        ).scalars().all()

    removed = _purge_shelf_storage(container, user.tenant_id, slug)

    async with session_scope(db) as s:
        # The uploads go with the shelf. `File.shelf_id` is ON DELETE SET NULL,
        # which is right when a shelf disappears for some other reason — but here
        # the documents were on the shelf being deleted, and leaving them would
        # silently reassign a maths textbook to the default shelf while its
        # chunks no longer exist anywhere.
        await s.execute(delete(File).where(File.shelf_id == key, File.user_id == owner))
        await s.execute(delete(Shelf).where(Shelf.id == key, Shelf.user_id == owner))

    files_removed = 0
    for path in paths:
        try:
            Path(path).unlink(missing_ok=True)
            files_removed += 1
        except OSError as exc:
            log.warning("shelf_file_unlink_failed", path=path, error=str(exc))

    log.info(
        "shelf_deleted", user=str(owner), slug=slug,
        chunks=removed, files=files_removed,
    )
    return ShelfDeleted(id=shelf_id, chunks_removed=removed, files_removed=files_removed)


def _purge_shelf_storage(container: Container, tenant_id: str, slug: str) -> int:
    """Drop the shelf's Neo4j corpus and its DuckDB file, best effort.

    Built from the store factory rather than `container.tenant()`: that would
    construct the embedder, reranker and agent as well, so deleting a shelf would
    load models and stall whenever an embedding provider happened to be down.
    """
    from graphrag.storage import build_graph_store

    database, corpus = container._resolve_scope(tenant_id, slug)
    removed = 0
    try:
        store = build_graph_store(container.driver, database, corpus, container.settings)
        removed = store.purge_corpus()
    except Exception as exc:
        log.warning("shelf_graph_purge_failed", corpus=corpus, error=str(exc))

    # Evict first: the tenant holds the open DuckDB handle, and Windows refuses
    # to unlink a file that is still open.
    container._tenants.pop(corpus, None)
    cfg = container.settings.storage.vector
    if cfg.provider == "duckdb":
        try:
            from graphrag.storage.vector.duckdb_store import close_file

            path = Path(cfg.duckdb_dir) / database / f"{corpus}.duckdb"
            close_file(path)
            path.unlink(missing_ok=True)
            Path(str(path) + ".wal").unlink(missing_ok=True)
        except OSError as exc:
            log.warning("shelf_vectors_purge_failed", corpus=corpus, error=str(exc))
    return removed


async def _owned(s, shelf_id: str, owner: uuid.UUID) -> Shelf:
    """The caller's shelf, or 404 — never 403, so an id from another account is
    indistinguishable from one that never existed."""
    try:
        key = uuid.UUID(shelf_id)
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=404, detail="No such shelf.") from None
    shelf = (
        await s.execute(select(Shelf).where(Shelf.id == key, Shelf.user_id == owner))
    ).scalar_one_or_none()
    if shelf is None:
        raise HTTPException(status_code=404, detail="No such shelf.")
    return shelf


