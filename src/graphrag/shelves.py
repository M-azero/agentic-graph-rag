"""Resolving a request to one shelf.

Every routine here answers the same question from a different direction: which
shelf is this request talking about, and is it really the caller's? Routers must
go through one of them rather than reading a `shelf_id` off the request — an
unchecked id is a pointer into another account's knowledge base, and the corpus
it resolves to is passed straight to the store.

The default shelf is the load-bearing case. A request that names no shelf, an
account created before shelves existed, a file uploaded last month, a dev-mode
identity with no account row at all — every one of those resolves to a shelf
whose slug is empty and whose corpus is therefore the bare tenant id, which is
where that data already is. So "no shelf" is never an error and never needs a
row to exist.
"""

from __future__ import annotations

import uuid

from fastapi import HTTPException
from sqlalchemy import func, select

from graphrag.container import sanitize_slug
from graphrag.db.engine import session_scope
from graphrag.db.models import File, Shelf

# The name a default shelf is created with. Matches migration 0004's backfill:
# an account created before the migration and one created after it should be
# indistinguishable.
DEFAULT_SHELF_NAME = "My documents"

# Ceiling on shelves per account. Each one is a separate corpus — its own graph
# constraints, its own community summaries, and under the DuckDB provider its
# own file and its own open handle — so this is a real resource, not a row
# count. Generous for the intended use (a shelf per subject) and far below the
# point where a script could exhaust file descriptors by creating them in a loop.
MAX_SHELVES = 24


class ShelfRef:
    """A resolved shelf: what the storage layer needs, plus what to show.

    Deliberately not the ORM object. Callers need `slug` (to reach the corpus)
    and `id` (to stamp on a file or thread) after the session has closed, and
    handing back a detached ORM instance is how a lazy attribute access turns
    into a `MissingGreenlet` three layers away.
    """

    __slots__ = ("id", "name", "preset", "slug", "is_default")

    def __init__(
        self,
        id: uuid.UUID | None,
        slug: str,
        name: str = DEFAULT_SHELF_NAME,
        preset: str = "general",
        is_default: bool = True,
    ) -> None:
        self.id = id
        self.slug = slug
        self.name = name
        self.preset = preset
        self.is_default = is_default

    @classmethod
    def from_row(cls, row: Shelf) -> ShelfRef:
        return cls(row.id, row.slug, row.name, row.preset, row.is_default)

    @classmethod
    def default(cls) -> ShelfRef:
        """The implicit shelf — no row required. Its empty slug is what makes
        `container.corpus_for` return the bare tenant id."""
        return cls(None, "")


def account_uuid(user) -> uuid.UUID | None:
    """The account row's id, or None for a dev-mode namespace identity."""
    try:
        return uuid.UUID(str(user.user_id))
    except (ValueError, AttributeError, TypeError):
        return None


async def shelf_for_request(db, user, shelf_id: str | None) -> ShelfRef:
    """The shelf this request means, verified to belong to `user`.

    A `shelf_id` that is unparseable, unknown, or owned by someone else is a
    404 rather than a fall back to the default. Falling back would be worse than
    unhelpful: the caller asked about their maths shelf, and silently answering
    from a different corpus reads as the assistant having lost the documents.
    404 (not 403) so an id from elsewhere is indistinguishable from one that
    never existed.
    """
    if not shelf_id:
        return ShelfRef.default()
    owner = account_uuid(user)
    if db is None or owner is None:
        # No account rows to check against, so no shelf can be verified as the
        # caller's. Dev mode gets the default shelf and nothing else.
        return ShelfRef.default()
    try:
        key = uuid.UUID(shelf_id)
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=404, detail="No such shelf.") from None

    async with session_scope(db) as s:
        row = (
            await s.execute(
                select(Shelf).where(Shelf.id == key, Shelf.user_id == owner)
            )
        ).scalar_one_or_none()
        if row is None:
            raise HTTPException(status_code=404, detail="No such shelf.")
        return ShelfRef.from_row(row)


async def shelf_by_id(db, owner: uuid.UUID | None, shelf_id: uuid.UUID | None) -> ShelfRef:
    """The shelf a stored row points at, for reads that already trust the row.

    Used where the `shelf_id` came from a file or thread this user was already
    proven to own, so there is nothing left to authorize — unlike
    `shelf_for_request`, which takes an id straight off the wire. A dangling id
    resolves to the default shelf, which is where a deleted shelf's leftovers
    belong.
    """
    if db is None or owner is None or shelf_id is None:
        return ShelfRef.default()
    async with session_scope(db) as s:
        row = (
            await s.execute(
                select(Shelf).where(Shelf.id == shelf_id, Shelf.user_id == owner)
            )
        ).scalar_one_or_none()
    return ShelfRef.from_row(row) if row is not None else ShelfRef.default()


async def ensure_default(db, owner: uuid.UUID | None) -> ShelfRef:
    """The user's default shelf row, created if it is missing.

    Called when a shelf is first created, so the picker never shows a new named
    shelf next to nothing. Idempotent under the partial unique index: two
    concurrent callers race, one loses on the constraint, and the loser re-reads
    the winner's row rather than failing the request.
    """
    if db is None or owner is None:
        return ShelfRef.default()
    async with session_scope(db) as s:
        row = (
            await s.execute(
                select(Shelf).where(Shelf.user_id == owner, Shelf.is_default.is_(True))
            )
        ).scalar_one_or_none()
        if row is not None:
            return ShelfRef.from_row(row)
    try:
        async with session_scope(db) as s:
            created = Shelf(
                user_id=owner, name=DEFAULT_SHELF_NAME, slug="",
                preset="general", is_default=True,
            )
            s.add(created)
            await s.flush()
            return ShelfRef.from_row(created)
    except Exception:
        # Lost the race, or the row appeared between the two statements above.
        async with session_scope(db) as s:
            row = (
                await s.execute(
                    select(Shelf).where(
                        Shelf.user_id == owner, Shelf.is_default.is_(True)
                    )
                )
            ).scalar_one_or_none()
        return ShelfRef.from_row(row) if row is not None else ShelfRef.default()


async def unique_slug(s, owner: uuid.UUID, name: str) -> str:
    """A storage-safe slug for `name`, not already used by this owner.

    Derived from the name for legibility — a `data/vectors/` listing and a Neo4j
    `corpus` value are both things an operator reads — but it is only a starting
    point. The slug is permanent and the name is not, so nothing may depend on
    the two continuing to correspond.

    A name that sanitizes to nothing at all ("数学", "***") still needs a slug,
    hence the `shelf` fallback. Collisions get a numeric suffix; the unique
    constraint is the real guarantee, this just avoids provoking it.
    """
    base = sanitize_slug(name) or "shelf"
    taken = set(
        (
            await s.execute(select(Shelf.slug).where(Shelf.user_id == owner))
        ).scalars().all()
    )
    if base not in taken:
        return base
    for n in range(2, 100):
        candidate = f"{base[:28]}-{n}"
        if candidate not in taken:
            return candidate
    return f"s-{uuid.uuid4().hex[:10]}"


async def chunks_elsewhere(db, owner: uuid.UUID | None, shelf: ShelfRef) -> int:
    """Chunks this user is storing on shelves *other than* `shelf`.

    The chunk quota is per account, but `IngestPipeline._check_capacity` can only
    count the corpus it is writing to. Left alone, a user with four shelves would
    get four times their allowance. Subtracting this from `max_chunks` before the
    ingest starts restores the per-account ceiling, using the mechanism the
    pipeline already has rather than teaching it about shelves.

    Reads `files.chunks`, stamped when an ingest finishes
    (`ingestion.status.finalize_file`) — the same figure the admin console and
    the account page report, so nobody is refused for storage no screen shows.
    """
    if db is None or owner is None:
        return 0

    # A NULL shelf_id *is* the default shelf, so it counts as "here" on the
    # default shelf and as "elsewhere" on any other. Getting this backwards is
    # how a user's oldest documents would stop counting against their quota.
    if shelf.is_default:
        elsewhere = File.shelf_id.is_not(None)
        if shelf.id is not None:
            elsewhere = elsewhere & (File.shelf_id != shelf.id)
    else:
        elsewhere = File.shelf_id.is_(None) | (File.shelf_id != shelf.id)

    async with session_scope(db) as s:
        total = (
            await s.execute(
                select(func.coalesce(func.sum(File.chunks), 0)).where(
                    File.user_id == owner, elsewhere
                )
            )
        ).scalar_one()
    return int(total or 0)


__all__ = [
    "DEFAULT_SHELF_NAME",
    "MAX_SHELVES",
    "ShelfRef",
    "account_uuid",
    "chunks_elsewhere",
    "ensure_default",
    "shelf_by_id",
    "shelf_for_request",
    "unique_slug",
]
