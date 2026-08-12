"""Shelves against a real database: ownership, quota, and the default shelf.

The unit tests cover the corpus *name*; these cover the rows that decide which
name a request gets. Three things need a database to state honestly:

- a shelf id from another account reaches nothing,
- the per-account chunk quota is not multiplied by owning several shelves,
- an account that predates shelves still has a working default one.
"""

from __future__ import annotations

import uuid

import pytest

from graphrag.api.routers.query import _owned_thread, _shelf_for
from graphrag.api.routers.shelves import list_shelves
from graphrag.db.engine import session_scope
from graphrag.db.models import File, Shelf, Thread, User
from graphrag.shelves import (
    MAX_SHELVES,
    ShelfRef,
    chunks_elsewhere,
    ensure_default,
    shelf_for_request,
    unique_slug,
)

from .conftest import requires_db

pytestmark = [pytest.mark.integration, requires_db]  # asyncio_mode = "auto"


class _User:
    """The AuthUser surface `shelf_for_request` reads."""

    def __init__(self, user_id: uuid.UUID) -> None:
        self.user_id = str(user_id)


async def _account(db, email: str = "a@example.com") -> uuid.UUID:
    async with session_scope(db) as s:
        user = User(
            email=email, password_hash="x", tenant_id=f"t-{uuid.uuid4().hex[:8]}",
            status="active",
        )
        s.add(user)
        await s.flush()
        return user.id


async def _shelf(db, owner: uuid.UUID, name: str, slug: str, preset="general") -> uuid.UUID:
    async with session_scope(db) as s:
        shelf = Shelf(user_id=owner, name=name, slug=slug, preset=preset)
        s.add(shelf)
        await s.flush()
        return shelf.id


async def _file(db, owner: uuid.UUID, shelf_id: uuid.UUID | None, chunks: int) -> None:
    async with session_scope(db) as s:
        s.add(
            File(
                id=uuid.uuid4().hex[:16], user_id=owner, shelf_id=shelf_id,
                name="d.pdf", path=f"data/uploads/{uuid.uuid4().hex[:8]}.pdf",
                chunks=chunks, status="ingested",
            )
        )


async def _thread(db, owner: uuid.UUID, shelf_id: uuid.UUID | None) -> uuid.UUID:
    async with session_scope(db) as s:
        thread = Thread(user_id=owner, shelf_id=shelf_id, title="t")
        s.add(thread)
        await s.flush()
        return thread.id


class _Req:
    """The QueryRequest surface `_shelf_for` reads."""

    def __init__(self, shelf_id: str | None) -> None:
        self.shelf_id = shelf_id


# --------------------------------------------------------------------------
# A conversation is pinned to its shelf
# --------------------------------------------------------------------------

async def test_an_existing_thread_overrides_the_requested_shelf(db):
    """A stale `shelf_id` — the picker moved after the conversation started —
    must not redirect the question. The thread's memory is keyed on its own
    shelf's corpus, so answering from another would cite documents the
    transcript above has never mentioned."""
    alice = await _account(db)
    maths = await _shelf(db, alice, "Maths", "maths")
    code = await _shelf(db, alice, "Code", "code")
    thread = await _thread(db, alice, maths)

    thread_id, thread_shelf = await _owned_thread(db, _User(alice), str(thread))
    chosen = await _shelf_for(db, _User(alice), _Req(str(code)), thread_id, thread_shelf)

    assert chosen.slug == "maths"


async def test_a_default_shelf_thread_is_pinned_too(db):
    """The regression this guards. A thread on the default shelf has
    `shelf_id IS NULL` — as does every thread predating shelves — so branching on
    the shelf rather than on the thread handed exactly those conversations back
    to whatever the request asked for. NULL means "the default shelf"."""
    alice = await _account(db)
    maths = await _shelf(db, alice, "Maths", "maths")
    thread = await _thread(db, alice, None)  # default shelf

    thread_id, thread_shelf = await _owned_thread(db, _User(alice), str(thread))
    chosen = await _shelf_for(db, _User(alice), _Req(str(maths)), thread_id, thread_shelf)

    assert chosen.is_default and chosen.slug == ""


async def test_the_request_decides_only_when_there_is_no_thread_yet(db):
    """The first turn of a new conversation is the one moment the client picks."""
    alice = await _account(db)
    maths = await _shelf(db, alice, "Maths", "maths")

    chosen = await _shelf_for(db, _User(alice), _Req(str(maths)), None, None)

    assert chosen.slug == "maths"


# --------------------------------------------------------------------------
# Ownership
# --------------------------------------------------------------------------

async def test_a_shelf_from_another_account_is_not_found(db):
    """404, not a fall back to the default shelf. Falling back would answer the
    question from a corpus the caller did not ask about, which reads as the
    assistant having lost their documents."""
    alice, bob = await _account(db, "alice@x.com"), await _account(db, "bob@x.com")
    bobs = await _shelf(db, bob, "Bob's maths", "maths")

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as caught:
        await shelf_for_request(db, _User(alice), str(bobs))
    assert caught.value.status_code == 404


@pytest.mark.parametrize("bad", ["not-a-uuid", str(uuid.uuid4()), ""])
async def test_unparseable_and_unknown_ids(db, bad):
    """An id that never existed and one belonging to someone else must be
    indistinguishable — otherwise the response enumerates real shelves."""
    alice = await _account(db)
    if bad == "":
        assert (await shelf_for_request(db, _User(alice), bad)).is_default
        return
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as caught:
        await shelf_for_request(db, _User(alice), bad)
    assert caught.value.status_code == 404


async def test_no_shelf_id_means_the_default_shelf(db):
    alice = await _account(db)
    ref = await shelf_for_request(db, _User(alice), None)
    assert ref.is_default and ref.slug == "" and ref.id is None


# --------------------------------------------------------------------------
# The default shelf
# --------------------------------------------------------------------------

async def test_an_account_with_no_rows_still_lists_a_default_shelf(db):
    """Accounts created before migration 0004 have no shelf row. The default one
    is implicit, so the picker never has to render an empty list — and the files
    they uploaded back then (`shelf_id IS NULL`) have to be counted under it."""
    alice = await _account(db)
    await _file(db, alice, None, chunks=5)

    listed = await list_shelves(user=_User(alice), db=db)

    assert listed.max_shelves == MAX_SHELVES
    assert len(listed.shelves) == 1
    only = listed.shelves[0]
    assert only.is_default and only.slug == "" and only.id is None
    assert only.files == 1


async def test_listing_counts_null_shelf_files_under_the_default(db):
    """Once a default row exists, pre-shelves uploads must still appear there
    rather than vanishing from every count."""
    alice = await _account(db)
    default = await ensure_default(db, alice)
    maths = await _shelf(db, alice, "Maths", "maths")
    await _file(db, alice, None, chunks=5)       # pre-shelves
    await _file(db, alice, default.id, chunks=5)  # explicitly on the default
    await _file(db, alice, maths, chunks=5)

    by_name = {s.name: s for s in (await list_shelves(user=_User(alice), db=db)).shelves}

    assert by_name["My documents"].files == 2
    assert by_name["Maths"].files == 1
    # Default first, so the picker opens on it.
    assert (await list_shelves(user=_User(alice), db=db)).shelves[0].is_default


async def test_ensure_default_is_idempotent(db):
    alice = await _account(db)
    first = await ensure_default(db, alice)
    second = await ensure_default(db, alice)
    assert first.id == second.id

    async with session_scope(db) as s:
        from sqlalchemy import func, select

        count = (
            await s.execute(
                select(func.count()).select_from(Shelf).where(Shelf.user_id == alice)
            )
        ).scalar_one()
    assert count == 1


async def test_slugs_are_unique_per_owner_but_not_globally(db):
    """Two people may both keep a "physics" shelf; the corpus is prefixed with
    the tenant, so their storage never meets."""
    alice, bob = await _account(db, "a@x.com"), await _account(db, "b@x.com")
    await _shelf(db, alice, "Physics", "physics")

    async with session_scope(db) as s:
        assert await unique_slug(s, alice, "Physics") == "physics-2"
        assert await unique_slug(s, bob, "Physics") == "physics"


async def test_a_name_that_sanitizes_to_nothing_still_gets_a_slug(db):
    alice = await _account(db)
    async with session_scope(db) as s:
        assert await unique_slug(s, alice, "数学") == "shelf"
        assert await unique_slug(s, alice, "***") == "shelf"


# --------------------------------------------------------------------------
# The chunk quota spans shelves
# --------------------------------------------------------------------------

async def test_other_shelves_count_against_the_same_allowance(db):
    """The quota is per account. `IngestPipeline._check_capacity` can only count
    the corpus it writes to, so without this a user with four shelves would get
    four times their chunks."""
    alice = await _account(db)
    maths = await _shelf(db, alice, "Maths", "maths")
    code = await _shelf(db, alice, "Code", "code")

    await _file(db, alice, maths, chunks=100)
    await _file(db, alice, code, chunks=250)

    maths_ref = ShelfRef(maths, "maths", "Maths", "general", False)
    code_ref = ShelfRef(code, "code", "Code", "general", False)

    assert await chunks_elsewhere(db, alice, maths_ref) == 250
    assert await chunks_elsewhere(db, alice, code_ref) == 100


async def test_null_shelf_files_belong_to_the_default_shelf(db):
    """A file uploaded before shelves existed has `shelf_id IS NULL` and lives in
    the default corpus. It must count as "here" on the default shelf and as
    "elsewhere" on any other — getting this backwards would stop a user's oldest
    documents counting against their quota at all."""
    alice = await _account(db)
    default = await ensure_default(db, alice)
    maths = await _shelf(db, alice, "Maths", "maths")

    await _file(db, alice, None, chunks=70)    # pre-shelves upload
    await _file(db, alice, maths, chunks=30)

    maths_ref = ShelfRef(maths, "maths", "Maths", "general", False)

    # On the default shelf, the NULL file is *here*, so only maths is elsewhere.
    assert await chunks_elsewhere(db, alice, default) == 30
    # On the maths shelf, the NULL file is elsewhere.
    assert await chunks_elsewhere(db, alice, maths_ref) == 70


async def test_an_implicit_default_shelf_still_excludes_only_other_shelves(db):
    """Same rule when the default shelf has no row at all (`id is None`)."""
    alice = await _account(db)
    maths = await _shelf(db, alice, "Maths", "maths")
    await _file(db, alice, None, chunks=70)
    await _file(db, alice, maths, chunks=30)

    assert await chunks_elsewhere(db, alice, ShelfRef.default()) == 30


async def test_another_account_never_counts_against_your_quota(db):
    alice, bob = await _account(db, "a@x.com"), await _account(db, "b@x.com")
    bobs = await _shelf(db, bob, "Bob", "bob")
    await _file(db, bob, bobs, chunks=9_999)

    assert await chunks_elsewhere(db, alice, ShelfRef.default()) == 0
