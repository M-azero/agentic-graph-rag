"""Add documents to a user's knowledge base.

Ingestion runs as a background task so a large upload never blocks the request.
Where it runs depends on deployment: with GRAPHRAG_USE_WORKER it goes to an Arq
worker in a separately resource-limited container; otherwise it runs in-process,
which is what the duckdb vector provider requires (one process must own each
tenant's database file). Status is persisted in Redis and polled by job id. All
ingestion is scoped to the current user."""

from __future__ import annotations

import asyncio
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, UploadFile
from sqlalchemy import delete, func, select, text

from graphrag.api.deps import (
    AuthUser,
    get_container,
    get_current_user,
    get_db,
    get_job_store,
    require_admin_user,
)
from graphrag.api.schemas import (
    DeleteResponse,
    FileList,
    IngestResponse,
    IngestStatus,
    StoredFile,
)
from graphrag.container import Container
from graphrag.core.logging import get_logger
from graphrag.core.net import BlockedURLError, open_public_url
from graphrag.core.redact import safe_detail
from graphrag.db.engine import session_scope
from graphrag.db.models import File
from graphrag.ingestion.status import finalize_file
from graphrag.jobs import JobStatus, JobStore
from graphrag.limits import effective_limits, reject_with
from graphrag.limits.service import LimitBreach, Limits
from graphrag.pipelines import IngestPipeline
from graphrag.shelves import ShelfRef, chunks_elsewhere, shelf_by_id, shelf_for_request
from graphrag.usage import INGEST_CHUNKS, UPLOAD

router = APIRouter(tags=["ingest"])
log = get_logger(__name__)

# Server-side ingest is confined to this tree. Without the fence, any caller
# could ingest an arbitrary server file (.env included) and read it back
# through /search.
_DATA_ROOT = Path("data")
_UPLOAD_DIR = _DATA_ROOT / "uploads"
_DOWNLOAD_DIR = _DATA_ROOT / "downloads"

_URL_SUFFIX = {
    "application/pdf": ".pdf",
    "text/html": ".html",
    "text/markdown": ".md",
    "text/plain": ".txt",
    "text/csv": ".csv",
}


# Only one ingest at a time in-process. Extraction fires a burst of concurrent
# LLM calls per document; two documents at once would double that against the
# same two vCPUs the chat stream is using. Queued jobs simply wait, and the
# client already polls for status.
_INGEST_SLOT = asyncio.Semaphore(1)


def _inproc_ingest(
    container: Container, store: JobStore, job_id: str, path: str, user_id,
    max_chunks: int | None = None, shelf: str | None = None,
) -> None:
    """Run the pipeline and record the outcome. Blocking — call it off the loop."""
    store.set(JobStatus(job_id, status="running", owner=user_id))
    try:
        stats = IngestPipeline(container, max_chunks=max_chunks).run(
            path, user_id=user_id, shelf=shelf
        )
        # "partial" when the text is searchable but the knowledge graph is not:
        # extraction failed for at least one chunk. Reporting that as "done"
        # hands the user an empty graph and no reason for it.
        detail = ""
        if stats.partial:
            detail = (
                f"Stored and searchable, but entity extraction failed for "
                f"{stats.extraction_failures} of {stats.chunks} chunks — the "
                "knowledge graph is incomplete. Check the model provider, then "
                "re-upload to fill it in."
            )
            log.warning(
                "ingest_partial", job=job_id,
                failures=stats.extraction_failures, chunks=stats.chunks,
            )
        store.set(
            JobStatus(job_id, status="partial" if stats.partial else "done",
                      detail=detail,
                      documents=stats.documents, chunks=stats.chunks,
                      entities=stats.entities, relations=stats.relations,
                      owner=user_id)
        )
    except Exception as exc:
        # Full error to the log, a scrubbed one to the client: provider SDKs put
        # the request URL in the message, and for some providers the key rides
        # in that URL as a query parameter.
        log.warning("ingest_job_failed", job=job_id, error=str(exc))
        store.set(JobStatus(job_id, status="error", detail=safe_detail(exc), owner=user_id))


async def _run_ingest(
    container: Container, store: JobStore, job_id: str, path: str, user_id,
    max_chunks: int | None = None, db=None, file_id: str | None = None,
    recorder=None, account_id: str | None = None, shelf: str | None = None,
):
    """One queued ingest, off the event loop so streaming stays responsive."""
    async with _INGEST_SLOT:
        await asyncio.to_thread(
            _inproc_ingest, container, store, job_id, path, user_id, max_chunks, shelf
        )
    status = store.get(job_id)
    await finalize_file(db, file_id, status)
    await _record_chunks(recorder, account_id, status)


async def _record_chunks(recorder, account_id: str | None, status: JobStatus | None) -> None:
    """Book the chunks an ingest produced against the durable usage log.

    Separate from the token counters on purpose: ingest is bounded by the file
    and chunk quotas rather than the token budget, but the admin charts still
    need to see that the work happened.
    """
    if recorder is None or not account_id or status is None:
        return
    chunks = getattr(status, "chunks", 0) or 0
    if chunks > 0:
        await recorder.record(account_id, INGEST_CHUNKS, chunks, {"job": status.job_id})


async def _enqueue(
    request: Request, background: BackgroundTasks, container: Container,
    store: JobStore, path: str, user_id,
    max_chunks: int | None = None, db=None, file_id: str | None = None,
    account_id: str | None = None, shelf: str | None = None,
) -> IngestResponse:
    job_id = uuid.uuid4().hex[:12]
    store.set(JobStatus(job_id, status="queued", owner=user_id))
    arq = getattr(request.app.state, "arq", None)
    recorder = getattr(request.app.state, "usage", None)
    if arq is not None:
        # The worker enforces the chunk ceiling, targets the shelf and stamps the
        # file row itself; dropping any of these here is how an off-process
        # ingest used to run unmetered and leave the document stuck on
        # "uploaded" — and would now also land it on the wrong shelf.
        await arq.enqueue_job(
            "ingest_task", job_id, path, user_id,
            max_chunks=max_chunks, file_id=file_id, shelf=shelf,
        )
    else:  # no worker -> run in this process
        background.add_task(
            _run_ingest, container, store, job_id, path, user_id, max_chunks, db,
            file_id, recorder, account_id, shelf,
        )
    return IngestResponse(job_id=job_id, status="queued")


def _files_key(user: str) -> str:
    return f"files:{user}"


def _account_uuid(user: AuthUser) -> uuid.UUID | None:
    """The account row's id, or None for a dev-mode namespace identity."""
    try:
        return uuid.UUID(str(user.user_id))
    except (ValueError, AttributeError, TypeError):
        return None


async def _reserve_file_slot(
    db, user: AuthUser, limits: Limits, file_id: str, name: str, path: str, size: int,
    shelf: ShelfRef | None = None,
) -> LimitBreach | None:
    """Claim a file slot in Postgres, or explain which quota refused it.

    Counting the rows that exist (rather than a counter that only goes up)
    means a deleted file or a failed ingest gives its slot back on its own.
    The advisory lock makes the count-then-insert atomic per user, so two
    uploads racing cannot both slip past the cap.

    The count is over the whole account, not the shelf. Every quota here is
    something a user holds once — shelves divide their documents, not their
    allowance — so the lock stays keyed on the owner and a second shelf buys
    nobody a second ten files.
    """
    owner = _account_uuid(user)
    if db is None or owner is None:
        return None  # nothing to enforce against; the size cap still applies

    async with session_scope(db) as s:
        await s.execute(
            text("SELECT pg_advisory_xact_lock(hashtext(:k))"), {"k": str(owner)}
        )
        used, stored = (
            await s.execute(
                select(func.count(), func.coalesce(func.sum(File.size_bytes), 0)).where(
                    File.user_id == owner
                )
            )
        ).one()
        if used >= limits.max_files:
            return LimitBreach("max_files", int(used), limits.max_files)

        storage_cap = limits.max_storage_mb * 1024 * 1024
        if int(stored) + size > storage_cap:
            return LimitBreach(
                "max_storage_mb", int(stored) // (1024 * 1024), limits.max_storage_mb
            )

        s.add(
            File(
                id=file_id, user_id=owner, name=name, path=path,
                size_bytes=size, status="uploaded",
                shelf_id=shelf.id if shelf is not None else None,
            )
        )
    return None


async def _release_file_slot(db, user: AuthUser, file_id: str) -> None:
    owner = _account_uuid(user)
    if db is None or owner is None:
        return
    async with session_scope(db) as s:
        await s.execute(
            delete(File).where(File.id == file_id, File.user_id == owner)
        )


@router.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(
    request: Request,
    background: BackgroundTasks,
    file: UploadFile,
    shelf_id: str | None = None,
    container: Container = Depends(get_container),
    store: JobStore = Depends(get_job_store),
    user: AuthUser = Depends(get_current_user),
    limits: Limits = Depends(effective_limits),
    db=Depends(get_db),
) -> IngestResponse:
    """Add a document to one shelf. `shelf_id` omitted means the default shelf."""
    api = container.settings.api
    user_key = user.tenant_id
    shelf = await shelf_for_request(db, user,shelf_id)

    data = await file.read()
    # The per-file cap is the smaller of the server ceiling and the user's own
    # allowance, so raising one user's quota can't exceed what the proxy passes
    # (MAX_UPLOAD_MB, enforced by Caddy before the request reaches here).
    per_file_mb = min(api.max_upload_mb, limits.max_file_mb)
    if len(data) > per_file_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {per_file_mb} MB limit")

    file_id = uuid.uuid4().hex[:8]
    name = Path(file.filename or "upload").name
    dest = _UPLOAD_DIR / (file_id + "_" + name)

    # Before the slot is reserved and the bytes hit the disk, so an over-quota
    # upload leaves nothing behind to clean up.
    budget = await _shelf_chunk_budget(db, user, limits, shelf)

    breach = await _reserve_file_slot(
        db, user, limits, file_id, name, str(dest), len(data), shelf
    )
    if breach is not None:
        raise reject_with(breach)

    try:
        _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    except OSError:
        await _release_file_slot(db, user, file_id)  # don't strand the slot
        raise

    recorder = getattr(request.app.state, "usage", None)
    if recorder is not None:
        await recorder.record(user.user_id, UPLOAD, 1, {"name": name, "bytes": len(data)})
    return await _enqueue(
        request, background, container, store, str(dest), user_key,
        max_chunks=budget, db=db, file_id=file_id,
        account_id=user.user_id, shelf=shelf.slug,
    )


async def _shelf_chunk_budget(
    db, user: AuthUser, limits: Limits, shelf: ShelfRef
) -> int:
    """What `max_chunks` means for an ingest into *this* shelf.

    The pipeline enforces the ceiling by counting the corpus it is writing to,
    which is the right mechanism and the wrong total once a user has several
    corpora: four shelves would each be allowed the full allowance. Handing it
    the allowance minus what the other shelves already hold makes the check it
    performs — `this shelf + incoming > budget` — equivalent to the per-account
    rule `everything + incoming > max_chunks`.

    A user with nothing left is rejected here rather than handed a budget of
    zero, because zero is *falsy* and `_check_capacity` skips the check entirely
    when `max_chunks` is falsy — so passing it on would turn the one user who is
    definitively over quota into the only one with no ceiling at all. Rejecting
    up front is also the better answer: they get a 429 that names the limit
    instead of a job that fails a minute later.
    """
    used = await chunks_elsewhere(db, _account_uuid(user), shelf)
    remaining = limits.max_chunks - used
    if remaining <= 0:
        raise reject_with(LimitBreach("max_chunks", used, limits.max_chunks))
    return remaining


@router.get("/ingest/files", response_model=FileList)
async def list_files(
    shelf_id: str | None = None,
    user: AuthUser = Depends(get_current_user),
    limits: Limits = Depends(effective_limits),
    db=Depends(get_db),
) -> FileList:
    """Documents, optionally narrowed to one shelf.

    `used` and `limit` stay account-wide even when the list is narrowed: the
    file quota is held once across every shelf, so reporting a per-shelf count
    against it would make "3/10" mean something different on each shelf.
    """
    owner = _account_uuid(user)
    if db is None or owner is None:
        return FileList(files=[], used=0, limit=limits.max_files)

    shelf = await shelf_for_request(db, user,shelf_id) if shelf_id else None
    async with session_scope(db) as s:
        query = select(File).where(File.user_id == owner)
        if shelf is not None:
            # NULL is the default shelf, so a file uploaded before shelves
            # existed must appear there rather than nowhere.
            query = query.where(
                File.shelf_id.is_(None) | (File.shelf_id == shelf.id)
                if shelf.is_default
                else File.shelf_id == shelf.id
            )
        rows = (
            await s.execute(query.order_by(File.created_at.desc()))
        ).scalars().all()
        total = (
            await s.execute(
                select(func.count()).select_from(File).where(File.user_id == owner)
            )
        ).scalar_one()

    files = [
        StoredFile(
            file_id=f.id, name=f.name, source=f.path,
            shelf_id=str(f.shelf_id) if f.shelf_id else None,
        )
        for f in rows
    ]
    return FileList(files=files, used=int(total), limit=limits.max_files)


@router.delete("/ingest/files/{file_id}", response_model=DeleteResponse)
async def delete_file(
    file_id: str,
    container: Container = Depends(get_container),
    user: AuthUser = Depends(get_current_user),
    db=Depends(get_db),
) -> DeleteResponse:
    """Remove an uploaded file, everything it put in the graph, and its slot."""
    user_key = user.tenant_id
    owner = _account_uuid(user)
    if db is None or owner is None:
        raise HTTPException(status_code=503, detail="File tracking needs a database")

    # Look the row up scoped to *this* user, so a file_id from elsewhere
    # cannot reach another tenant's document.
    async with session_scope(db) as s:
        row = (
            await s.execute(
                select(File).where(File.id == file_id, File.user_id == owner)
            )
        ).scalar_one_or_none()
        if row is None:
            raise HTTPException(status_code=404, detail=f"No such file: {file_id}")
        source, shelf_key = row.path, row.shelf_id
        await s.execute(delete(File).where(File.id == file_id, File.user_id == owner))

    # The shelf comes off the row, never off the request: the chunks live in
    # exactly one corpus, and deleting against any other silently succeeds
    # having removed nothing while the row disappears.
    shelf = await shelf_by_id(db, owner, shelf_key)
    tenant = container.tenant(user_key, shelf.slug)
    removed = tenant.graph_store.delete_document(source)
    removed += tenant.vector_store.delete_source(source)  # no-op for Neo4j vectors
    Path(source).unlink(missing_ok=True)
    log.info(
        "file_deleted", user=user_key, file=file_id, shelf=shelf.slug, chunks=removed
    )
    return DeleteResponse(file_id=file_id, chunks_removed=removed)


def _fetch_url(url: str, max_bytes: int) -> tuple[Path, str]:
    """Download a document into data/downloads with a size cap.

    Returns `(path, display_name)`. The two differ because the path carries a
    random prefix to keep two downloads of `report.pdf` apart on disk, while the
    file list should show what the user actually fetched — uploads already make
    that distinction and a URL should not read differently.

    The destination is checked by address, not by scheme: the fetched bytes
    become a document the caller can then query, so an unchecked URL here is a
    read primitive aimed at this deployment's own network — the cloud metadata
    service included. See `graphrag.core.net`.
    """
    parsed = urllib.parse.urlparse(url)
    try:
        with open_public_url(url, timeout=30) as resp:
            data = resp.read(max_bytes + 1)
            ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip()
    except BlockedURLError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except HTTPException:
        raise
    except Exception as exc:
        # Scrubbed: the caller chose the URL, but the failure text can carry
        # more of this server's internals than they asked about.
        raise HTTPException(
            status_code=400, detail=f"Could not fetch URL: {safe_detail(exc, 160)}"
        ) from exc
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail="Remote document exceeds the upload limit")

    name = Path(parsed.path).name or "download"
    if not Path(name).suffix:
        name += _URL_SUFFIX.get(ctype, ".html" if "html" in ctype else ".txt")
    dest = _DOWNLOAD_DIR / (uuid.uuid4().hex[:8] + "_" + name)
    _DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dest, name


def _server_path(path: str) -> Path:
    """Resolve a server-side ingest path, or raise.

    Two conditions, and the second is not a formality. Containment in `data/`
    alone accepted a *directory*, and `iter_documents` walks a directory
    recursively — so `path=data/uploads` ingested every tenant's documents into
    the caller's own corpus, which is precisely the boundary the rest of this
    module maintains. Only a single file is addressable here.
    """
    requested = Path(path)
    try:
        resolved = requested.resolve()
        resolved.relative_to(_DATA_ROOT.resolve())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Server-side ingest is restricted to {_DATA_ROOT}/ "
                   "(upload the file, or pass an http(s) URL)",
        ) from None
    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")
    if not resolved.is_file():
        raise HTTPException(
            status_code=400,
            detail="Server-side ingest takes a single file, not a directory.",
        )
    return resolved


@router.post("/ingest", response_model=IngestResponse)
async def ingest_path(
    request: Request,
    background: BackgroundTasks,
    path: str,
    shelf_id: str | None = None,
    container: Container = Depends(get_container),
    store: JobStore = Depends(get_job_store),
    user: AuthUser = Depends(get_current_user),
    limits: Limits = Depends(effective_limits),
    db=Depends(get_db),
) -> IngestResponse:
    """Ingest an http(s) URL, or a single server-side file under `data/`.

    The two branches are not equally privileged. A URL is the caller's own
    content and is metered exactly like an upload — same per-file cap, same
    file and storage slots, same chunk ceiling. A server-side path addresses
    files this deployment already holds, including other tenants' uploads, so
    it is an operator tool and is gated on admin.
    """
    shelf = await shelf_for_request(db, user,shelf_id)
    if path.startswith(("http://", "https://")):
        # The smaller of the server ceiling and the caller's own allowance,
        # matching /ingest/upload — a URL must not be the cheap way past it.
        per_file_mb = min(container.settings.api.max_upload_mb, limits.max_file_mb)
        # Checked before the fetch: a user with no chunk budget left should not
        # cause this server to make an outbound request on their behalf.
        budget = await _shelf_chunk_budget(db, user, limits, shelf)
        dest, name = _fetch_url(path, per_file_mb * 1024 * 1024)

        file_id = uuid.uuid4().hex[:8]
        breach = await _reserve_file_slot(
            db, user, limits, file_id, name, str(dest), dest.stat().st_size, shelf
        )
        if breach is not None:
            dest.unlink(missing_ok=True)  # we downloaded it; don't leave it behind
            raise reject_with(breach)
        return await _enqueue(
            request, background, container, store, str(dest), user.tenant_id,
            max_chunks=budget, db=db, file_id=file_id,
            account_id=user.user_id, shelf=shelf.slug,
        )

    # Admin only, and checked before the path is even resolved so a non-admin
    # cannot use the 400/404 difference to probe what exists on the disk.
    await require_admin_user(request)
    resolved = _server_path(path)
    return await _enqueue(
        request, background, container, store, str(resolved), user.tenant_id,
        max_chunks=limits.max_chunks, shelf=shelf.slug,
    )


@router.get("/ingest/{job_id}", response_model=IngestStatus)
def ingest_status(
    job_id: str,
    user: AuthUser = Depends(get_current_user),
    store: JobStore = Depends(get_job_store),
) -> IngestStatus:
    """Progress of one of *your* ingests.

    Scoped, and authenticated: without both, a job id is a bearer token by
    accident — 48 bits of uuid4 that names how much of someone's document was
    ingested, and on failure why. A job belonging to anyone else is reported as
    unknown rather than forbidden, so the response cannot be used to tell real
    ids from invented ones.
    """
    job = store.get(job_id, owner=user.tenant_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    return IngestStatus(**job.public())
