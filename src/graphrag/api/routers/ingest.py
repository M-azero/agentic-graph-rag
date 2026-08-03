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

from graphrag.api.deps import AuthUser, get_container, get_current_user, get_db, get_job_store
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
from graphrag.jobs import JobStatus, JobStore
from graphrag.limits import effective_limits, reject_with
from graphrag.limits.service import LimitBreach, Limits
from graphrag.pipelines import IngestPipeline

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
    max_chunks: int | None = None,
) -> None:
    """Run the pipeline and record the outcome. Blocking — call it off the loop."""
    store.set(JobStatus(job_id, status="running", owner=user_id))
    try:
        stats = IngestPipeline(container, max_chunks=max_chunks).run(path, user_id=user_id)
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
):
    """One queued ingest, off the event loop so streaming stays responsive."""
    async with _INGEST_SLOT:
        await asyncio.to_thread(
            _inproc_ingest, container, store, job_id, path, user_id, max_chunks
        )
    await _finalize_file(db, file_id, store.get(job_id))


async def _finalize_file(db, file_id: str | None, status: JobStatus | None) -> None:
    """Record how the ingest ended on the file row, so the UI can show a
    failed document instead of one that silently never became searchable."""
    if db is None or not file_id or status is None:
        return
    from sqlalchemy import update as sql_update

    try:
        async with session_scope(db) as s:
            await s.execute(
                sql_update(File)
                .where(File.id == file_id)
                .values(
                    # A partial ingest stored its chunks and they are
                    # searchable — only the graph is thin. Marking the file
                    # "error" would tell the user to re-upload a document that
                    # is in fact working.
                    status="ingested" if status.status in ("done", "partial") else "error",
                    chunks=getattr(status, "chunks", 0) or 0,
                )
            )
    except Exception as exc:
        log.warning("file_status_update_failed", file=file_id, error=str(exc))


async def _enqueue(
    request: Request, background: BackgroundTasks, container: Container,
    store: JobStore, path: str, user_id,
    max_chunks: int | None = None, db=None, file_id: str | None = None,
) -> IngestResponse:
    job_id = uuid.uuid4().hex[:12]
    store.set(JobStatus(job_id, status="queued", owner=user_id))
    arq = getattr(request.app.state, "arq", None)
    if arq is not None:
        await arq.enqueue_job("ingest_task", job_id, path, user_id)
    else:  # no worker -> run in this process
        background.add_task(
            _run_ingest, container, store, job_id, path, user_id, max_chunks, db, file_id
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
    db, user: AuthUser, limits: Limits, file_id: str, name: str, path: str, size: int
) -> LimitBreach | None:
    """Claim a file slot in Postgres, or explain which quota refused it.

    Counting the rows that exist (rather than a counter that only goes up)
    means a deleted file or a failed ingest gives its slot back on its own.
    The advisory lock makes the count-then-insert atomic per user, so two
    uploads racing cannot both slip past the cap.
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
    container: Container = Depends(get_container),
    store: JobStore = Depends(get_job_store),
    user: AuthUser = Depends(get_current_user),
    limits: Limits = Depends(effective_limits),
    db=Depends(get_db),
) -> IngestResponse:
    api = container.settings.api
    user_key = user.tenant_id

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

    breach = await _reserve_file_slot(db, user, limits, file_id, name, str(dest), len(data))
    if breach is not None:
        raise reject_with(breach)

    try:
        _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    except OSError:
        await _release_file_slot(db, user, file_id)  # don't strand the slot
        raise
    return await _enqueue(
        request, background, container, store, str(dest), user_key,
        max_chunks=limits.max_chunks, db=db, file_id=file_id,
    )


@router.get("/ingest/files", response_model=FileList)
async def list_files(
    user: AuthUser = Depends(get_current_user),
    limits: Limits = Depends(effective_limits),
    db=Depends(get_db),
) -> FileList:
    owner = _account_uuid(user)
    if db is None or owner is None:
        return FileList(files=[], used=0, limit=limits.max_files)
    async with session_scope(db) as s:
        rows = (
            await s.execute(
                select(File).where(File.user_id == owner).order_by(File.created_at.desc())
            )
        ).scalars().all()
    files = [StoredFile(file_id=f.id, name=f.name, source=f.path) for f in rows]
    return FileList(files=files, used=len(files), limit=limits.max_files)


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
        source = row.path
        await s.execute(delete(File).where(File.id == file_id, File.user_id == owner))

    tenant = container.tenant(user_key)
    removed = tenant.graph_store.delete_document(source)
    removed += tenant.vector_store.delete_source(source)  # no-op for Neo4j vectors
    Path(source).unlink(missing_ok=True)
    log.info("file_deleted", user=user_key, file=file_id, chunks=removed)
    return DeleteResponse(file_id=file_id, chunks_removed=removed)


def _fetch_url(url: str, max_bytes: int) -> Path:
    """Download a document into data/downloads with a size cap.

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
    return dest


@router.post("/ingest", response_model=IngestResponse)
async def ingest_path(
    request: Request,
    background: BackgroundTasks,
    path: str,
    container: Container = Depends(get_container),
    store: JobStore = Depends(get_job_store),
    user: AuthUser = Depends(get_current_user),
) -> IngestResponse:
    """Ingest a server-side path under `data/`, or an http(s) URL."""
    if path.startswith(("http://", "https://")):
        max_bytes = container.settings.api.max_upload_mb * 1024 * 1024
        dest = _fetch_url(path, max_bytes)
        return await _enqueue(
            request, background, container, store, str(dest), user.tenant_id
        )

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
    return await _enqueue(
        request, background, container, store, str(resolved), user.tenant_id
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
