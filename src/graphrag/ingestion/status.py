"""Record how an ingest ended on the `files` row.

Split out of the API router because the arq worker has to do exactly the same
thing: without it a file ingested off-process stays `uploaded` forever and the
UI shows a document that never became searchable. One definition, so the two
paths cannot drift.
"""

from __future__ import annotations

from graphrag.core.logging import get_logger
from graphrag.db.engine import session_scope
from graphrag.db.models import File

log = get_logger(__name__)

# Job states that mean the chunks are stored and searchable. `partial` counts:
# only the knowledge graph is thin, so marking the file `error` would tell the
# user to re-upload a document that is in fact working.
_SEARCHABLE = ("done", "partial")


async def finalize_file(db, file_id: str | None, status) -> None:
    """Stamp the terminal ingest state on a file row. Never raises."""
    if db is None or not file_id or status is None:
        return
    from sqlalchemy import update as sql_update

    try:
        async with session_scope(db) as s:
            await s.execute(
                sql_update(File)
                .where(File.id == file_id)
                .values(
                    status="ingested" if status.status in _SEARCHABLE else "error",
                    chunks=getattr(status, "chunks", 0) or 0,
                )
            )
    except Exception as exc:
        log.warning("file_status_update_failed", file=file_id, error=str(exc))
