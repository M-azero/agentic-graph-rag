"""Liveness and readiness probes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy import text

from graphrag import __version__
from graphrag.api.deps import get_container
from graphrag.api.schemas import Health, Ready
from graphrag.container import Container
from graphrag.core.logging import get_logger

router = APIRouter(tags=["health"])
log = get_logger(__name__)


@router.get("/health", response_model=Health)
def health() -> Health:
    """Liveness: is this process up? Deliberately checks nothing else, so a
    dependency outage restarts nothing."""
    return Health(status="ok", version=__version__)


@router.get("/ready", response_model=Ready)
async def ready(
    request: Request, container: Container = Depends(get_container)
) -> Ready:
    """Readiness: can this process actually serve a request?

    Postgres counts toward `ready` only when auth is enabled — that is when it
    holds the accounts every request must be resolved against, so without it
    logins and queries 503 while the probe insists all is well. With auth off
    it is optional and its state is reported without gating.
    """
    neo4j_ok = _check_neo4j(container)
    redis_ok = container.redis is not None
    db_ok = await _check_db(request)
    db_required = container.settings.auth.enabled
    return Ready(
        ready=neo4j_ok and (db_ok or not db_required),
        neo4j=neo4j_ok,
        redis=redis_ok,
        database=db_ok,
    )


async def _check_db(request: Request) -> bool:
    db = getattr(request.app.state, "db", None)
    if db is None:
        return False
    try:
        async with db() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        log.warning("ready_database_check_failed", error=str(exc))
        return False


def _check_neo4j(container: Container) -> bool:
    try:
        with container.driver.session(
            database=container.settings.storage.graph.database
        ) as session:
            session.run("RETURN 1").consume()
        return True
    except Exception:
        return False
