"""Arq worker: runs ingestion off the API process so a big upload never blocks a
request, and so the heavy work runs in a container you can resource-limit
separately.

Run it:  arq graphrag.worker.WorkerSettings
"""

from __future__ import annotations

import asyncio
import os

from arq.connections import RedisSettings

from graphrag.config.settings import Secrets
from graphrag.container import Container
from graphrag.core.logging import get_logger
from graphrag.core.redact import safe_detail
from graphrag.jobs import JobStatus, JobStore
from graphrag.pipelines import IngestPipeline

log = get_logger(__name__)

# Through Secrets, not raw os.environ — so a bare `arq graphrag.worker...` run
# picks the URL up from .env exactly like every other entrypoint.
_REDIS_URL = Secrets().redis_url


async def ingest_task(ctx: dict, job_id: str, path: str, user_id: str | None) -> None:
    container: Container = ctx["container"]
    store: JobStore = ctx["jobs"]
    # `owner` on every write, including the failure path: the status endpoint
    # matches on it, so an unowned job is not merely unprotected — it is
    # invisible to the user who created it.
    store.set(JobStatus(job_id, status="running", owner=user_id))
    try:
        # The pipeline is blocking (embeddings, Neo4j) — run it off the event loop.
        stats = await asyncio.to_thread(IngestPipeline(container).run, path, user_id)
        store.set(
            JobStatus(
                job_id, status="done", documents=stats.documents, chunks=stats.chunks,
                entities=stats.entities, relations=stats.relations, owner=user_id,
            )
        )
    except Exception as exc:  # record failure; don't crash the worker
        log.warning("ingest_task_failed", job=job_id, error=str(exc))
        store.set(
            JobStatus(job_id, status="error", detail=safe_detail(exc), owner=user_id)
        )


async def startup(ctx: dict) -> None:
    # Build the container once per worker so models load a single time.
    container = Container()
    provider = container.settings.storage.vector.provider
    if provider == "duckdb":
        # DuckDB takes an exclusive file lock, so the API and this worker cannot
        # both hold a tenant's database. Refuse at startup rather than fail
        # every ingest with a confusing IO error at write time.
        raise RuntimeError(
            "The duckdb vector provider requires single-process ownership, so it "
            "cannot run with a separate ingest worker. Either unset "
            "GRAPHRAG_USE_WORKER (ingest runs inside the API), or switch "
            "storage.vector.provider to neo4j."
        )
    ctx["container"] = container
    ctx["jobs"] = JobStore(container.redis)
    log.info("worker_started", vector_provider=provider)


class WorkerSettings:
    functions = [ingest_task]
    on_startup = startup
    redis_settings = RedisSettings.from_dsn(_REDIS_URL)
    max_jobs = int(os.environ.get("GRAPHRAG_WORKER_CONCURRENCY", "2"))
    keep_result = 3600
    job_timeout = int(os.environ.get("GRAPHRAG_JOB_TIMEOUT", "3600"))
