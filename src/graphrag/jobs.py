"""Ingest job status, persisted in Redis so it survives restarts and is visible
across API replicas and the worker. Falls back to an in-process dict when Redis
is unavailable (single-process dev)."""

from __future__ import annotations

import contextlib
import json
from dataclasses import asdict, dataclass

_TTL = 86400  # keep job records for a day


@dataclass
class JobStatus:
    job_id: str
    # queued | running | done | partial | error
    # `partial`: stored and searchable, but graph extraction failed for some
    # chunks — a degraded success, not a failure.
    status: str = "queued"
    detail: str = ""
    documents: int = 0
    chunks: int = 0
    entities: int = 0
    relations: int = 0
    # Tenant this job belongs to. A job id is a bearer token by accident
    # otherwise: 48 bits of uuid4, handed to the client, with a status that
    # names how much of someone's document was ingested and why it failed.
    owner: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def public(self) -> dict:
        """What the API returns — `owner` is an internal ACL, not a field to echo."""
        data = self.to_dict()
        data.pop("owner", None)
        return data


class JobStore:
    def __init__(self, redis_client=None) -> None:
        self._redis = redis_client
        self._mem: dict[str, JobStatus] = {}

    @staticmethod
    def _key(job_id: str) -> str:
        return f"ingest:job:{job_id}"

    def set(self, status: JobStatus) -> None:
        self._mem[status.job_id] = status
        if self._redis is not None:
            with contextlib.suppress(Exception):
                self._redis.setex(self._key(status.job_id), _TTL, json.dumps(status.to_dict()))

    def get(self, job_id: str, owner: str | None = None) -> JobStatus | None:
        """Fetch a job, optionally requiring it to belong to `owner`.

        A mismatch returns None so the caller 404s: a job someone else owns
        should be indistinguishable from one that never existed, or the
        response itself confirms which ids are real.

        Jobs written before `owner` existed have None and are treated as
        unowned — readable only by a caller that asks without an owner, which
        the API never does.
        """
        job = None
        if self._redis is not None:
            with contextlib.suppress(Exception):
                raw = self._redis.get(self._key(job_id))
                if raw:
                    job = JobStatus(**json.loads(raw))
        if job is None:
            job = self._mem.get(job_id)
        if job is None:
            return None
        if owner is not None and job.owner != owner:
            return None
        return job
