"""Finished traces, persisted in Redis so they survive a restart and are visible
across the API and the ingest worker.

A direct mirror of `graphrag.jobs.JobStore`, and deliberately so: the ingest
pipeline may run in a separate arq process, where a tracer held in the API's
memory does not exist. The worker writes the trace here; the page reads it back
by job id.

Ownership is enforced the same way too. A trace holds the user's question and
excerpts of their documents, so `get` requires the caller's tenant and returns
None on a mismatch — a trace belonging to someone else is indistinguishable
from one that never existed, or the response itself confirms which ids are real.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections import OrderedDict

from graphrag.core.logging import get_logger

log = get_logger(__name__)

# Bounded like the job store's fallback: without a TTL to expire them, the
# in-process dict is a slow leak in exactly the configuration that has no Redis
# to trim it.
_MEM_MAX = 50

# A trace carries chunk text; a very large one is not worth the round trip.
_MAX_BYTES = 512_000

_INDEX_MAX = 25


class TraceStore:
    def __init__(self, redis_client=None, ttl_seconds: int = 3600) -> None:
        self._redis = redis_client
        self._ttl = max(60, ttl_seconds)
        self._mem: OrderedDict[str, dict] = OrderedDict()

    @staticmethod
    def _key(trace_id: str) -> str:
        return f"trace:run:{trace_id}"

    @staticmethod
    def _index_key(owner: str) -> str:
        return f"trace:index:{owner}"

    def save(self, record: dict) -> None:
        """Store a finished trace and add it to its owner's recent list."""
        trace_id = record.get("trace_id")
        if not trace_id:
            return
        self._mem[trace_id] = record
        self._mem.move_to_end(trace_id)
        while len(self._mem) > _MEM_MAX:
            self._mem.popitem(last=False)

        if self._redis is None:
            return
        with contextlib.suppress(Exception):
            blob = json.dumps(record)
            if len(blob) > _MAX_BYTES:
                # Drop the payloads rather than the trace: the shape of the run
                # (which blocks ran, how long, in what order) is the part that
                # cannot be reconstructed later.
                trimmed = dict(record)
                trimmed["steps"] = [
                    {**s, "input": {}, "output": {}, "meta": {"trimmed": True}}
                    for s in record.get("steps", [])
                ]
                trimmed["trimmed"] = True
                blob = json.dumps(trimmed)
                log.info("trace_trimmed", trace=trace_id, bytes=len(blob))
            self._redis.setex(self._key(trace_id), self._ttl, blob)
            owner = record.get("owner")
            if owner:
                self._index(owner, record)

    def _index(self, owner: str, record: dict) -> None:
        """Keep a short recent-runs list per owner, newest first."""
        entry = json.dumps(
            {
                "trace_id": record.get("trace_id"),
                "pipeline": record.get("pipeline"),
                "title": record.get("title", ""),
                "status": record.get("status"),
                "duration_ms": record.get("duration_ms"),
                "steps": len(record.get("steps", [])),
                "at": record.get("started_at", time.time()),
            }
        )
        key = self._index_key(owner)
        pipe = self._redis.pipeline()
        pipe.lpush(key, entry)
        pipe.ltrim(key, 0, _INDEX_MAX - 1)
        pipe.expire(key, self._ttl)
        pipe.execute()

    def get(self, trace_id: str, owner: str | None = None) -> dict | None:
        record = None
        if self._redis is not None:
            with contextlib.suppress(Exception):
                raw = self._redis.get(self._key(trace_id))
                if raw:
                    record = json.loads(raw)
        if record is None:
            record = self._mem.get(trace_id)
        if record is None:
            return None
        if owner is not None and record.get("owner") != owner:
            return None
        return record

    def recent(self, owner: str) -> list[dict]:
        """This owner's recent runs, newest first."""
        if self._redis is not None:
            with contextlib.suppress(Exception):
                raw = self._redis.lrange(self._index_key(owner), 0, _INDEX_MAX - 1)
                if raw:
                    return [json.loads(r) for r in raw]
        # In-process fallback: newest last in the OrderedDict, so reverse it.
        return [
            {
                "trace_id": r.get("trace_id"),
                "pipeline": r.get("pipeline"),
                "title": r.get("title", ""),
                "status": r.get("status"),
                "duration_ms": r.get("duration_ms"),
                "steps": len(r.get("steps", [])),
                "at": r.get("started_at"),
            }
            for r in reversed(self._mem.values())
            if r.get("owner") == owner
        ]


__all__ = ["TraceStore"]
