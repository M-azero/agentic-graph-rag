"""Ingest job status: scoped to its owner, and scrubbed before it leaves.

Found by audit: `GET /ingest/{job_id}` took no authentication and no owner, so
a job id was a bearer token by accident — 48 bits of uuid4 that reported how
much of someone's document was ingested and, on failure, the raw exception
behind it.
"""

from __future__ import annotations

import inspect

from graphrag.core.redact import PLACEHOLDER, redact_secrets, safe_detail
from graphrag.jobs import JobStatus, JobStore


# ── ownership ───────────────────────────────────────────────────────────────
def test_owner_gets_the_job():
    store = JobStore()
    store.set(JobStatus("j1", status="done", owner="tenant-a"))
    assert store.get("j1", owner="tenant-a").status == "done"


def test_another_tenant_gets_nothing():
    store = JobStore()
    store.set(JobStatus("j1", status="done", owner="tenant-a"))
    assert store.get("j1", owner="tenant-b") is None


def test_a_wrong_owner_is_indistinguishable_from_a_missing_job():
    """Both return None so the route 404s identically. A 403 for a real id and
    a 404 for an invented one would let a caller enumerate which ids exist."""
    store = JobStore()
    store.set(JobStatus("real", status="done", owner="tenant-a"))
    assert store.get("real", owner="tenant-b") == store.get("never-existed", owner="tenant-b")


def test_jobs_written_before_owners_existed_are_not_readable_by_anyone():
    """A record with owner=None predates this field. It must not become
    everyone's job just because it belongs to no one."""
    store = JobStore()
    store.set(JobStatus("legacy", status="done"))
    assert store.get("legacy", owner="tenant-a") is None


def test_owner_is_not_echoed_to_the_client():
    """It is an ACL, not a field to publish — it names another account's tenant."""
    job = JobStatus("j1", status="done", owner="tenant-a")
    assert "owner" not in job.public()
    assert job.public()["status"] == "done"


def test_the_status_route_requires_a_user_and_passes_the_owner():
    from graphrag.api.routers import ingest

    src = inspect.getsource(ingest.ingest_status)
    assert "get_current_user" in src, "job status must be authenticated"
    assert "owner=user.tenant_id" in src, "job status must be scoped to the caller"
    assert ".public()" in src, "job status must not echo the owner field"


def test_every_job_write_stamps_an_owner():
    """A write without one is worse than insecure: the status endpoint matches
    on `owner`, so the user who started the ingest could not see it either.

    Parsed rather than grepped, because these constructors span several lines
    and a substring check would pass on a neighbouring call's keyword.
    """
    import ast

    from graphrag import worker
    from graphrag.api.routers import ingest as api

    missing = []
    for module in (api, worker):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if getattr(node.func, "id", None) != "JobStatus":
                continue
            if not any(kw.arg == "owner" for kw in node.keywords):
                missing.append(f"{module.__name__}:{node.lineno}")
    assert not missing, f"JobStatus written without an owner at {missing}"


# ── error scrubbing ─────────────────────────────────────────────────────────
# The fixtures below are shaped like real credentials and are all invented.
# Never paste a live key into a test: the file is committed, and a secret in a
# public repo outlives whatever it was meant to demonstrate.
FAKE_GOOGLE = "AIzaSyEXAMPLEEXAMPLEEXAMPLEEXAMPLE0000"
FAKE_KEYS = (
    "sk-0000000000000000000000000000EXAMPLE",
    "sk-ant-api03-0000000000000000EXAMPLE",
    "re_0000000000000000000EXAMPLE",
    "grk_0000000000000000000000EXAMPLE",
)


def test_a_key_in_a_url_query_is_removed():
    """Google's client puts the API key in the request URL, and the URL in the
    error message."""
    text = redact_secrets(
        f"403 from https://generativelanguage.googleapis.com/v1/models?key={FAKE_GOOGLE}"
    )
    assert FAKE_GOOGLE not in text
    assert PLACEHOLDER in text


def test_bare_vendor_prefixed_keys_are_removed():
    for secret in FAKE_KEYS:
        assert secret not in redact_secrets(f"upstream said: {secret} was rejected")


def test_an_authorization_header_is_removed():
    assert "supersecretvalue" not in redact_secrets(
        "request failed, headers: Authorization: Bearer supersecretvalue"
    )


def test_ordinary_error_text_survives():
    """Over-redaction would make failures unreadable, which is its own outage."""
    msg = "Neo4j connection refused on bolt://neo4j:7687 after 3 attempts"
    assert redact_secrets(msg) == msg


def test_safe_detail_flattens_and_truncates():
    exc = ValueError("line one\n    line two " + "x" * 500)
    out = safe_detail(exc)
    assert "\n" not in out and len(out) <= 301 and out.endswith("…")


def test_safe_detail_never_returns_empty():
    """An empty detail tells the user nothing about why their upload failed."""
    assert safe_detail(RuntimeError("")) == "RuntimeError"
