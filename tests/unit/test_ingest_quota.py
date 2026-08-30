"""Every route that adds a document goes through the same quota.

`/ingest/upload` reserved a file slot, checked storage, and passed the chunk
ceiling down to the pipeline. `POST /ingest` did none of it — no
`effective_limits` dependency, no `_reserve_file_slot`, no `max_chunks` — so a
URL was the cheap way past `max_files`, `max_storage_mb` and `max_chunks` at
once, and left no `files` row to delete afterwards.

The tests below hold the two paths to one contract rather than checking each in
isolation, because the failure was a *divergence* between them.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from graphrag.accounts import KeyOwner
from graphrag.api.app import create_app
from graphrag.api.routers import ingest as ingest_router
from graphrag.config.settings import APICfg, AuthCfg, Secrets, Settings
from graphrag.container import Container
from graphrag.limits.service import LimitBreach

USER_KEY = "grk_quota-user"
ACCOUNT_ID = "22222222-2222-2222-2222-222222222222"


class _FakeKeyStore:
    available = True

    async def resolve(self, key: str) -> KeyOwner | None:
        if key != USER_KEY:
            return None
        return KeyOwner(ACCOUNT_ID, "tenant-q", "user", "active", "q@example.com")


@pytest.fixture
def client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "downloads").mkdir(parents=True)

    settings = Settings(api=APICfg(docs_enabled=False), auth=AuthCfg(enabled=True))
    app = create_app(Container(settings, Secrets(GRAPHRAG_ADMIN_KEY="k")))
    app.state.key_store = _FakeKeyStore()
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def downloaded(tmp_path, monkeypatch) -> Path:
    """Stand in for the network fetch; returns a real file on disk."""
    def _fake_fetch(url: str, max_bytes: int) -> tuple[Path, str]:
        dest = Path("data/downloads") / "abcd1234_doc.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"remote document body")
        _fake_fetch.max_bytes = max_bytes
        return dest, "doc.md"

    monkeypatch.setattr(ingest_router, "_fetch_url", _fake_fetch)
    return _fake_fetch


def _headers() -> dict:
    return {"Authorization": f"Bearer {USER_KEY}"}


def test_url_ingest_reserves_a_file_slot(client, downloaded, monkeypatch):
    seen: dict = {}

    async def _reserve(db, user, limits, file_id, name, path, size, shelf=None):
        seen.update(file_id=file_id, name=name, path=path, size=size)
        return None

    monkeypatch.setattr(ingest_router, "_reserve_file_slot", _reserve)
    monkeypatch.setattr(ingest_router, "_enqueue", _capture_enqueue(seen))

    response = client.post("/ingest", params={"path": "https://example.com/doc.md"},
                           headers=_headers())
    assert response.status_code == 200
    # The name the user sees, not the on-disk name — the random prefix keeps two
    # downloads of the same filename apart and has no business in the file list.
    assert seen["name"] == "doc.md"
    assert seen["path"].endswith("abcd1234_doc.md")
    assert seen["size"] == len(b"remote document body")


def test_url_ingest_is_refused_when_the_quota_is_full(client, downloaded, monkeypatch):
    async def _reserve(*_args, **_kwargs):
        return LimitBreach("max_files", 10, 10)

    monkeypatch.setattr(ingest_router, "_reserve_file_slot", _reserve)

    response = client.post("/ingest", params={"path": "https://example.com/doc.md"},
                           headers=_headers())
    assert response.status_code == 429
    assert response.json()["detail"]["limit"] == "max_files"


def test_a_refused_url_ingest_leaves_no_file_behind(client, downloaded, monkeypatch, tmp_path):
    """The download happened before the quota was known, so it is ours to undo —
    otherwise a user at their limit can still fill the disk by retrying."""
    async def _reserve(*_args, **_kwargs):
        return LimitBreach("max_storage_mb", 100, 100)

    monkeypatch.setattr(ingest_router, "_reserve_file_slot", _reserve)

    client.post("/ingest", params={"path": "https://example.com/doc.md"}, headers=_headers())
    assert list((tmp_path / "data" / "downloads").iterdir()) == []


def test_url_ingest_passes_the_chunk_ceiling_down(client, downloaded, monkeypatch):
    seen: dict = {}

    async def _reserve(*_args, **_kwargs):
        return None

    monkeypatch.setattr(ingest_router, "_reserve_file_slot", _reserve)
    monkeypatch.setattr(ingest_router, "_enqueue", _capture_enqueue(seen))

    client.post("/ingest", params={"path": "https://example.com/doc.md"}, headers=_headers())
    # The shipped default; the point is that *something* is passed, where the
    # old code passed nothing and the pipeline skipped `_check_capacity`.
    assert seen["max_chunks"] == 20_000
    assert seen["file_id"]


def test_the_per_file_cap_is_the_smaller_of_the_two(client, downloaded, monkeypatch):
    """A URL must not be able to exceed what an upload of the same size would."""
    async def _reserve(*_args, **_kwargs):
        return None

    monkeypatch.setattr(ingest_router, "_reserve_file_slot", _reserve)
    monkeypatch.setattr(ingest_router, "_enqueue", _capture_enqueue({}))

    client.post("/ingest", params={"path": "https://example.com/doc.md"}, headers=_headers())
    # min(api.max_upload_mb=25, limits.max_file_mb=15) -> 15 MB
    assert downloaded.max_bytes == 15 * 1024 * 1024


def test_the_worker_path_carries_the_quota_too(client, downloaded, monkeypatch):
    """`_enqueue` used to hand arq only (job_id, path, user_id), so an
    off-process ingest ran with no chunk ceiling and never stamped the file row.
    Both live in the job arguments, so both have to be sent."""
    enqueued: dict = {}

    class _FakeArq:
        async def enqueue_job(self, name, *args, **kwargs):
            enqueued.update(name=name, args=args, kwargs=kwargs)

    async def _reserve(*_args, **_kwargs):
        return None

    monkeypatch.setattr(ingest_router, "_reserve_file_slot", _reserve)
    client.app.state.arq = _FakeArq()

    response = client.post("/ingest", params={"path": "https://example.com/doc.md"},
                           headers=_headers())
    assert response.status_code == 200
    assert enqueued["name"] == "ingest_task"
    assert enqueued["kwargs"]["max_chunks"] == 20_000
    assert enqueued["kwargs"]["file_id"]


def test_the_worker_accepts_what_the_api_sends():
    """A signature check, because the two sides are wired by name across a
    process boundary — a rename here fails at runtime, in a worker log nobody
    is watching, on the quota-enforcing argument."""
    import inspect

    from graphrag.worker import ingest_task

    params = inspect.signature(ingest_task).parameters
    assert {"max_chunks", "file_id"} <= set(params)
    assert params["max_chunks"].default is None
    assert params["file_id"].default is None


def test_an_unsupported_upload_is_refused_before_it_costs_anything(client, monkeypatch):
    """415 at the request, not a failed job a minute later.

    No loader claims `.zip`, so this ingest fails either way. The question is
    where: accepting it means a 200, a reserved file slot, the caller's storage
    allowance spent and the bytes on disk, with the real answer arriving later
    in the job's `detail` — after the UI has already said the upload worked.
    """
    reserved: list = []

    async def _reserve(*args, **kwargs):
        reserved.append(args)
        return None

    monkeypatch.setattr(ingest_router, "_reserve_file_slot", _reserve)

    response = client.post(
        "/ingest/upload",
        files={"file": ("payload.zip", b"PKnope", "application/zip")},
        headers=_headers(),
    )
    assert response.status_code == 415
    assert ".pdf" in response.json()["detail"]  # tells the caller what would work
    assert reserved == []  # no slot taken, so nothing to release


def test_a_supported_upload_still_gets_through_the_type_gate(client, monkeypatch):
    """The positive control: the gate must not be the thing that breaks upload."""
    seen: dict = {}

    async def _budget(*_args, **_kwargs):
        return 20_000

    async def _reserve(*_args, **_kwargs):
        return None

    monkeypatch.setattr(ingest_router, "_shelf_chunk_budget", _budget)
    monkeypatch.setattr(ingest_router, "_reserve_file_slot", _reserve)
    monkeypatch.setattr(ingest_router, "_enqueue", _capture_enqueue(seen))

    response = client.post(
        "/ingest/upload",
        files={"file": ("notes.md", b"# Notes", "text/markdown")},
        headers=_headers(),
    )
    assert response.status_code == 200
    assert seen["path"].endswith("notes.md")


def test_the_type_gate_reads_the_loader_registry(client, monkeypatch):
    """A format added to `build_loaders` must be uploadable with no second edit.

    The gate used to be the kind of thing that gets written out as a literal
    list next to the handler; that list is stale the first time a loader lands.
    """
    from graphrag.ingestion.loaders import supported_suffixes

    async def _budget(*_args, **_kwargs):
        return 20_000

    async def _reserve(*_args, **_kwargs):
        return None

    monkeypatch.setattr(ingest_router, "_shelf_chunk_budget", _budget)
    monkeypatch.setattr(ingest_router, "_reserve_file_slot", _reserve)
    monkeypatch.setattr(ingest_router, "_enqueue", _capture_enqueue({}))

    assert ".xlsx" in supported_suffixes()
    response = client.post(
        "/ingest/upload",
        files={"file": ("book.xlsx", b"stub", "application/octet-stream")},
        headers=_headers(),
    )
    assert response.status_code == 200


def test_an_unsupported_url_is_refused_without_fetching_it(client, monkeypatch):
    """The same 415 as an upload, and before the outbound request.

    The URL path already refuses a caller who is out of chunk budget before it
    fetches, on the grounds that they should not cost this server an outbound
    request. A URL naming a file type nothing can read is the same case: the
    ingest cannot succeed, so the download, the disk write and the file slot
    are all spent on a job that exists only to fail.
    """
    fetched: list = []

    def _fake_fetch(url: str, max_bytes: int):
        fetched.append(url)
        raise AssertionError("should have been refused before the fetch")

    monkeypatch.setattr(ingest_router, "_fetch_url", _fake_fetch)

    response = client.post(
        "/ingest", params={"path": "https://example.com/archive.zip"}, headers=_headers()
    )
    assert response.status_code == 415
    assert fetched == []  # no outbound request made on the caller's behalf


def test_an_extensionless_url_is_still_allowed_through(client, downloaded, monkeypatch):
    """`_fetch_url` names those from the response's content type, so refusing
    them up front would break the ordinary `https://example.com/article` case."""
    async def _reserve(*_args, **_kwargs):
        return None

    monkeypatch.setattr(ingest_router, "_reserve_file_slot", _reserve)
    monkeypatch.setattr(ingest_router, "_enqueue", _capture_enqueue({}))

    response = client.post(
        "/ingest", params={"path": "https://example.com/article"}, headers=_headers()
    )
    assert response.status_code == 200


def test_a_fetch_that_lands_on_an_unsupported_name_is_cleaned_up(client, monkeypatch, tmp_path):
    """The post-fetch guard: `_URL_SUFFIX` must not be able to introduce a
    suffix no loader claims. We downloaded the bytes, so we delete them."""
    def _fake_fetch(url: str, max_bytes: int):
        dest = Path("data/downloads") / "abcd1234_thing.bin"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"payload")
        return dest, "thing.bin"

    monkeypatch.setattr(ingest_router, "_fetch_url", _fake_fetch)

    response = client.post(
        "/ingest", params={"path": "https://example.com/thing"}, headers=_headers()
    )
    assert response.status_code == 415
    assert list((tmp_path / "data" / "downloads").iterdir()) == []


def _capture_enqueue(sink: dict):
    async def _enqueue(request, background, container, store, path, user_id, **kwargs):
        sink.update(kwargs, path=path, user_id=user_id)
        from graphrag.api.schemas import IngestResponse

        return IngestResponse(job_id="test", status="queued")

    return _enqueue


def test_the_file_list_reports_how_each_ingest_ended(client, monkeypatch):
    """The durable answer to "did this work?" has to reach the browser.

    `files.status`, `.chunks` and `.job_id` were recorded on the row from the
    start and never returned, so the only carrier of an upload's outcome was
    the panel's in-memory job list — gone the moment the panel closes, the
    shelf changes or the page reloads. A document then looks identical whether
    it ingested, failed, or is still running.
    """
    from graphrag.api.schemas import StoredFile

    fields = StoredFile.model_fields
    assert {"status", "chunks", "job_id"} <= set(fields)
    # "uploaded" is the row's own default: written before the work starts, so a
    # file that has not finished must not read as a successful one.
    assert fields["status"].default == "uploaded"


def test_a_file_is_linked_to_its_job_when_the_job_is_created(client, downloaded, monkeypatch):
    """Linked at creation, not completion — the link is most useful while the
    ingest is still running, and it is what points a finished document at its
    run in the inspector (an ingest is traced under its job id)."""
    linked: dict = {}

    async def _link(db, file_id, job_id):
        linked.update(file_id=file_id, job_id=job_id)

    async def _reserve(*_args, **_kwargs):
        return None

    monkeypatch.setattr(ingest_router, "link_file_to_job", _link)
    monkeypatch.setattr(ingest_router, "_reserve_file_slot", _reserve)

    response = client.post(
        "/ingest", params={"path": "https://example.com/doc.md"}, headers=_headers()
    )
    assert response.status_code == 200
    assert linked["job_id"] == response.json()["job_id"]
    assert linked["file_id"]
