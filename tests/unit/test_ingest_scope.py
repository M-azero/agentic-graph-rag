"""What `POST /ingest` will and will not reach.

The regression this pins: the server-side path fence checked only that a path
resolved *inside* `data/`, and accepted a directory. `iter_documents` walks a
directory recursively, so `path=data/uploads` ingested every tenant's uploaded
documents into the caller's own corpus — readable afterwards through /query and
/search. Containment was never the whole check; "is it one file, and are you
allowed to name server-side files at all" is.

Two independent controls, tested separately, because either one alone leaves a
hole: a file-only fence still lets any user name another tenant's upload by id,
and an admin gate alone still lets an admin slurp the tree by accident.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from graphrag.accounts import KeyOwner
from graphrag.api.app import create_app
from graphrag.api.routers.ingest import _server_path
from graphrag.config.settings import APICfg, AuthCfg, Secrets, Settings
from graphrag.container import Container

ADMIN_KEY = "test-admin-key-not-a-real-one"
USER_KEY = "grk_a-normal-user"


class _FakeKeyStore:
    """Resolves one key to one ordinary, active account."""

    available = True

    async def resolve(self, key: str) -> KeyOwner | None:
        if key != USER_KEY:
            return None
        return KeyOwner("11111111-1111-1111-1111-111111111111", "tenant-a", "user",
                        "active", "a@example.com")


@pytest.fixture
def client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "uploads").mkdir(parents=True)
    (tmp_path / "data" / "uploads" / "victim_secrets.md").write_text("another tenant's document")
    (tmp_path / "data" / "mine.md").write_text("a real file")

    settings = Settings(api=APICfg(docs_enabled=False), auth=AuthCfg(enabled=True))
    app = create_app(Container(settings, Secrets(GRAPHRAG_ADMIN_KEY=ADMIN_KEY)))
    app.state.key_store = _FakeKeyStore()
    return TestClient(app, raise_server_exceptions=False)


def _as_user(**headers) -> dict:
    return {"Authorization": f"Bearer {USER_KEY}", **headers}


# -- the fence itself ---------------------------------------------------------

def test_directory_is_refused(tmp_path, monkeypatch):
    """The bug. A directory resolves inside data/ and used to be accepted."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "uploads").mkdir(parents=True)
    for directory in ("data", "data/uploads", "data/uploads/../uploads"):
        with pytest.raises(HTTPException) as caught:
            _server_path(directory)
        assert caught.value.status_code == 400
        assert "single file" in str(caught.value.detail)


def test_a_single_file_is_accepted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "doc.md").write_text("hello")
    assert _server_path("data/doc.md").name == "doc.md"


@pytest.mark.parametrize("outside", ["../.env", "/etc/passwd", "data/../../secrets.txt"])
def test_traversal_out_of_data_is_refused(outside, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    with pytest.raises(HTTPException) as caught:
        _server_path(outside)
    assert caught.value.status_code == 400


def test_missing_path_is_404(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    with pytest.raises(HTTPException) as caught:
        _server_path("data/nope.md")
    assert caught.value.status_code == 404


# -- the admin gate over the endpoint -----------------------------------------

def test_ordinary_user_cannot_name_a_server_path(client):
    """Even a legitimate single file: server-side paths address documents this
    deployment already holds, which includes other tenants' uploads."""
    response = client.post("/ingest", params={"path": "data/mine.md"}, headers=_as_user())
    assert response.status_code == 403


def test_ordinary_user_cannot_reach_another_tenants_upload(client):
    response = client.post(
        "/ingest",
        params={"path": "data/uploads/victim_secrets.md"},
        headers=_as_user(),
    )
    assert response.status_code == 403


def test_the_gate_runs_before_the_path_is_resolved(client):
    """A non-admin gets the same 403 for a real file and an invented one, so the
    403/404 split cannot be used to map the disk."""
    real = client.post("/ingest", params={"path": "data/mine.md"}, headers=_as_user())
    fake = client.post("/ingest", params={"path": "data/invented.md"}, headers=_as_user())
    assert real.status_code == fake.status_code == 403


def test_admin_still_cannot_ingest_a_directory(client):
    response = client.post(
        "/ingest",
        params={"path": "data/uploads"},
        headers={"X-Admin-Key": ADMIN_KEY, **_as_user()},
    )
    assert response.status_code == 400
    assert "single file" in response.json()["detail"]


def test_unauthenticated_is_rejected(client):
    assert client.post("/ingest", params={"path": "data/mine.md"}).status_code == 401
