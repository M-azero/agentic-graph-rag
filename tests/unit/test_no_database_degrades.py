"""What the API does when it has no database but the caller has credentials.

This is the state the lifespan leaves behind when GRAPHRAG_DATABASE_URL is unset
or unparseable: `app.state.db` is None, while `accounts` and `key_store` exist
with a None session factory. Every request carrying a session cookie or an API
key then reached `session_scope(None)` -> `TypeError: 'NoneType' object is not
callable` -> 500.

A 500 is the wrong answer twice over. It is not what happened (the caller is
simply not authenticated against anything), and the UI routes to the sign-in
page on 401 — so a 500 renders as a broken app rather than a signed-out one, for
every returning visitor at once.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from graphrag.accounts import AccountService, PgKeyStore
from graphrag.accounts.emails import ConsoleSender
from graphrag.api.app import create_app
from graphrag.config.settings import APICfg, AuthCfg, Secrets, Settings
from graphrag.container import Container
from graphrag.core.errors import ConfigError
from graphrag.db.engine import session_scope


@pytest.fixture
def settings() -> Settings:
    return Settings(api=APICfg(docs_enabled=False), auth=AuthCfg(enabled=True))


@pytest.fixture
def client(settings) -> TestClient:
    app = create_app(Container(settings, Secrets(GRAPHRAG_ADMIN_KEY="k")))
    # Exactly what app.py's lifespan builds when there is no database URL.
    app.state.db = None
    app.state.accounts = AccountService(None, settings, ConsoleSender(), None)
    app.state.key_store = PgKeyStore(None, None)
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize(
    ("label", "kwargs"),
    [
        ("no credentials", {}),
        ("session cookie", {"cookies": {"graphrag_session": "stale-token"}}),
        ("api key", {"headers": {"Authorization": "Bearer grk_bogus"}}),
        ("x-api-key", {"headers": {"X-API-Key": "grk_bogus"}}),
    ],
)
def test_credentials_without_a_database_are_401_not_500(client, label, kwargs):
    assert client.get("/auth/me", **kwargs).status_code == 401, label


def test_a_mutating_route_is_also_401(client):
    response = client.post(
        "/threads", json={"title": "x"}, cookies={"graphrag_session": "stale"}
    )
    assert response.status_code == 401


def test_logout_with_a_stale_cookie_succeeds(client):
    """Signing out with no database is already true, so it must not raise."""
    assert client.post("/auth/logout", cookies={"graphrag_session": "stale"}).status_code == 200


def test_signing_in_reports_the_real_problem(client):
    """401 says "not signed in"; the sign-in attempt itself has to say why it
    cannot help, or the user loops between the two."""
    response = client.post(
        "/auth/login", json={"email": "a@example.com", "password": "x" * 12}
    )
    assert response.status_code == 503
    assert "database" in response.json()["detail"].lower()


# -- the backstop -------------------------------------------------------------

@pytest.mark.anyio
async def test_session_scope_rejects_a_none_factory_legibly():
    """Callers that genuinely need the database should get a named error, not
    `'NoneType' object is not callable`."""
    with pytest.raises(ConfigError, match="GRAPHRAG_DATABASE_URL"):
        async with session_scope(None):
            pass  # pragma: no cover


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_resolvers_return_none_rather_than_raising(settings):
    accounts = AccountService(None, settings, ConsoleSender(), None)
    keys = PgKeyStore(None, None)
    assert accounts.available is False
    assert keys.available is False
    assert await accounts.resolve_session("anything") is None
    assert await keys.resolve("grk_anything") is None
