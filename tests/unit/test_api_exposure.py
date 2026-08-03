"""What a public deployment does and does not publish.

The admin endpoints have always been locked. What used to be reachable by
anyone who typed the URL was the *description* of them — Swagger, ReDoc and the
OpenAPI schema documented every admin route, its parameters and its responses —
plus /metrics, which maps every route and its traffic. These pin both closed in
production and open in development, since the failure mode in each direction is
silent.

The lifespan is deliberately not entered: none of this should depend on a
database being reachable.
"""

import pytest
from fastapi.testclient import TestClient

from graphrag.api.app import create_app
from graphrag.config.settings import APICfg, AuthCfg, Secrets, Settings
from graphrag.container import Container

ADMIN_KEY = "test-admin-key-not-a-real-one"


def _client(**api_kwargs) -> TestClient:
    settings = Settings(api=APICfg(**api_kwargs), auth=AuthCfg(enabled=True))
    secrets = Secrets(GRAPHRAG_ADMIN_KEY=ADMIN_KEY)
    return TestClient(create_app(Container(settings, secrets)))


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_docs_are_not_served_when_disabled(path):
    """Including /openapi.json: the HTML pages are only viewers for it, so
    leaving the schema up would publish the whole contract regardless."""
    assert _client(docs_enabled=False).get(path).status_code == 404


@pytest.mark.parametrize("path", ["/docs", "/openapi.json"])
def test_docs_are_served_when_enabled(path):
    assert _client(docs_enabled=True).get(path).status_code == 200


def test_metrics_needs_the_admin_key_when_not_public():
    client = _client(metrics_public=False)
    assert client.get("/metrics").status_code == 403
    assert client.get("/metrics", headers={"X-Admin-Key": "wrong"}).status_code == 403
    assert client.get("/metrics", headers={"X-Admin-Key": ADMIN_KEY}).status_code == 200


def test_metrics_stays_open_when_public():
    assert _client(metrics_public=True).get("/metrics").status_code == 200


def test_admin_endpoints_are_closed_without_credentials():
    client = _client(docs_enabled=False)
    for path in ("/admin/users", "/admin/system", "/admin/limits"):
        assert client.get(path).status_code == 403, path


def _routes(app):
    """Every endpoint, flattened.

    `app.routes` is not a flat list: FastAPI wraps each `include_router` call in
    an `_IncludedRouter` whose children hang off `original_router`. Walking only
    the top level finds the app's own routes and none of the routers', which
    would make the test below pass by inspecting nothing.
    """
    def walk(routes):
        for route in routes:
            included = getattr(route, "original_router", None)
            if included is not None:
                yield from walk(included.routes)
            elif getattr(route, "routes", None):
                yield from walk(route.routes)
            else:
                yield route

    return list(walk(app.routes))


def _dependency_names(route) -> set[str]:
    """Every callable in a route's dependency tree, by qualified name."""
    names: set[str] = set()

    def walk(dependant) -> None:
        for sub in getattr(dependant, "dependencies", []):
            call = getattr(sub, "call", None)
            if call is not None:
                names.add(getattr(call, "__qualname__", call.__class__.__name__))
            walk(sub)

    walk(getattr(route, "dependant", None))
    return names


# /auth/logout is the one unauthenticated POST here that is not a guessing
# surface: with no cookie it is a no-op, and with one it needs a 256-bit token
# that is not worth rate limiting the honest case for.
_UNTHROTTLED_BY_DESIGN = {"/auth/logout"}


def test_auth_routes_are_rate_limited():
    """Every credential-guessing endpoint carries a per-IP limit.

    This is the replacement for `_AUTH_PATHS`: a tuple that listed exactly these
    paths, was never referenced by anything, and so documented a protection the
    app did not have. A hand-maintained list cannot notice a new endpoint — this
    fails the moment one ships without a throttle.

    Endpoints behind `get_current_user` are exempt: a caller who already holds a
    session is bounded by the global per-credential bucket, and there is nothing
    left to guess.
    """
    app = create_app(Container(Settings(auth=AuthCfg(enabled=True)), Secrets()))
    unprotected = []
    for route in _routes(app):
        path = getattr(route, "path", "")
        methods = getattr(route, "methods", None) or set()
        if not path.startswith("/auth/") or "POST" not in methods:
            continue
        if path in _UNTHROTTLED_BY_DESIGN:
            continue
        names = _dependency_names(route)
        if "get_current_user" in names:
            continue
        if not any(name.startswith("auth_throttle") for name in names):
            unprotected.append(f"{sorted(methods - {'HEAD', 'OPTIONS'})} {path}")

    assert not unprotected, (
        "unauthenticated /auth endpoints without a per-IP limit: "
        + ", ".join(sorted(unprotected))
    )


def test_production_enables_lockout(production):
    """The counter is useless at zero, and zero is a valid config value."""
    assert production.auth.lockout_threshold > 0
    assert production.auth.rate_limits.login


@pytest.fixture(scope="module")
def production():
    """The real production profile, merged over default.yaml exactly as the
    deployment loads it."""
    from graphrag.config.loader import load_settings

    return load_settings(profile="production")[0]


def test_the_production_profile_actually_sets_these(production):
    """The flags exist to be used. A profile that forgot them would pass every
    test above and still publish the admin API on the real deployment."""
    assert production.api.docs_enabled is False
    assert production.api.metrics_public is False


def test_production_pins_the_secure_cookie_flag(production):
    # Not "auto": behind Caddy every request is HTTPS, and a session cookie
    # without Secure is one downgrade away from being readable.
    assert str(production.auth.cookie_secure).lower() == "true"


def test_production_requires_auth(production):
    """With auth off, the X-User-Id header is the identity and any caller can
    act as any user."""
    assert production.auth.enabled is True


def test_production_does_not_enable_observability(production):
    """llmlens is not deployed on this box. Enabling it would send every prompt
    to a service that isn't running."""
    assert production.observability.enabled is False


def test_production_keeps_the_relevance_gate_on(production):
    assert production.retrieval.min_relevance > 0
