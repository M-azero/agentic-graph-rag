"""The per-IP `/auth/*` throttle, end to end.

This is the one place the limits are left ON — every other integration module
calls `relax_auth_limits`, because driving signup and login dozens of times from
one address is precisely what these limits exist to stop.

Every test keys on a freshly generated address. When a local Redis is reachable
the limiter is Redis-backed, so buckets are shared across app instances *and*
survive the process: a fixed address would make the suite pass once and then
fail for the rest of the minute.
"""

from __future__ import annotations

import uuid

import httpx
import pytest
import pytest_asyncio

from graphrag.accounts import AccountService, PgKeyStore
from graphrag.api.app import create_app
from graphrag.container import Container
from tests.integration.conftest import requires_db

pytestmark = [pytest.mark.integration, requires_db]

PASSWORD = "correct-horse-battery"


def _fresh_ip() -> str:
    """An address nothing else in the run will use.

    IPv6 documentation range (2001:db8::/32) rather than 203.0.113.0/24: the
    v4 documentation block has 254 usable hosts, and with a Redis-backed
    limiter a collision between two tests — or between two runs in the same
    minute — is a flaky failure that looks like a real limit breach.
    """
    return f"2001:db8::{uuid.uuid4().hex[:8]}"


@pytest_asyncio.fixture
async def client(db, email_sender):
    container = Container()
    container.settings.auth.enabled = True
    container.settings.storage.vector.provider = "duckdb"
    # Deliberately NOT relaxed. Tightened instead, so a test needs three
    # requests to prove the limit rather than eleven.
    container.settings.auth.rate_limits.login = "3/minute"
    container.settings.auth.rate_limits.signup = "2/minute"

    app = create_app(container)
    app.state.db = db
    app.state.accounts = AccountService(db, container.settings, email_sender)
    app.state.key_store = PgKeyStore(db)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_repeated_logins_from_one_address_are_throttled(client):
    ip = _fresh_ip()
    body = {"email": "nobody@example.com", "password": "guessing"}
    headers = {"X-Real-IP": ip}

    for _ in range(3):
        assert (await client.post("/auth/login", json=body, headers=headers)).status_code == 401

    blocked = await client.post("/auth/login", json=body, headers=headers)
    assert blocked.status_code == 429
    detail = blocked.json()["detail"]
    assert detail["code"] == "rate_limited"
    assert detail["retry_after"] > 0
    assert blocked.headers["retry-after"]


async def test_a_different_address_is_unaffected(client):
    """The point of a per-IP limit. Behind the single Caddy every request
    arrives from one container address, so without `X-Real-IP` forwarding this
    would be a global limit and one attacker would lock out every user."""
    noisy, quiet = _fresh_ip(), _fresh_ip()
    body = {"email": "nobody@example.com", "password": "guessing"}

    for _ in range(4):
        await client.post("/auth/login", json=body, headers={"X-Real-IP": noisy})
    assert (
        await client.post("/auth/login", json=body, headers={"X-Real-IP": noisy})
    ).status_code == 429

    assert (
        await client.post("/auth/login", json=body, headers={"X-Real-IP": quiet})
    ).status_code == 401


async def test_each_endpoint_has_its_own_bucket(client):
    """Exhausting signup must not also block login from the same address —
    otherwise one noisy endpoint takes the whole sign-in flow down with it."""
    ip = _fresh_ip()
    headers = {"X-Real-IP": ip}

    for i in range(2):
        r = await client.post(
            "/auth/signup",
            json={"email": f"u{i}-{uuid.uuid4().hex[:6]}@example.com", "password": PASSWORD},
            headers=headers,
        )
        assert r.status_code == 200

    exhausted = await client.post(
        "/auth/signup",
        json={"email": f"u3-{uuid.uuid4().hex[:6]}@example.com", "password": PASSWORD},
        headers=headers,
    )
    assert exhausted.status_code == 429

    still_open = await client.post(
        "/auth/login",
        json={"email": "nobody@example.com", "password": "guessing"},
        headers=headers,
    )
    assert still_open.status_code == 401


async def test_a_forged_forwarded_for_cannot_reset_the_bucket(client):
    """The spoofing case, over HTTP.

    A caller controls the first `X-Forwarded-For` entry, so if that were the
    bucket key an attacker would get a fresh quota every request. Here the
    trusted `X-Real-IP` stays fixed while the forged header changes, and the
    limit must still bite.
    """
    ip = _fresh_ip()
    body = {"email": "nobody@example.com", "password": "guessing"}

    for i in range(3):
        r = await client.post(
            "/auth/login",
            json=body,
            headers={"X-Real-IP": ip, "X-Forwarded-For": f"10.9.9.{i}"},
        )
        assert r.status_code == 401

    blocked = await client.post(
        "/auth/login",
        json=body,
        headers={"X-Real-IP": ip, "X-Forwarded-For": "10.9.9.250"},
    )
    assert blocked.status_code == 429
