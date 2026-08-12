"""Per-IP throttling for the endpoints that can be reached without credentials.

The global `api.rate_limit` bucket keys on the session cookie or API key when
there is one, and falls back to the address otherwise. That is right for the
authenticated surface and far too generous for `/auth/*`, where every request
is a guess at a password or a six-digit code. This module adds a second,
tighter bucket per endpoint.

Two things have to be true for a per-IP limit to mean anything, and neither was
true before:

**The address has to be the caller's, not the proxy's.** Behind the bundled
Caddy every request arrives from one container IP. A limit keyed on that is a
*global* limit — one attacker would lock out every user at once.

**The address has to be one the caller cannot choose.** `X-Forwarded-For` is
appended to, not replaced, so a client that sends its own header contributes
the first entry: trusting `XFF[0]` lets an attacker pick a fresh bucket per
request and makes the limit decorative. The last entry is the one our own
proxy added.
"""

from __future__ import annotations

import ipaddress
import math
import time

from fastapi import HTTPException, Request
from limits import RateLimitItem, parse
from slowapi.util import get_remote_address

from graphrag.core.logging import get_logger

log = get_logger(__name__)

RATE_LIMITED = "rate_limited"


def _peer_may_speak_for_others(request: Request) -> bool:
    """Whether the immediate peer is allowed to name the real caller.

    Only the bundled proxy (a container on the compose network) or a local
    process (loopback — an SSH tunnel, the box itself) can reach this API
    without passing through Caddy's `header_up`. Both arrive from private or
    loopback addresses. A *public* peer address means the API itself has been
    exposed and there is no proxy in the path — at which point the forwarding
    headers are whatever the caller typed, and honoring them would hand an
    attacker a fresh rate-limit bucket per request and let them forge the
    address recorded against every session.
    """
    client = request.client
    if client is None or not client.host:
        return False
    try:
        peer = ipaddress.ip_address(client.host)
    except ValueError:
        # Not an IP at all (some in-process test transports) — no proxy there.
        return False
    return peer.is_private or peer.is_loopback


def client_ip(request: Request) -> str:
    """The caller's address, as far as it can be trusted.

    Forwarding headers are only consulted when the request came *through our
    proxy* (see `_peer_may_speak_for_others`); a directly-reached API counts
    the socket address, which no caller can choose. `X-Real-IP` first: the
    bundled Caddy *sets* it (`header_up X-Real-IP {remote_host}`), overwriting
    anything the client sent. `X-Forwarded-For` is the fallback for a
    different proxy in front, and we take its **last** hop for the reason in
    the module docstring.
    """
    if _peer_may_speak_for_others(request):
        real = (request.headers.get("X-Real-IP") or "").strip()
        if real:
            return real
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            hops = [hop.strip() for hop in forwarded.split(",") if hop.strip()]
            if hops:
                return hops[-1]
    return get_remote_address(request)


def too_many_requests(retry_after: int, message: str) -> HTTPException:
    """The one 429 shape the API emits.

    Structured rather than a bare string so the UI can say "try again in N
    seconds" instead of rendering "429 Too Many Requests" at the user. It
    mirrors the quota 429 raised by `limits.deps`, which carries the same
    `code` / `message` / `retry_after` keys — one client-side branch handles
    both.
    """
    retry_after = max(1, int(retry_after))
    return HTTPException(
        status_code=429,
        detail={"code": RATE_LIMITED, "message": message, "retry_after": retry_after},
        headers={"Retry-After": str(retry_after)},
    )


def _retry_after(strategy, item: RateLimitItem, *identifiers: str) -> int:
    """Seconds until the window reopens; the window length if that can't be read."""
    try:
        stats = strategy.get_window_stats(item, *identifiers)
        return max(1, math.ceil(stats.reset_time - time.time()))
    except Exception:  # pragma: no cover - storage-specific
        return item.get_expiry()


def auth_throttle(name: str):
    """A dependency applying `auth.rate_limits.<name>` per IP to one endpoint.

    It reuses the app's existing `Limiter` rather than creating its own. That
    matters: a module-level limiter (which `@limiter.limit` decorators need, to
    bind at import time) has to choose its storage backend at import, before
    the container exists. A configured-but-unreachable Redis would then make
    slowapi raise on *every* request instead of degrading — turning a cache
    blip into a total outage. `app.state.limiter` has already made that choice
    correctly, once, in `create_app`.

    Reusing it also means this limit *adds to* the global one rather than
    replacing it, which is what slowapi's decorator does by default
    (`override_defaults=True`).
    """

    async def dependency(request: Request) -> None:
        settings = request.app.state.container.settings
        raw = str(getattr(settings.auth.rate_limits, name, "") or "")
        if not raw:
            return  # explicitly disabled in config
        limiter = getattr(request.app.state, "limiter", None)
        if limiter is None:  # pragma: no cover - create_app always sets it
            return

        item = parse(raw)
        # The scope keeps each endpoint's bucket separate, so exhausting
        # /auth/resend does not also block /auth/login from the same address.
        identifiers = ("auth", name, client_ip(request))
        if not limiter.limiter.hit(item, *identifiers):
            wait = _retry_after(limiter.limiter, item, *identifiers)
            log.info("auth_rate_limited", endpoint=name, limit=raw, retry_after=wait)
            raise too_many_requests(
                wait, f"Too many attempts. Try again in {wait} seconds."
            )

    return dependency
