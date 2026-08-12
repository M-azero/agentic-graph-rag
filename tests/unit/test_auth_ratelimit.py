"""Which address a per-IP limit is actually counting.

Both cases here are the difference between a working control and a decorative
one, and both are silent when wrong — the limit still "works", it just counts
the wrong thing.
"""

from __future__ import annotations

from starlette.requests import Request

from graphrag.api.ratelimit import client_ip


def _request(headers: dict[str, str] | None = None, client=("10.0.0.9", 51234)) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request(
        {"type": "http", "method": "POST", "path": "/auth/login",
         "headers": raw, "client": client}
    )


def test_x_real_ip_wins():
    """Our own Caddy *sets* this header, so it is the trustworthy one."""
    req = _request({"X-Real-IP": "203.0.113.7", "X-Forwarded-For": "198.51.100.1"})
    assert client_ip(req) == "203.0.113.7"


def test_forwarded_for_uses_the_last_hop_not_the_first():
    """The regression test for header spoofing.

    Proxies APPEND to X-Forwarded-For, so a client that sends its own header
    contributes the first entry. Trusting `XFF[0]` would let an attacker choose
    a fresh rate-limit bucket per request — the limit would exist and count
    nothing. The last hop is the one our proxy added.
    """
    req = _request({"X-Forwarded-For": "1.2.3.4, 10.0.0.5"})
    assert client_ip(req) == "10.0.0.5"


def test_forwarded_for_single_hop():
    assert client_ip(_request({"X-Forwarded-For": "198.51.100.4"})) == "198.51.100.4"


def test_falls_back_to_the_socket_address():
    """No proxy in front: the peer address is already the caller's."""
    assert client_ip(_request()) == "10.0.0.9"


def test_empty_headers_do_not_shadow_the_socket_address():
    """A proxy that sets the header but leaves it blank must not produce an
    empty bucket key that every caller would share."""
    req = _request({"X-Real-IP": "   ", "X-Forwarded-For": " , "})
    assert client_ip(req) == "10.0.0.9"


def test_missing_client_is_survivable():
    """ASGI transports without a peer (in-process tests, some servers) must not
    raise out of a rate-limit check."""
    assert client_ip(_request(client=None)) == "127.0.0.1"


def test_headers_from_a_public_peer_are_ignored():
    """The bundled proxy lives on the compose network, so a public peer address
    means the request did NOT come through it — the API itself is exposed and
    the forwarding headers are the caller's own invention. Honoring them would
    hand out a fresh rate-limit bucket per request and let the caller forge the
    address recorded against every session.

    A genuinely public peer, not a 198.51.100.x documentation address — the
    stdlib files the RFC 5737 ranges under `is_private`, so a doc address here
    would exercise the *trusted* path and pass for the wrong reason."""
    req = _request(
        {"X-Real-IP": "203.0.113.7", "X-Forwarded-For": "1.2.3.4, 10.0.0.5"},
        client=("142.250.72.14", 40000),
    )
    assert client_ip(req) == "142.250.72.14"


def test_headers_from_a_nameless_peer_are_ignored():
    """A peer that is not an IP at all (some in-process transports) is not our
    proxy either; its headers get the same treatment as a public caller's."""
    req = _request({"X-Real-IP": "203.0.113.7"}, client=("testclient", 123))
    assert client_ip(req) == "testclient"
