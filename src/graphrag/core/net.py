"""Outbound URL fetching that can't be aimed back at the infrastructure.

`POST /ingest` takes a URL from the caller and the server fetches it. Without a
destination check that is a server-side request forgery primitive, and an
unusually complete one: the response is not merely reflected, it is *ingested*,
so whatever the server can reach becomes a document in the caller's own
knowledge base for them to query at leisure.

What that reaches from inside a container:

* ``169.254.169.254`` — the cloud metadata service. On DigitalOcean
  ``/metadata/v1.json`` includes the droplet's ``user_data``, which is where
  provisioning secrets usually live; the AWS and GCP equivalents serve
  short-lived credentials.
* Service names on the compose network — ``neo4j:7474``, ``guardrails:8080`` —
  none of which expect a hostile caller because none of them are exposed.
* ``127.0.0.1`` — the API itself, from a position no firewall is watching.

So every hop is resolved and checked against the address, not the name.
Redirects are re-checked rather than followed blindly: a public URL that 302s to
``169.254.169.254`` would otherwise walk straight past a check done only on the
URL the user typed.

One residual risk is left, and named rather than papered over: a hostname whose
DNS answer changes between this check and the connection (DNS rebinding) can
still slip through, because the standard library resolves again when it
connects. Closing that means pinning the connection to the validated address,
which needs a custom transport.
"""

from __future__ import annotations

import ipaddress
import socket
import urllib.parse
import urllib.request

from graphrag.core.logging import get_logger

log = get_logger(__name__)

ALLOWED_SCHEMES = ("http", "https")


class BlockedURLError(ValueError):
    """The URL resolves somewhere the server must not fetch on a user's behalf."""


def _is_public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    # `is_global` already excludes private, loopback, link-local and reserved
    # ranges; the rest are spelled out because a future stdlib change to one
    # property should not silently reopen this.
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local          # 169.254.0.0/16 — the metadata service
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    ) and ip.is_global


def resolved_addresses(host: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise BlockedURLError(f"Could not resolve host: {host}") from exc
    return sorted({info[4][0] for info in infos})


def assert_public_url(url: str) -> None:
    """Raise `BlockedURLError` unless every address this URL resolves to is public."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise BlockedURLError("Only http(s) URLs can be ingested")
    host = parsed.hostname
    if not host:
        raise BlockedURLError("URL has no host")

    addresses = resolved_addresses(host)
    if not addresses:
        raise BlockedURLError(f"Could not resolve host: {host}")
    for address in addresses:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            raise BlockedURLError(f"Unusable address for {host}") from None
        if not _is_public(ip):
            # The message names the host but never the address: confirming
            # *which* internal range answered turns a blocked fetch into a
            # port scanner with a nicer interface.
            log.warning("url_fetch_blocked", host=host, address=address)
            raise BlockedURLError(
                f"{host} resolves to a private or link-local address. "
                "Only public URLs can be ingested."
            )


class _ValidatingRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-checks the destination at every hop."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        assert_public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_public_url(url: str, *, timeout: float = 30.0, user_agent: str = "graphrag-ingest"):
    """Open a URL, refusing anything that points at this deployment's own network."""
    assert_public_url(url)
    opener = urllib.request.build_opener(_ValidatingRedirectHandler)
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    return opener.open(request, timeout=timeout)  # noqa: S310 — destination validated above
