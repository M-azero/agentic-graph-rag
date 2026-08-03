"""URL ingest must not be aimable at the deployment's own network.

Found by audit: `_fetch_url` checked only the scheme, so any signed-up user
could have the server fetch `http://169.254.169.254/metadata/v1.json` — on
DigitalOcean that returns the droplet's `user_data`, where provisioning secrets
live — and the response was *ingested*, landing in the caller's own knowledge
base for them to query.
"""

from __future__ import annotations

import pytest

from graphrag.core.net import BlockedURLError, assert_public_url


def _resolves_to(monkeypatch, address: str) -> None:
    monkeypatch.setattr(
        "graphrag.core.net.resolved_addresses", lambda host: [address]
    )


# ── the addresses that matter ───────────────────────────────────────────────
@pytest.mark.parametrize(
    ("address", "what"),
    [
        ("169.254.169.254", "cloud metadata service"),
        ("127.0.0.1", "the API itself"),
        ("::1", "loopback over IPv6"),
        ("10.0.0.5", "private range"),
        ("172.17.0.2", "docker bridge"),
        ("192.168.1.10", "home LAN"),
        ("0.0.0.0", "unspecified"),
        ("fd00::1", "IPv6 unique-local"),
        ("fe80::1", "IPv6 link-local"),
    ],
)
def test_private_destinations_are_refused(monkeypatch, address, what):
    _resolves_to(monkeypatch, address)
    with pytest.raises(BlockedURLError):
        assert_public_url("http://anything.example.com/doc")


def test_a_literal_metadata_ip_is_refused(monkeypatch):
    """The obvious attempt, with no DNS in the way."""
    _resolves_to(monkeypatch, "169.254.169.254")
    with pytest.raises(BlockedURLError):
        assert_public_url("http://169.254.169.254/metadata/v1.json")


def test_a_public_address_is_allowed(monkeypatch):
    _resolves_to(monkeypatch, "93.184.216.34")
    assert_public_url("https://example.com/paper.pdf")


def test_one_private_answer_among_several_blocks_the_whole_host(monkeypatch):
    """A host that resolves to both must not be allowed on the strength of its
    public answer — the connection could take either."""
    monkeypatch.setattr(
        "graphrag.core.net.resolved_addresses",
        lambda host: ["93.184.216.34", "169.254.169.254"],
    )
    with pytest.raises(BlockedURLError):
        assert_public_url("http://dual.example.com/")


# ── scheme and shape ────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "url",
    ["file:///etc/passwd", "gopher://x/", "ftp://host/f", "data:text/plain,hi"],
)
def test_non_http_schemes_are_refused(url):
    with pytest.raises(BlockedURLError):
        assert_public_url(url)


def test_a_url_with_no_host_is_refused():
    with pytest.raises(BlockedURLError):
        assert_public_url("http:///just-a-path")


def test_an_unresolvable_host_is_refused(monkeypatch):
    monkeypatch.setattr("graphrag.core.net.resolved_addresses", lambda host: [])
    with pytest.raises(BlockedURLError):
        assert_public_url("http://nowhere.invalid/")


# ── redirects ───────────────────────────────────────────────────────────────
def test_redirects_are_revalidated(monkeypatch):
    """A public URL that 302s to the metadata service would otherwise walk past
    a check performed only on the URL the user typed."""
    from graphrag.core.net import _ValidatingRedirectHandler

    _resolves_to(monkeypatch, "169.254.169.254")
    handler = _ValidatingRedirectHandler()
    with pytest.raises(BlockedURLError):
        handler.redirect_request(
            None, None, 302, "Found", {}, "http://169.254.169.254/metadata/v1.json"
        )


def test_the_error_never_names_the_address(monkeypatch):
    """Reporting which internal address answered turns a blocked fetch into a
    port scanner with a friendlier interface."""
    _resolves_to(monkeypatch, "10.1.2.3")
    with pytest.raises(BlockedURLError) as err:
        assert_public_url("http://probe.example.com/")
    assert "10.1.2.3" not in str(err.value)


# ── the route uses it ───────────────────────────────────────────────────────
def test_ingest_fetches_through_the_guard():
    import inspect

    from graphrag.api.routers import ingest

    src = inspect.getsource(ingest._fetch_url)
    assert "open_public_url" in src
    assert "urllib.request.urlopen" not in src, (
        "a raw urlopen here bypasses the destination check"
    )
