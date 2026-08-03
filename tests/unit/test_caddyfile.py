"""Structural guards on docker/Caddyfile.

This file is the only way into the deployment: it terminates TLS, serves both
web apps and reverse-proxies the API. A mistake in it takes the whole site down
or quietly republishes something that should not be public — and no other test
in the repo compiles it. CI additionally runs `caddy validate` against the real
adapter; these are the cheap checks that catch a *valid* config that is wrong.

Every assertion here corresponds to a mistake that was actually made while
writing this file, or to a security control that must not be edited away.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

CADDYFILE = Path(__file__).resolve().parents[2] / "docker" / "Caddyfile"


@pytest.fixture(scope="module")
def caddyfile() -> str:
    return CADDYFILE.read_text(encoding="utf-8")


def test_the_file_exists():
    assert CADDYFILE.is_file(), f"{CADDYFILE} is missing"


# -- what must not be published -----------------------------------------------

@pytest.mark.parametrize(
    "path", ["/api/docs*", "/api/redoc*", "/api/openapi.json", "/api/metrics*"]
)
def test_internal_paths_are_blocked(caddyfile, path):
    """Second of the two independent controls. `api.docs_enabled: false` is the
    first; this one survives a profile edit or a default flipping back."""
    assert path in caddyfile


def test_the_block_is_a_handle_not_a_bare_respond(caddyfile):
    """A bare `respond @internal_only` is sorted by Caddy's *directive* order,
    which puts it after `handle` — so `handle_path /api/*` matches first, the
    request is proxied through, and the rule is silently dead."""
    assert re.search(r"handle\s+@internal_only\s*\{", caddyfile), (
        "the internal-only block must be `handle @internal_only { ... }`"
    )


@pytest.mark.parametrize(
    "header",
    [
        "Strict-Transport-Security",
        "X-Content-Type-Options",
        "X-Frame-Options",
        "Referrer-Policy",
        "Permissions-Policy",
    ],
)
def test_security_headers_are_set(caddyfile, header):
    assert header in caddyfile


def test_the_server_header_is_stripped(caddyfile):
    assert "-Server" in caddyfile


# -- the API path -------------------------------------------------------------

def test_sse_streaming_is_not_buffered(caddyfile):
    """Without this the agent's tokens arrive in one lump at the end, and the
    chat UI looks frozen for the whole generation."""
    assert "flush_interval -1" in caddyfile


def test_the_real_client_ip_is_forwarded(caddyfile):
    """`header_up X-Real-IP` is what makes the per-IP `/auth/*` limits real.

    Without it every request reaches the API from this container's address, so
    a per-IP limit becomes a global one and a single attacker locks out every
    user at once. It must be X-Real-IP specifically: X-Forwarded-For is
    appended to, so its first hop is attacker-controlled.
    """
    assert re.search(r"header_up\s+X-Real-IP\s+\{remote_host\}", caddyfile)


def test_the_upload_cap_is_enforced_and_configurable(caddyfile):
    """Replaces the cap the frontend's nginx used to hold — which never worked,
    because the env var was never passed to that container."""
    assert re.search(r"max_size\s+\{\$MAX_UPLOAD_MB:\d+\}MB", caddyfile), (
        "the API block needs `request_body { max_size {$MAX_UPLOAD_MB:N}MB }`"
    )


# -- static hosting -----------------------------------------------------------

def test_both_apps_are_served(caddyfile):
    assert "/srv/app" in caddyfile
    assert "/srv/admin" in caddyfile


def test_bare_admin_redirects_with_an_explicit_matcher(caddyfile):
    """`redir /admin/ 301` is a trap, not a typo.

    Any directive whose first argument starts with `/` has that argument parsed
    as an inline path matcher. The short form therefore means "for requests to
    /admin/, respond 301 to nowhere" — it validates, and serves an empty 200.
    The `*` matcher is what makes /admin/ the destination.
    """
    assert re.search(r"redir\s+\*\s+/admin/\s+301", caddyfile), (
        "bare /admin must redirect via `redir * /admin/ 301` (note the `*`)"
    )


def test_spa_deep_links_fall_back_to_the_entry_document(caddyfile):
    """Both apps use client-side routing, so /admin/users must serve
    index.html rather than 404."""
    assert caddyfile.count("try_files {path} /index.html") >= 1
    assert caddyfile.count("import spa") == 2, "both roots must use the spa snippet"


def test_the_spa_snippet_is_defined_before_it_is_imported(caddyfile):
    """A snippet declared after the site block that imports it is not in scope,
    and the adapter fails with "File to import not found"."""
    assert caddyfile.index("(spa)") < caddyfile.index("import spa")


def test_cache_rules_use_mutually_exclusive_matchers(caddyfile):
    """The subtlest bug in this file.

    A bare `header` followed by `header @immutable` reads as "default, then
    override" and is not: both handlers run, the unmatched one runs last, and
    every fingerprinted asset comes back `no-store`. It validates and serves
    correctly while silently defeating caching. Written as two `not`-paired
    matchers, there is no ordering to get wrong.
    """
    assert re.search(r"@immutable\s+path\s+/assets/\*", caddyfile)
    assert re.search(r"@entry\s+not\s+path\s+/assets/\*", caddyfile)
    assert "max-age=31536000, immutable" in caddyfile
    assert "no-store, no-cache, must-revalidate" in caddyfile


def test_nothing_still_points_at_the_removed_frontend_container(caddyfile):
    """The static files are baked into this image now. A leftover
    `reverse_proxy frontend:80` would fail DNS on every page load."""
    assert "frontend:80" not in caddyfile
