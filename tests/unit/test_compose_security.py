"""The compose files carry security decisions, and every one of them fails
silently: an env var that stops being forwarded is ignored, not rejected; a
widened procedure grant or a reused credential changes nothing observable.
These tests pin the decisions so a refactor cannot quietly undo them.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def base():
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def overlay():
    return yaml.safe_load(
        (ROOT / "docker-compose.integrations.yml").read_text(encoding="utf-8")
    )


def test_guardrails_forwards_the_fail_mode(overlay):
    """The guardrails container has no env_file (so it can't see the RAG
    stack's whole secret set), which means every knob must be forwarded
    explicitly — or a value set in .env is silently ignored. For
    GUARD_FAIL_MODE that silence is the bad kind: an operator who chose
    `closed` keeps running fail-open, and nothing anywhere says so."""
    guard = overlay["services"]["guardrails"]
    assert "env_file" not in guard, "no env_file is the design; forward vars explicitly"
    env = guard["environment"]
    for var in (
        "GUARD_FAIL_MODE",
        "GUARD_ALLOW_CLIENT_MODE",
        "GUARD_ENABLE_DOCS",
        "GUARD_LOG_INPUTS",
    ):
        assert var in env, f"{var} set in .env would be silently ignored"


def test_llmlens_does_not_hold_the_accounts_credentials(overlay):
    """llmlens shares the Postgres *server*, never the graphrag role: a
    compromised observability stack should get at telemetry, not at password
    hashes and sessions."""
    dsn = overlay["services"]["llmlens-api"]["environment"]["LLMLENS_POSTGRES_DSN"]
    assert dsn.startswith("postgresql://llmlens:")
    assert "GRAPHRAG_POSTGRES_PASSWORD" not in dsn
    assert "POSTGRES_USER" not in dsn


def test_llmlens_clickhouse_is_not_passwordless(overlay):
    env = overlay["services"]["llmlens-clickhouse"]["environment"]
    assert env.get("CLICKHOUSE_PASSWORD"), "an empty password admits every container on the network"
    assert "CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT" not in env, (
        "the server only creates tables; user management is not its to have"
    )


def test_neo4j_apoc_grant_is_narrow(base):
    """apoc.* would also unlock apoc.load.* / apoc.export.* — network and file
    reach from inside the database, waiting for a Cypher injection to use it.
    The code calls exactly two namespaces, both with plain-Cypher fallbacks."""
    grant = base["services"]["neo4j"]["environment"][
        "NEO4J_dbms_security_procedures_unrestricted"
    ]
    assert grant == "apoc.merge.*,apoc.refactor.*"


def test_only_the_proxy_publishes_to_the_network(base, overlay):
    """Every other published port must be loopback-bound: `ufw deny` does not
    close a Docker-published port, so the 127.0.0.1 prefix IS the firewall."""
    for compose in (base, overlay):
        for name, service in compose["services"].items():
            for entry in service.get("ports", []):
                if name == "proxy":
                    continue
                assert str(entry).startswith("127.0.0.1:"), (
                    f"{name} publishes {entry} beyond loopback"
                )
