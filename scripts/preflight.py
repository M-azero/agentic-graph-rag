#!/usr/bin/env python
"""Check every external dependency this deployment needs, before users do.

    python scripts/preflight.py              # everything
    python scripts/preflight.py --quick      # skip the paid model calls
    python scripts/preflight.py --vendors    # also: how many vendors each role reaches
    docker compose exec -T api python scripts/preflight.py

Run it after any change to `.env` and before opening a deployment to traffic.
The failure it exists to catch is the quiet one: a key that was revoked or hit
a billing block still *constructs* a perfectly good client, so nothing looks
wrong until a user's question returns a 500. Each provider here gets one real
(tiny) call, which is the only thing that actually proves a credential works.

Fallback chains change what "broken" means, so the exit code follows the same
rule the app does:

  FAIL  a surface has no working provider at all -> exit 1
  WARN  a fallback link is down but the surface still has one -> exit 0

so a degraded-but-serving deployment doesn't block a deploy, and a dead one does.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphrag.config.loader import load_settings  # noqa: E402
from graphrag.llm.factory import build_chat_model, has_credentials  # noqa: E402

OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"

_MARK = {OK: "PASS", WARN: "WARN", FAIL: "FAIL", SKIP: "SKIP"}
_COLOR = {OK: "\033[32m", WARN: "\033[33m", FAIL: "\033[31m", SKIP: "\033[90m"}
_RESET = "\033[0m"


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str]] = []

    def add(self, surface: str, target: str, state: str, detail: str = "") -> str:
        self.rows.append((surface, target, state, detail))
        return state

    @property
    def failed(self) -> bool:
        return any(state == FAIL for _, _, state, _ in self.rows)

    def render(self) -> None:
        width_s = max(len(r[0]) for r in self.rows) + 2
        width_t = max(len(r[1]) for r in self.rows) + 2
        print()
        for surface, target, state, detail in self.rows:
            colour = _COLOR[state] if sys.stdout.isatty() else ""
            reset = _RESET if sys.stdout.isatty() else ""
            line = f"  {colour}{_MARK[state]:<5}{reset} {surface:<{width_s}}{target:<{width_t}}"
            print(f"{line}{detail}" if detail else line)
        print()


def _short(exc: BaseException, limit: int = 110) -> str:
    text = " ".join(str(exc).split())
    return text[:limit] + ("…" if len(text) > limit else "")


# --- model providers ---------------------------------------------------------

def check_chat_chain(
    report: Report, surface: str, primary, fallbacks, secrets, quick: bool,
    *, degraded_ok: str = "",
):
    """One real call per link. A chain passes if any link answers.

    A dead link is only WARN while some other link still serves — that is the
    whole point of the chain. `degraded_ok` names a non-LLM backstop (tesseract
    for OCR) that keeps even a fully dead chain from being a deployment blocker.
    """
    links = [(primary.provider, primary.model)] + [(f.provider, f.model) for f in fallbacks]
    alive = 0
    for provider, model in links:
        target = f"{provider}:{model}"
        if not has_credentials(provider, secrets):
            report.add(surface, target, WARN, "no API key set")
            continue
        if quick:
            report.add(surface, target, SKIP, "--quick")
            alive += 1
            continue
        try:
            client = build_chat_model(provider, model, secrets, max_tokens=4)
            client.invoke("Reply with the single word: ok")
        except Exception as exc:
            report.add(surface, target, WARN, _short(exc))
            continue
        report.add(surface, target, OK)
        alive += 1

    if alive:
        report.add(surface, "(chain)", OK, f"{alive}/{len(links)} links healthy")
    elif degraded_ok:
        report.add(surface, "(chain)", WARN, f"no LLM available; falling back to {degraded_ok}")
    else:
        report.add(surface, "(chain)", FAIL, "no working provider - this surface is down")
    return alive


def check_embeddings(report: Report, settings, secrets, quick: bool):
    """Embeddings have no fallback by design, so this one is pass/fail: without
    it nothing can be ingested and no query can be answered."""
    cfg = settings.embeddings
    target = f"{cfg.provider}:{cfg.model}"
    if quick:
        return report.add("embeddings", target, SKIP, "--quick")
    try:
        from graphrag.container import Container

        vector = Container(settings, secrets).embedder.embed_query("preflight")
    except Exception as exc:
        return report.add("embeddings", target, FAIL, _short(exc))
    if cfg.dimensions and len(vector) != cfg.dimensions:
        return report.add(
            "embeddings", target, FAIL,
            f"returned {len(vector)} dims, config says {cfg.dimensions}",
        )
    return report.add("embeddings", target, OK, f"{len(vector)} dims")


def check_rerank(report: Report, settings, secrets, quick: bool):
    cfg = settings.retrieval.rerank
    if not cfg.enabled or cfg.provider == "none":
        return report.add("rerank", "(disabled)", SKIP)
    target = f"{cfg.provider}:{cfg.model}"
    if quick:
        return report.add("rerank", target, SKIP, "--quick")
    try:
        from graphrag.core.types import RetrievedChunk
        from graphrag.retrieval import build_reranker

        chunks = [RetrievedChunk(chunk_id="p", text="preflight probe", source="p", score=0.0)]
        out = build_reranker(cfg, secrets).rerank("preflight", chunks, top_k=1)
    except Exception as exc:
        return report.add("rerank", target, FAIL, _short(exc))
    from graphrag.retrieval.reranker import CALIBRATED

    if not out or not out[0].metadata.get(CALIBRATED):
        # Every link failed and the chain degraded to retrieval order. Queries
        # still work, but the closed-domain gate suspends itself.
        return report.add("rerank", target, WARN, "chain degraded — relevance gate will suspend")
    return report.add("rerank", target, OK)


# --- credential hygiene --------------------------------------------------------

# Every placeholder this repo itself publishes (.env.example, compose
# fallbacks). Anyone on the internet can read these, so a deployment still
# using one has a password in name only.
_PUBLISHED_DEFAULTS = {"", "12345678", "change-me", "please-change-me"}


def check_credentials(report: Report, settings, secrets):
    """Fail a production deploy that still runs on the repo's own placeholders.

    The loopback port bindings contain the blast radius, but that containment
    is one edited `ports:` line thick — and these are the passwords guarding
    every account row. Gate on `auth.enabled` because that is what separates
    the profiles that face strangers from the dev ones.
    """
    hard = FAIL if settings.auth.enabled else WARN
    if (secrets.neo4j_password or "") in _PUBLISHED_DEFAULTS:
        report.add(
            "credentials", "neo4j", hard,
            "GRAPHRAG_NEO4J_PASSWORD is a published placeholder - set a real one "
            "(new value needs a fresh volume; it bakes in at first start)",
        )
    else:
        report.add("credentials", "neo4j", OK)

    if secrets.database_url:
        from urllib.parse import urlsplit

        password = urlsplit(secrets.database_url).password or ""
        if password in _PUBLISHED_DEFAULTS:
            report.add(
                "credentials", "postgres", hard,
                "the database password is a published placeholder - set "
                "GRAPHRAG_POSTGRES_PASSWORD (bakes in at first start, like neo4j)",
            )
        else:
            report.add("credentials", "postgres", OK)


# --- infrastructure ----------------------------------------------------------

def check_neo4j(report: Report, settings, secrets):
    try:
        from graphrag.storage.neo4j_client import driver_from_secrets

        driver = driver_from_secrets(secrets)
        with driver.session(database=settings.storage.graph.database) as session:
            session.run("RETURN 1").consume()
        driver.close()
    except Exception as exc:
        return report.add("neo4j", secrets.neo4j_uri, FAIL, _short(exc))
    return report.add("neo4j", secrets.neo4j_uri, OK)


def check_redis(report: Report, secrets):
    try:
        from graphrag.cache import get_redis

        get_redis(secrets.redis_url).ping()
    except Exception as exc:
        return report.add("redis", secrets.redis_url, FAIL, _short(exc))
    return report.add("redis", secrets.redis_url, OK)


def check_postgres(report: Report, settings, secrets):
    """Required whenever auth is on — it holds every account."""
    required = settings.auth.enabled
    if not secrets.database_url:
        return report.add(
            "postgres", "(unset)", FAIL if required else SKIP,
            "GRAPHRAG_DATABASE_URL is unset and auth is on" if required else "",
        )
    safe = secrets.database_url.split("@")[-1]
    try:
        import asyncio

        from sqlalchemy import text

        from graphrag.db import build_engine

        async def _ping():
            engine = build_engine(secrets.database_url)
            try:
                async with engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))
            finally:
                await engine.dispose()

        asyncio.run(_ping())
    except Exception as exc:
        return report.add("postgres", safe, FAIL if required else WARN, _short(exc))
    return report.add("postgres", safe, OK)


def check_email(report: Report, settings, secrets):
    """Open registration is only usable if codes actually reach strangers."""
    from graphrag.accounts import build_email_sender

    sender = type(build_email_sender(settings, secrets)).__name__
    public_signup = settings.auth.enabled and settings.auth.open_registration

    if sender == "ConsoleSender":
        return report.add(
            "email", "console", FAIL if public_signup else WARN,
            "codes only go to the log - nobody but you can complete a signup",
        )
    # Resend's shared sender is the trap here: the key is valid, the API
    # returns 200 on the call that matters to us, and delivery still only ever
    # reaches the address the Resend account itself is registered under. With
    # open registration that means every real signup silently dead-ends.
    if "resend.dev" in secrets.email_from:
        return report.add(
            "email", secrets.email_from, FAIL if public_signup else WARN,
            "shared resend.dev sender only delivers to your own address; "
            "verify your domain and send from it",
        )
    return report.add("email", f"{sender} <{secrets.email_from}>", OK)


def check_guardrails(report: Report, settings, secrets) -> list[str]:
    """Probe the guard, and report the judge chain behind it.

    Returns the chain labels so the vendor-spread section can include a role
    that lives in a different process and isn't in this repo's config.
    """
    if not settings.safety.enabled:
        report.add("guardrails", "(disabled)", SKIP)
        return []
    url = (secrets.guardrails_url or settings.safety.base_url).rstrip("/")
    # fail_open means an unreachable guard still serves traffic, unscreened.
    down = WARN if settings.safety.fail_open else FAIL
    headers = (
        {"Authorization": f"Bearer {secrets.guardrails_api_key}"}
        if secrets.guardrails_api_key
        else {}
    )
    try:
        import httpx

        health = httpx.get(f"{url}/health", timeout=5.0)
        health.raise_for_status()
        body = health.json()
    except Exception as exc:
        report.add("guardrails", url, down, _short(exc))
        return []

    provider = body.get("provider", "?")
    chain: list[str] = list(body.get("chain") or [])
    if provider == "mock":
        # The compose overlay defaults the judge to `mock` so the stack comes
        # up with no key — fine on a laptop, but a deployment that faces
        # strangers with safety.enabled and a mock judge is *reporting* a
        # safety layer it does not have. Same auth.enabled gate as the
        # credential check: that is the line between dev and production.
        report.add(
            "guardrails", url, FAIL if settings.auth.enabled else WARN,
            "judge is 'mock' - nothing is really screened; set GUARD_LLM_PROVIDER",
        )
        return chain

    # A judge chain is only as configured as its links. Links get DROPPED at
    # startup when their key is missing from the guardrails container's
    # environment — it has no env_file, so a key that is set for the API is not
    # automatically set here. The service comes up healthy either way.
    for label in chain[1:]:
        report.add("guardrails", label, OK, "judge fallback configured")
    if len(chain) <= 1:
        report.add(
            "guardrails", "(judge chain)", WARN,
            "single judge, no failover - if it stops answering the guard fails "
            "open and nothing is screened",
        )

    # /health reports the *configured* judge, not a working one. Screening runs
    # on a real LLM call, so a denied key leaves the service healthy and every
    # verdict useless. Only an actual verdict proves otherwise.
    try:
        verdict = httpx.post(
            f"{url}/v1/guard/input",
            json={"input": "preflight probe", "policy_id": settings.safety.policy_id},
            headers=headers,
            timeout=settings.safety.timeout_s + 10,
        )
        verdict.raise_for_status()
        body_v = verdict.json()
    except Exception as exc:
        report.add("guardrails", url, down, _short(exc))
        return chain

    # `categories` carries dicts on a real verdict and bare strings for service
    # level signals like judge_unavailable, so flatten before looking.
    categories = {
        c.get("name") or c.get("category") if isinstance(c, dict) else c
        for c in (body_v.get("categories") or [])
    }
    categories |= set(body_v.get("reasons") or [])

    if "judge_unavailable" in categories:
        report.add(
            "guardrails", url, WARN,
            f"every judge link failed ({len(chain)} tried) - nothing is screened",
        )
        return chain

    # Which link answered, not which is configured: a verdict served by link 3
    # means the two in front of it are down and nobody was told.
    served = body_v.get("judge", {}).get("provider") or provider
    served_model = body_v.get("judge", {}).get("model") or body.get("model", "?")
    if chain and f"{served}:{served_model}" != chain[0]:
        report.add(
            "guardrails", url, WARN,
            f"judge degraded: {served}:{served_model} answered, primary {chain[0]} did not",
        )
        return chain
    report.add("guardrails", url, OK, f"judge {served}:{served_model}")
    return chain


# --- vendor concentration ----------------------------------------------------

def check_vendor_spread(report: Report, settings, guard_chain: list[str]) -> None:
    """How many *vendors* each role can reach, not how many models.

    Chain depth is not redundancy when every link shares one API key. The
    outages that actually happen are account-level — a billing block, a revoked
    key, a vendor down — and they take out every model behind that key at once.
    A five-deep chain on one key survives none of them.

    `embeddings` is single-vendor on purpose and says so: a fallback there would
    write vectors into a different space and silently poison the index.
    """
    roles: list[tuple[str, list[str], str]] = []

    def chain_vendors(cfg) -> list[str]:
        return [cfg.provider] + [f.provider for f in getattr(cfg, "fallbacks", [])]

    roles.append(("chat", chain_vendors(settings.llm), ""))
    if settings.ingestion.llm is not None:
        roles.append(("extraction", chain_vendors(settings.ingestion.llm), ""))
    if settings.ocr.enabled and settings.ocr.engine == "vision_llm":
        vendors = chain_vendors(settings.ocr.vision_llm)
        if settings.ocr.fallback_engine:
            vendors.append(f"local:{settings.ocr.fallback_engine}")
        roles.append(("ocr", vendors, ""))
    rerank = settings.retrieval.rerank
    if rerank.enabled and rerank.provider != "none":
        roles.append(("rerank", chain_vendors(rerank), ""))
    roles.append((
        "embeddings", [settings.embeddings.provider],
        "single vendor by design - a fallback would corrupt the vector space",
    ))
    if guard_chain:
        # These are guardrails preset names, not vendor names — `custom` is
        # whatever GUARD_LLM_BASE_URL points at. Distinct presets still mean
        # distinct credentials, which is what this count is really about, but
        # two presets aimed at the same host would flatter the number.
        roles.append(("guard", [label.split(":", 1)[0] for label in guard_chain], ""))

    for role, vendors, exempt in roles:
        unique: list[str] = []
        for v in vendors:
            if v not in unique:
                unique.append(v)
        detail = ", ".join(unique)
        if exempt:
            report.add("vendors", role, SKIP, f"{detail} - {exempt}")
        elif len(unique) == 1:
            report.add("vendors", role, WARN, f"{detail} only - one key takes this role down")
        else:
            report.add("vendors", role, OK, f"{len(unique)} vendors: {detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true",
        help="skip the live model calls (checks keys are present, not that they work)",
    )
    parser.add_argument(
        "--vendors", action="store_true",
        help="also show how many distinct vendors each role can reach",
    )
    args = parser.parse_args()

    settings, secrets = load_settings()
    report = Report()

    print(f"\nProfile: {secrets.profile}")

    check_credentials(report, settings, secrets)
    check_neo4j(report, settings, secrets)
    check_postgres(report, settings, secrets)
    check_redis(report, secrets)

    check_chat_chain(
        report, "chat", settings.llm, settings.llm.fallbacks, secrets, args.quick
    )
    extraction = settings.ingestion.llm
    if extraction is not None:
        check_chat_chain(
            report, "extraction", extraction, extraction.fallbacks, secrets, args.quick
        )
    if settings.ocr.enabled and settings.ocr.engine == "vision_llm":
        vision = settings.ocr.vision_llm
        backstop = settings.ocr.fallback_engine or ""
        check_chat_chain(
            report, "ocr", vision, vision.fallbacks, secrets, args.quick,
            degraded_ok=backstop,
        )
        if backstop:
            report.add("ocr", f"engine:{backstop}", OK, "local last resort, no API needed")

    check_embeddings(report, settings, secrets, args.quick)
    check_rerank(report, settings, secrets, args.quick)

    check_email(report, settings, secrets)
    guard_chain = check_guardrails(report, settings, secrets)

    if args.vendors:
        check_vendor_spread(report, settings, guard_chain)

    report.render()

    if report.failed:
        print("  Not ready: at least one required dependency is down.\n")
        return 1
    if any(state == WARN for _, _, state, _ in report.rows):
        print("  Ready, degraded: serving, but something is down — see WARN above.\n")
        return 0
    print("  Ready.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
