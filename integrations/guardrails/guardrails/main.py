"""FastAPI application + entrypoint.

Wires Settings -> PolicyRegistry -> Provider chain -> Judge -> Pipeline at startup and
exposes:

    POST /v1/guard/input     POST /v1/guard/output
    GET  /health             GET  /v1/policies      GET /v1/policies/{id}

Design contract: every *well-formed* request returns HTTP 200 with a verdict. Only 422
(malformed body), 404 (unknown policy), and 401 (bad key) are non-200 — judge outages are
folded into the verdict via ``fail_mode``, never surfaced as 5xx.
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets as _secrets
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request

from . import __version__
from .config import Settings, get_settings
from .judge.judge import Judge
from .judge.providers import build_provider_chain
from .pipeline import GuardPipeline
from .policy import PolicyRegistry
from .schemas import (
    GuardInputRequest,
    GuardInputResponse,
    GuardOutputRequest,
    GuardOutputResponse,
    HealthResponse,
    PolicySummary,
)

logger = logging.getLogger("guardrails")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        registry = PolicyRegistry.load_dir(settings.policy_dir, settings.default_policy)
        providers, skipped = build_provider_chain(settings)
        judge = Judge(providers, settings)
        app.state.settings = settings
        app.state.registry = registry
        app.state.judge = judge
        app.state.pipeline = GuardPipeline(settings, registry, judge)
        logger.info(
            json.dumps({"event": "startup", "provider": judge.provider_name,
                        "model": judge.model, "chain": judge.chain,
                        "policies": registry.ids()})
        )
        # Loud, because a silently dropped link is the difference between a
        # chain and a single provider — and the guard fails open, so the first
        # you would otherwise hear of it is nothing being screened.
        if skipped:
            logger.warning(json.dumps({"event": "judge_links_skipped", "links": skipped}))
        if len(providers) == 1 and settings.llm_fallbacks.strip():
            logger.warning(json.dumps({
                "event": "judge_chain_empty",
                "detail": "GUARD_LLM_FALLBACKS is set but no link survived; "
                          "the judge is a single point of failure",
            }))
        yield

    # Hide the interactive docs (/docs, /redoc, /openapi.json) unless explicitly enabled,
    # so the server surfaces nothing beyond the endpoints it serves.
    docs_kwargs: dict = (
        {} if settings.enable_docs
        else {"docs_url": None, "redoc_url": None, "openapi_url": None}
    )
    app = FastAPI(
        title="Guardrails Server", version=__version__, lifespan=lifespan, **docs_kwargs
    )

    async def require_auth(request: Request) -> None:
        key = request.app.state.settings.api_key
        if not key:
            return
        provided = None
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            provided = auth[7:].strip()
        provided = provided or request.headers.get("x-api-key")
        # Constant-time comparison to avoid leaking the key via response timing.
        if provided is None or not _secrets.compare_digest(provided, key):
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    # ── Health (no auth) ─────────────────────────────────────────────────────
    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        judge: Judge = request.app.state.judge
        registry: PolicyRegistry = request.app.state.registry
        return HealthResponse(
            status="ok", version=__version__, provider=judge.provider_name,
            model=judge.model, chain=judge.chain, policies=registry.ids(),
        )

    # ── Guard endpoints ──────────────────────────────────────────────────────
    @app.post("/v1/guard/input", response_model=GuardInputResponse,
              dependencies=[Depends(require_auth)])
    async def guard_input(req: GuardInputRequest, request: Request) -> GuardInputResponse:
        _ensure_policy(request, req.policy_id)
        resp = await request.app.state.pipeline.check_input(req)
        _log_verdict(request.app.state.settings, "input", req.input, resp)
        return resp

    @app.post("/v1/guard/output", response_model=GuardOutputResponse,
              dependencies=[Depends(require_auth)])
    async def guard_output(req: GuardOutputRequest, request: Request) -> GuardOutputResponse:
        _ensure_policy(request, req.policy_id)
        resp = await request.app.state.pipeline.check_output(req)
        _log_verdict(request.app.state.settings, "output", req.output, resp)
        return resp

    # ── Policy introspection ─────────────────────────────────────────────────
    @app.get("/v1/policies", response_model=list[PolicySummary],
             dependencies=[Depends(require_auth)])
    async def list_policies(request: Request) -> list[PolicySummary]:
        return request.app.state.registry.summaries()

    @app.get("/v1/policies/{policy_id}", response_model=PolicySummary,
             dependencies=[Depends(require_auth)])
    async def get_policy(policy_id: str, request: Request) -> PolicySummary:
        registry: PolicyRegistry = request.app.state.registry
        if policy_id not in registry:
            raise HTTPException(status_code=404, detail=f"unknown policy: {policy_id}")
        return registry.get(policy_id).to_summary()

    return app


def _ensure_policy(request: Request, policy_id: str) -> None:
    if policy_id not in request.app.state.registry:
        raise HTTPException(status_code=404, detail=f"unknown policy: {policy_id}")


def _log_verdict(settings: Settings, direction: str, text: str, resp: Any) -> None:
    if not settings.log_verdicts:
        return
    record: dict[str, Any] = {
        "event": "verdict",
        "direction": direction,
        "request_id": resp.request_id,
        "policy_id": resp.policy_id,
        "action": resp.action,
        "categories": [c.category for c in resp.categories],
        "judge": {"invoked": resp.judge.invoked, "cached": resp.judge.cached,
                  "error": resp.judge.error, "provider": resp.judge.provider,
                  "model": resp.judge.model},
        "latency_ms": resp.latency_ms,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
    }
    if settings.log_inputs:
        record["text"] = text
    logger.info(json.dumps(record, ensure_ascii=False))


def run() -> None:
    """Console-script entrypoint (`guardrails-server`)."""
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(message)s",
    )
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port,
                log_level=settings.log_level)


if __name__ == "__main__":
    run()
