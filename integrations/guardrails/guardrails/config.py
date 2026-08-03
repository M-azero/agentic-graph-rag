"""Process configuration — the *deployment* surface.

Two configuration surfaces exist and they do different jobs:

* **This file / the environment (a local ``.env``)** — *where and how the server runs*:
  which judge **model/provider/key** to use, the bind address, the server auth key,
  cache/log knobs. Copy ``.env.example`` to ``.env`` and edit it. Every field below maps
  to an env var named ``GUARD_`` + the field name (e.g. ``llm_model`` -> ``GUARD_LLM_MODEL``).
* **``policies/*.yaml``** — *what the guard actually does*: scope, categories, thresholds,
  redaction, custom rules. See :mod:`guardrails.policy`.

``get_settings()`` is cached so the app reads the environment once at startup.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

FailMode = Literal["open", "closed", "flag"]


class Settings(BaseSettings):
    """Server + judge configuration (env prefix ``GUARD_``)."""

    model_config = SettingsConfigDict(
        env_prefix="GUARD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Judge model / provider (set these in .env) ──────────────────────────
    # The "which brain does the reasoning" knobs. Pick a preset with GUARD_LLM_PROVIDER,
    # name the model with GUARD_LLM_MODEL, and supply the key with GUARD_LLM_API_KEY
    # (or the preset's native key env var, e.g. ANTHROPIC_API_KEY / OPENAI_API_KEY).
    llm_provider: str = "gemini"
    llm_model: str | None = "gemini-2.5-flash-lite"
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_timeout_s: float = Field(default=10.0, gt=0)
    llm_max_tokens: int = Field(default=1024, gt=0)

    # ── Judge failover chain ────────────────────────────────────────────────
    # Judges tried in order after the primary above, as a comma-separated list
    # of ``provider:model`` (a bare ``provider`` uses that preset's default
    # model, and ``provider:model@https://host/v1`` overrides the base URL):
    #
    #     GUARD_LLM_FALLBACKS=deepseek:deepseek-v4-flash,cohere:command-a-03-2025
    #
    # This matters more than an ordinary fallback list. The guard fails *open*:
    # when no judge answers, every verdict is `judge_unavailable` and requests
    # sail through unscreened with nothing but a log line to say so. One dead
    # provider should not be able to do that, so put the links on DIFFERENT
    # VENDORS — a second model at the same vendor shares the outage and the key.
    #
    # Each link resolves its own key from that preset's native env var
    # (DEEPSEEK_API_KEY, COHERE_API_KEY, ...), never from GUARD_LLM_API_KEY,
    # which belongs to the primary. Links with no key are dropped at startup
    # and named in the log rather than left to 401 on every request.
    llm_fallbacks: str = ""

    # Consecutive failures that take a link out of rotation, and for how long.
    # A refused key answers in ~200ms and paying that on every request adds up;
    # a tripped link is retried once the cooldown expires and rejoins on its
    # first success. Tripped links are still tried as a last resort when every
    # other link is also down — a stale breaker must not be the reason nothing
    # gets screened.
    llm_failover_max_failures: int = Field(default=2, ge=1)
    llm_failover_cooldown_s: float = Field(default=300.0, ge=0)

    # Wall-clock ceiling for one verdict across the WHOLE chain, including the
    # repair retry. Without it the budget is per-link, so a 3-link chain can
    # spend 3x llm_timeout_s and blow past the caller's own timeout — at which
    # point the caller has already given up and failed open, and the chain is
    # burning tokens to produce a verdict nobody will read. Unset means no
    # overall cap (the historical single-provider behaviour).
    llm_total_timeout_s: float | None = Field(default=None, gt=0)

    @field_validator("llm_total_timeout_s", mode="before")
    @classmethod
    def _blank_means_unset(cls, v: object) -> object:
        """Treat an empty value as "not configured".

        A container orchestrator can pass a variable but cannot omit one, so
        this arrives as ``GUARD_LLM_TOTAL_TIMEOUT_S=""`` whenever it is left
        unset. Without this the process refuses to start on a blank string.
        """
        return None if v in ("", None) else v

    # ── Policy / behaviour ──────────────────────────────────────────────────
    fail_mode: FailMode = "flag"
    policy_dir: str = "./policies"
    default_policy: str = "default"

    # ── Server ──────────────────────────────────────────────────────────────
    api_key: str | None = None
    # Bind to loopback by default: the server is reachable only from this machine and
    # is never exposed on the network. Set GUARD_HOST=0.0.0.0 to expose it (the Docker
    # image does this so the container is reachable via -p).
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    # Interactive API docs (/docs, /redoc, /openapi.json). Off by default so the server
    # exposes nothing but the endpoints it serves. Flip to true for local exploration.
    enable_docs: bool = True

    # ── Verdict cache ───────────────────────────────────────────────────────
    cache_enabled: bool = True
    cache_size: int = Field(default=2048, ge=1)
    cache_ttl_s: int = Field(default=300, ge=0)

    # ── Concurrency ─────────────────────────────────────────────────────────
    max_concurrent_judge: int = Field(default=16, ge=1)

    # ── Logging ─────────────────────────────────────────────────────────────
    log_level: str = "info"
    log_verdicts: bool = True
    log_inputs: bool = False


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide, cached :class:`Settings`."""
    return Settings()
