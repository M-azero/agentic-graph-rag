# Integrated features — how the three projects connect into one

This system is the **head** of a three-project series. The other two are
standalone products with their own GitHub repos, and they stay that way — but
they're also **wired into this RAG app as first-class features**:

| Feature | What it adds to the RAG | Standalone repo | Vendored at |
|---|---|---|---|
| **Guardrails & Safety Layer** | A safety verdict around every answer: screens the user's message *before* the agent runs and the model's answer *after* (PII/secret redaction, groundedness, prompt-leak). | `Guardrails-Safety-Layer-for-LLM-Apps` | `integrations/guardrails/` |
| **llmlens** | LLM observability: every agent run traced — prompts, latency, tokens, **cost per user**, tool calls, errors — in a dashboard, with alerting. | `LLMlens` | `integrations/llmlens/` |

This document is the **"how they're connected"** reference: what glue exists in
this repo, where it hooks in, how to switch each feature on, and how the pieces
run together. For what each feature *is*, read its own README under
`integrations/<name>/README.md`.

---

## The design contract

Three rules govern every line of the integration, so that adding two features to
a working system can't destabilize it:

1. **Off by default.** `safety.enabled` and `observability.enabled` are both
   `false` in `configs/default.yaml`. A stock checkout behaves exactly as before;
   the feature code is dormant.
2. **Fail-safe, never fail-into-the-request.** Neither feature can take the app
   down. The guardrails client resolves *every* error (disabled, unreachable,
   timeout, 5xx, bad JSON) to a verdict — `safety.fail_open` decides allow vs.
   block — and never raises into the query path. The llmlens hook swallows setup
   errors and a down collector just drops spans (the SDK's own guarantee).
3. **Separate repos stay separate.** The two features are **vendored copies**
   (source only — no `.git`, `.venv`, or caches) of their own repos. This RAG
   repo owns the *glue*; the features remain independently developed and
   deployable. See [Keeping them in sync](#keeping-the-vendored-copies-in-sync).

---

## Where each feature hooks in

```
                          ┌─────────────────────────── llmlens ───────────┐
                          │  setup_observability() at API startup          │
                          │  instrument("langchain") → every agent/LLM/    │
                          │  tool span auto-traced;  query_trace(user_id)   │
                          │  roots them per user                            │
                          └───────────────▲────────────────────────────────┘
                                          │ (spans)
  POST /query ─▶ [Guardrails: check_input] ─▶ agent (LangGraph) ─▶ answer
                        │ block?                    │                  │
                        │ └─▶ refusal (no model call, no tokens)       │
                        │                                              ▼
                        └──────────────────── [Guardrails: check_output] ──▶ client
                                              redact / block / groundedness
```

### Guardrails — the glue in this repo

| Piece | File | Role |
|---|---|---|
| HTTP client | `src/graphrag/safety/guardrails.py` | Async `GuardrailsClient` → `POST /v1/guard/input` and `/v1/guard/output`. Normalizes the verdict; fail-open/closed on any error. One shared `httpx.AsyncClient`. |
| Wiring | `src/graphrag/container.py` → `Container.guardrails` | Builds the client once (shared across tenants). `GRAPHRAG_GUARDRAILS_URL` overrides the YAML `base_url`. Opens no socket until the first real check. |
| Enforcement | `src/graphrag/api/routers/query.py` | `/query` and `/compare`: input check before the agent (a block short-circuits — no model, no tokens); output check after (non-streaming can **block or redact**). |
| Streaming | `src/graphrag/api/streaming.py` | `sse_refusal()` for a blocked input; a trailing `safety` SSE event carries the output verdict (monitor mode — see the tradeoff below). |
| Config | `src/graphrag/config/settings.py` → `SafetyCfg` | `enabled`, `base_url`, `policy_id`, `check_input`, `check_output`, `fail_open`, `timeout_s`. |

**The streaming tradeoff.** On the non-streaming path the output guard *enforces*
— it can withhold the answer (`block`) or swap in the redacted text
(`sanitized_output`). On the **streaming** path the tokens have already reached
the client by the time the answer is complete, so the guard runs in **monitor
mode**: it emits a `safety` event with the verdict (which the UI can act on)
rather than pretending it can un-send text. Input-guarding is full enforcement on
both paths, because it happens before a single token is produced.

### llmlens — the glue in this repo

| Piece | File | Role |
|---|---|---|
| Startup hook | `src/graphrag/observability/__init__.py` → `setup_observability()` | Called once in `create_app()`. Configures the SDK and `instrument("langchain")`, which registers a **global** LangChain callback so every agent/LLM/retriever/tool call is traced automatically. |
| Per-user attribution | same file → `query_trace()` | Wraps each request in `pipelines/query.py`. A no-op (`nullcontext`) unless observability is active; when active, roots the auto-traced spans under a `user_id` — the basis for cost-per-user. |
| Config | `src/graphrag/config/settings.py` → `ObservabilityCfg` | `enabled`, `url`, `service`. |
| SDK | `integrations/llmlens/sdk` (installed into the image by `docker/Dockerfile`) | `import llmlens`. Only dependency is httpx, already present. |

Because the RAG agent is a LangChain/LangGraph app, **one** `instrument("langchain")`
call captures the whole tool-using loop — no per-call wiring. This is exactly the
recipe in `integrations/llmlens/examples/instrument_project1.md`.

---

## Activate

The plumbing is inert until you flip the switch in your active config profile.

### 1. Turn the feature(s) on (YAML)

In `configs/<profile>.yaml` (e.g. `production.yaml`, or `default.yaml` for all):

```yaml
safety:
  enabled: true          # screen inputs + outputs
  policy_id: default     # a policies/*.yaml id on the guardrails server
  fail_open: true        # guard down → keep answering (false → refuse)

observability:
  enabled: true          # trace agent runs to llmlens
```

> Config lives in YAML by design (this project keeps *behavior* in YAML and only
> *secrets/URLs* in env). The env vars below point the features at the right
> hosts; the on/off switch is YAML.

### 2. Point them at the services (env / `.env`)

```bash
# Guardrails (integrations/guardrails)
GRAPHRAG_GUARDRAILS_URL=http://localhost:8080     # or http://guardrails:8080 in docker
GRAPHRAG_GUARDRAILS_API_KEY=                       # only if the guard sets GUARD_API_KEY

# llmlens (integrations/llmlens)
LLMLENS_URL=http://localhost:8100                  # its API, remapped off :8000
LLMLENS_API_KEY=sk_...                             # from `llmlens-server create-project`
```

---

## Run it

### Bare (no Docker)

```bash
# 1) Guardrails — offline mock judge, no keys (its own venv or the RAG venv):
cd integrations/guardrails && cp .env.example .env
pip install -e ".[dev]" && guardrails-server         # :8080

# 2) llmlens SDK into the RAG environment (so the agent can be traced):
pip install -e integrations/llmlens/sdk
#    …and the llmlens server stack (its own compose) if you want the dashboard.

# 3) RAG API with both features enabled in config + env above:
make serve PROFILE=api
```

Guardrails on the **mock** judge is fully offline and needs no API key — enough
to see `allow` / `block` / redaction end to end before wiring a real judge.

### Docker — one project, one command

Both features run **in the same compose project as the RAG stack** — no second
project, no second `up`. The overlay `docker-compose.integrations.yml` layers
onto the base file. Two levels:

```bash
# Light — RAG + Guardrails (one small container):
docker compose -f docker-compose.yml -f docker-compose.integrations.yml up -d

# Full — also the llmlens observability platform (behind a profile):
docker compose -f docker-compose.yml -f docker-compose.integrations.yml \
    --profile observability up -d

# …or just:
make deploy            # the full command above + migrations, one shot
make up-features       # the light command
```

Prefer a plain `docker compose up -d` with no flags at all? Put the file list
and profile in `.env` — Compose reads them from there on every command
(`up`, `down`, `logs`, `ps`, `exec`):

```bash
COMPOSE_PATH_SEPARATOR=:
COMPOSE_FILE=docker-compose.yml:docker-compose.integrations.yml
COMPOSE_PROFILES=observability
```

`make deploy` brings up **everything**: Neo4j, Postgres, Redis, the RAG API (with
the llmlens SDK baked in), the UI, the Caddy proxy, `guardrails`, and the
llmlens platform (ClickHouse + its own Redis + api + worker + dashboard). It
stays dormant until you enable `safety` / `observability` in config.

**One Postgres server, not two.** llmlens does not get its own Postgres: it uses
the RAG Postgres server with a separate `llmlens` database, which the
`llmlens-db-init` one-shot creates on first deploy (idempotent, so it also works
against a server whose volume predates the overlay). It *does* get its own Redis
— its Redis is a durable `noeviction` ingest queue while the RAG Redis is an
`allkeys-lru` cache, and one Redis server can only have one eviction policy —
and ClickHouse, the columnar span store nothing else can absorb.

**Why it's one project when llmlens hardcodes the same service names.** llmlens's
containers resolve `api`, `redis`, `clickhouse` — names the RAG stack also uses,
and Compose forbids duplicate service names. So the llmlens services are renamed
`llmlens-*` but given those plain names as **network aliases on an isolated
`llmlens` network**; inside that network they still find each other, while the
RAG services (on the default network) never see them. The shared Postgres joins
the llmlens network under its natural name. `llmlens-api` also joins a small
`telemetry` network shared with the RAG API, which reaches it as
`http://llmlens-api:8000` — deliberately *not* the default network, where `redis`
would resolve to two different containers. No published host ports collide — the
llmlens datastores aren't published at all.

| Published on the host | Port |
|---|---|
| RAG UI · API · Neo4j | 5173 · 8000 · 7474 |
| **llmlens dashboard** | **5273** |
| llmlens API (+ /docs) | 8100 |
| Guardrails (debug) | 8080 |

The llmlens API defaults to `LLMLENS_PROFILE=local` (auth off) so the RAG API can
send traces without a project key. For a hardened deployment, set
`LLMLENS_PROFILE=default`, mint a key (`llmlens-server create-project`), and set
`LLMLENS_API_KEY` on the RAG API.

> **Footprint.** The full stack is ~12 containers. The base RAG stack targets a
> 4 vCPU / 12 GB box and takes 7.6 GB of that in caps; the llmlens platform
> adds ClickHouse (2 GB) + its own Redis + api/worker (Postgres is shared with
> the RAG stack) for another 4.6 GB. The two together overrun a 12 GB box, so
> budget **16 GB** for `make deploy`, or run the light stack
> (`make up-features`) and host llmlens elsewhere.

---

## What a request looks like with both on

1. `POST /query {"question": "..."}` →
2. **Guardrails input check.** Prompt injection / jailbreak / off-topic / pasted
   secrets → `block` returns a refusal immediately (no agent, no tokens spent).
3. The **LangGraph agent** runs — hybrid retrieval, tools, answer. Every step is
   an **llmlens span**, rooted under this user's `user_id`.
4. **Guardrails output check.** Groundedness against the retrieved chunks, PII/
   secret redaction, system-prompt leak. Non-streaming: block or swap in the
   redacted text. Streaming: a `safety` event carries the verdict.
5. The answer (or refusal, or redacted text) returns; the trace lands in llmlens
   with tokens, latency and cost attributed to the user.

---

## Keeping the vendored copies in sync

The copies under `integrations/` are source snapshots of the two repos (secrets
and build caches stripped). Two ways to keep them current — pick one:

**A. Re-vendor (what's in place).** Copy the source again after changes upstream:

```bash
rsync -a --delete \
  --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
  --exclude '.pytest_cache' --exclude '.ruff_cache' --exclude '*.egg-info' \
  --exclude 'node_modules' --exclude '.env' \
  ../llmlens/ integrations/llmlens/
```

**B. Convert to git submodules** (if you'd rather link the exact upstream commit):

```bash
rm -rf integrations/llmlens integrations/guardrails
git submodule add https://github.com/mohamedshahrah/LLMlens.git integrations/llmlens
git submodule add https://github.com/mohamedshahrah/Guardrails-Safety-Layer-for-LLM-Apps.git integrations/guardrails
git submodule update --init --recursive
```

Either way the glue in `src/graphrag/` is unaffected — it depends only on the
two HTTP contracts (`/v1/guard/*`) and the llmlens SDK's public API, not on the
features' internals.

---

## What's verified

- **Guardrails, live end-to-end.** The RAG `GuardrailsClient` against a running
  guardrails server (mock judge): benign input `allow`, prompt-injection `block`
  (deterministic rule), and an answer containing a key + email redacted to
  `[REDACTED:OPENAI_KEY]` / `[REDACTED:EMAIL]` via `sanitized_output`.
- **Guardrails, unit.** `tests/unit/test_safety_guardrails.py` — disabled no-op,
  verdict parsing, fail-open vs. fail-closed, per-direction toggles.
- **llmlens.** `tests/unit/test_observability.py` — disabled setup is inert and
  `query_trace` is a real no-op; enabled setup instruments LangChain and yields a
  live trace context. A span tree emits through the SDK exporter without error.
- **Whole suite.** All 169 existing unit tests still pass; new code is ruff-clean.
- **Compose.** The unified overlay validates with `docker compose config` in
  both modes: light (7 services: RAG + guardrails) and full
  (`--profile observability`, 13 services). The rendered config confirms the
  network aliases (`llmlens-api` answers to `api` only on the `llmlens` network),
  the dual-network bridge for the RAG API to reach it, and that **no published
  host port collides** (the llmlens datastores aren't published).
```
