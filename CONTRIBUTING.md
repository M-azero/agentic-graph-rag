# Contributing

Thanks for looking. This file covers the two things a clone can't tell you: what
is deliberately **not** in the repo and how to supply your own, and what the
project expects from a change.

---

## What this repo does not contain

The code here is complete and runnable. What is missing is anything that
belongs to a *specific running copy* of it — credentials, and the operational
detail of one deployed box. Nothing below is a gap you need to fill in to read
the code; it is the list of what you provide to run your own instance.

| Not in the repo | What it is | How to get your own |
|---|---|---|
| `.env` | Every credential and service URL | `cp .env.example .env`, then fill it in — the example file documents each key inline |
| `data/` contents | Uploaded documents, DuckDB vector files, ingest jobs | Created on first run. `data/sample.md` and `data/eval/` **are** tracked, so the quickstart works on a fresh clone |
| Host-hardening material | SSH policy, firewall rules and audit notes for one server | Deployment-specific and deliberately withheld. [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) covers everything you need to deploy safely, including the two settings that decide whether you are exposed |
| TLS certificates | Issued by the proxy | Caddy obtains them automatically — Let's Encrypt for a real domain, self-signed for `localhost`. They live in a Docker volume, never in the repo |

### Filling in `.env`

Only set keys for the providers you actually enable. The shipped `production`
profile needs three things; everything else has a working default:

- **`GRAPHRAG_NEO4J_PASSWORD`** and **`GRAPHRAG_POSTGRES_PASSWORD`** — pick your
  own, anything. These are yours; nothing outside your machine ever sees them.
- **At least one model provider key.** The profile's chat model is DeepSeek with
  a fallback chain through DeepInfra and Cohere, and embeddings are Cohere. Any
  one working link is enough to start — see
  [`docs/PROVIDERS.md`](docs/PROVIDERS.md) for the full matrix.

If you want to run with **no API keys at all**, use `make setup PROFILE=local`.
That switches chat, embeddings and OCR to models served by Ollama on your own
machine. It is slower and needs the hardware, but it is fully offline.

> **`.env` is ignored, and so are `*.pem`, `*.key`, `*.sql`, `*.duckdb`, `*~`
> and friends.** If you add a new kind of credential or dump, add its shape to
> `.gitignore` in the same commit. The rule that has kept this repo clean is
> that the ignore list describes *shapes*, not the one filename someone
> remembered.

---

## Getting it running

**With Docker — the whole stack, one command:**

```bash
cp .env.example .env          # set the two passwords and a provider key
make setup PROFILE=production # or PROFILE=local for no-keys, fully offline
make up                       # brings everything up and applies migrations
```

Then open `http://localhost` (or your domain). `make logs` tails everything;
`make down` stops it.

**Without Docker — for working on the Python:**

```bash
make install                  # editable install with dev extras
make setup PROFILE=api
docker compose up -d neo4j redis postgres   # the datastores only
make migrate
make serve                    # API on :8000, with reload
```

The web apps run separately with `npm run dev` in `frontend/` or `admin/`; both
proxy to `localhost:8000`.

**If something doesn't come up**, run the preflight check — it validates the
resolved config against the services actually reachable, from inside the
network where it means something:

```bash
docker compose exec -T api python scripts/preflight.py
```

---

## The dev loop

```bash
make test        # unit tests — fast, no services needed
make lint        # ruff + mypy
make fmt         # auto-fix formatting and imports
make web         # build and type-check both web apps
```

CI runs `ruff check src tests migrations`, the unit tests, both web builds, and
a Caddyfile validation. `mypy` runs but is advisory — the codebase is not fully
typed yet.

Integration tests need a real Postgres and are skipped without one:

```bash
docker compose up -d postgres
GRAPHRAG_TEST_DATABASE_URL=postgresql://graphrag:<pw>@localhost:5432/graphrag_test make test-integration
```

The fixtures drop every table between runs, so they **refuse** any database
whose name doesn't contain `test`, `tmp`, `scratch` or `ci`. That guard exists
because the alternative is losing your accounts table to a stray `export`.

---

## Good places to start

**Add a job preset.** The ten answer presets are Markdown, one file per preset,
in [`prompts/`](prompts/README.md). Adding one is a file plus an enum member —
no second list to update, and `prompts/README.md` states the contract every
preset has to satisfy. This is the easiest useful contribution in the project
and it needs no Python at all.

**Add a model provider.** `src/graphrag/llm/factory.py` is the seam; the
registry and fallback chain pick it up automatically.

**Add a document loader.** `src/graphrag/ingestion/loaders/` — one class,
registered by extension.

**Add a storage backend.** `src/graphrag/storage/` defines `GraphStore` and
`VectorStore`; the container calls the factories per tenant. Retrieval never
touches a backend directly, so a new one is an adapter, not a refactor.

---

## What a change should look like

**Explain the "why", not the "what".** The code says what it does. Comments and
docstrings in this repo exist to record the reasoning that isn't recoverable
from reading it — the constraint that forced a design, the bug a guard prevents,
the thing that was tried and didn't work. If a comment restates the line below
it, delete the comment.

**Cover the failure you fixed.** A bug fix without a test that fails before it
is a bug fix that comes back. Tests live next to the concern they protect:
`tests/unit/` for anything that runs without services, `tests/integration/` for
anything that needs a real database.

**Respect the grounding rules.** The system prompt in
`src/graphrag/agent/prompts.py` is closed-domain and hardened against prompt
injection: the assistant answers only from retrieved documents, cites every
claim, and refuses when the knowledge base doesn't cover the question. Anything
that reaches the model — a preset, a tool description, a tool's output — must
narrow that contract, never widen it. `tests/unit/test_prompt_hardening.py` and
`tests/unit/test_presets.py` enforce it; if you find yourself editing those to
make a change pass, the change is the problem.

**Keep tenant scoping intact.** Every Neo4j read and write is keyed on `corpus`,
and every router scopes by `user.tenant_id` — never by an id the caller
supplied. If you add an endpoint that takes an id, prove ownership before you
use it. `src/graphrag/shelves.py` shows the pattern.

---

## Reporting a security issue

Please don't open a public issue for a vulnerability. Report it privately
through GitHub's **Security → Report a vulnerability** on this repository, and
include what you did, what happened, and what you expected.

Things worth reporting: anything that lets one account reach another's
documents, graph or conversations; anything that gets a request past the auth
or quota layer; any way to make retrieved document content act as an
instruction rather than as data.
