# Agentic Graph RAG

A retrieval system that answers questions over your documents by combining two
views of the data — a **vector index** for meaning and a **knowledge graph** for
relationships — and letting a **tool-using agent** decide how to search.

It runs fully **local** (Ollama + open models, no API keys) or on **cloud APIs**
(Claude, Gemini, OpenAI, Voyage). Run just the API with `uvicorn`, or the whole
stack — graph database, cache, API, and web UI — with a single `docker compose up`.

---

## Why this exists

Ordinary RAG embeds your documents and returns the chunks closest to your
question. That is great for *"find the passage about X"* and useless for
*"how are X and Y connected?"* — because closeness in meaning is not the same as
a relationship in the world.

So this project stores your data **twice**:

| Representation | Answers questions like | How |
| --- | --- | --- |
| Vector index | "find text about warehouse robots" | embed chunks, nearest-neighbor search |
| Knowledge graph | "who founded the company that makes the Pallet Pilot?" | entities + typed relationships |

An agent reads your question and chooses the right tool — vector search, graph
traversal, keyword lookup, or a combination. That is the "agentic" part: it is
not a fixed pipeline, it *reasons about how to retrieve*.

---

## How it works (the 60-second version)

**Ingesting a document:**

```
file ──▶ load ──▶ chunk ──▶ embed ──▶ vector index
   (PDF/Word/    │                        │
    HTML/CSV/    └──▶ LLM extracts ──▶ knowledge graph ──▶ resolve duplicates
    text/image)       entities +       (entities linked to      + summarize
                      relationships     the chunks that           communities
                                        mention them)
```

Images and scanned PDFs are read by a small vision model (**OCR**) before
chunking, so a picture of text becomes searchable text. A page goes to OCR when
its text layer is thinner than `ocr.min_text_chars` — not merely when it's empty,
because scans usually carry a page number or a scanner header, and "has some
text" is not "was read".

After a document lands, two enrichment passes run: **entity resolution** folds
duplicates the per-chunk extractor couldn't see were the same thing ("Acme" and
"Acme Robotics"), and **community summaries** cluster the graph and describe each
cluster, so the agent can answer whole-corpus questions, not just chunk lookups.

**Answering a question:**

```
question ──▶ agent ──▶ picks tool(s) ──▶ hybrid retrieval ──▶ rerank ──▶ answer + sources
                        │                 (vector ⊕ graph ⊕ keyword,
                        │                  run in parallel, fused with RRF)
                        └── graph_neighbors, expand_subgraph, compare,
                            get_entity, global_search, ...
```

The agent cites the exact chunks it used. Answers come in a **style** you choose
(concise / detailed / technical / ELI5), it can **compare** several subjects side
by side, and for corpus-wide questions ("what are the main themes?") it reads the
**community summaries** instead of hunting for a single passage. The three legs of
hybrid retrieval run concurrently, and the UI shows which tool the agent is
running while you wait.

For the full picture, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## New: two features, built in — safety & observability

This is the head of a three-project series. Its two companions — linked in the
table below — are **separate standalone projects of mine, merged into this repo as
optional, off-by-default features** while each still lives and evolves in its own
repository. Both are **fail-safe**: a stock run is unchanged, and neither can take
the app down.

| Feature | What it does here | Turn it on |
| --- | --- | --- |
| 🛡️ **[Guardrails & Safety Layer](https://github.com/mohamedshahrah/Guardrails-Safety-Layer-for-LLM-Apps)** | Screens every question **before** the agent runs (prompt-injection / jailbreak / off-topic / pasted secrets → refuse without spending a token) and every answer **after** (PII & secret **redaction**, RAG **groundedness**, system-prompt **leak**). | `safety.enabled: true` |
| 🔭 **[llmlens observability](https://github.com/mohamedshahrah/LLMlens)** | Traces every agent run — prompts, latency, tokens, **cost per user**, tool calls, errors — to a self-hosted dashboard with alerting. One `instrument("langchain")` call captures the whole tool-using loop. | `observability.enabled: true` |

The safety layer doesn't just observe — it **enforces and explains**. On the
non-streaming answer path a `block` verdict withholds the answer and a redaction
swaps in the PII-clean text *before* either reaches the browser, and the verdict
rides back on the response so the UI shows the reader *why* an answer was blocked,
flagged, or redacted. The judge behind it is any OpenAI-compatible model you point
it at — `GUARD_LLM_PROVIDER` / `_MODEL` / `_BASE_URL` / `_API_KEY`, from a hosted
Gemini/DeepSeek to a local Ollama, or `mock` for an offline, key-free run.

What lives here under [`integrations/`](integrations/) is a **vendored copy of
each repo linked above** — merged in so the whole system builds and ships as one
project — and the glue that wires them into the RAG is in `src/graphrag/safety/`
and `src/graphrag/observability/`. Both run in the **same compose project** as the
RAG stack — one command brings up everything:

```bash
make deploy          # RAG + Guardrails + llmlens, one stack, one command
# or lighter:
make up-features     # RAG + Guardrails only (skips the heavier llmlens platform)
```

**How they're connected, end to end → [`docs/INTEGRATIONS.md`](docs/INTEGRATIONS.md).**

---

## Grounded answering — it only speaks from your documents

A RAG system's worst failure isn't a bad retrieval — it's a confident answer the
documents never supported, with a citation bolted on for credibility. Ask a stock
agent *"what's 55 × 88?"* or *"write me a web scraper"* and it will happily answer
from its training data and cite your PDF anyway. This one is **closed-domain**: it
answers from your corpus or it refuses — it never falls back on the model's
general knowledge.

Two layers enforce that, one deterministic and one in the prompt.

**A relevance gate, before the model runs.** Every question is first probed
against the knowledge base; if the best chunk's rerank score is below
`retrieval.min_relevance`, the request is refused on the spot — no model call, no
tokens spent, no fabricated citation. The threshold is a real number calibrated to
your corpus, not a guess. Measured on a set of software-engineering notes with
Cohere `rerank-v4`:

| Question | Top score | |
| --- | --- | --- |
| "use case diagram and actors" — *in the docs* | 0.89 | ✅ answered |
| "software requirements engineering" — *in the docs* | 0.70 | ✅ answered |
| "write me a python web scraper" — *adjacent, not in the docs* | 0.49 | ⛔ refused |
| "what is 55 × 88?" | 0.17 | ⛔ refused |
| "weather forecast tomorrow" | 0.20 | ⛔ refused |

In-topic questions cluster at 0.70–0.89 and off-topic at 0.11–0.20, so
`min_relevance: 0.5` separates them with margin to spare. Raise it to be stricter,
lower it if oblique-but-valid questions get turned away, or set `0` to disable the
gate. It costs one extra retrieval per question and guards `/query` and `/compare`
alike.

**A strict closed-domain prompt, as the model's own rule.** The agent is told it
has *no outside knowledge*: it must call a tool before answering, every claim must
trace to a retrieved chunk it can cite, it may never invent a citation, and when
the documents don't contain the answer it emits one fixed refusal instead of
improvising. This catches the borderline cases the numeric gate lets through; the
safety layer's **groundedness check** blocks an ungrounded answer as the final
backstop.

The net effect: a question your documents can answer comes back grounded and
cited; anything else comes back as an honest *"I can't find that in the knowledge
base"* — never a plausible invention.

---

## Quickstart

### Option A — one command (Docker)

Brings up Neo4j, **Postgres**, Redis, the API, the web UI, and a **TLS reverse
proxy** together. Per-container RAM/CPU caps are set in `.env` (`*_MEM_LIMIT` /
`*_CPU_LIMIT`); the defaults target a 4 vCPU / 12 GB VPS.

```bash
git clone <your-repo-url> agentic-graph-rag && cd agentic-graph-rag
cp .env.example .env          # set the Neo4j + Postgres passwords and your API keys
make setup PROFILE=production # or: PROFILE=local  (fully local, no keys, no accounts)
make up
docker compose exec api alembic upgrade head   # create the account tables
```

The proxy only starts once the API reports healthy (compose waits on a
healthcheck), so when it's up, the whole stack is up. Then open:

- **Web UI** — http://localhost
- **Admin console** — http://localhost/admin  (needs an account with the admin role)
- **API docs (Swagger, test any endpoint here)** — http://localhost:8000/docs
- **Neo4j browser** — http://localhost:7474

Sign up with your email, enter the 6-digit code, and you're in. Without a
`RESEND_API_KEY` / `BREVO_API_KEY` the code is written to the server log instead
of sent — fine for a first run:

```bash
docker compose logs api | grep "code is"
```

To claim the admin panel, set `GRAPHRAG_ADMIN_EMAIL` to the address you signed
up with and restart, or promote it directly:

```bash
docker compose exec api graphrag promote-admin you@example.com
```

**The whole workflow lives in the browser:** drag-and-drop document upload with
per-file ingest progress, streaming chat that shows the **sources** behind every
answer, an account page with your usage against your limits, and — for admins —
users, limits, usage charts and per-tenant graph inspection.

> **Profiles decide whether authentication is on.** `local` and `api` are
> development profiles: auth is **off**, and any caller can act as any user via
> the `X-User-Id` header. Use `production` for anything reachable from a
> network. The API logs a warning at startup when auth is disabled.

**Switch the LLM between local and API with one line.** Set `GRAPHRAG_LLM` in
`.env` and restart the API — only the chat model changes, so your ingested
corpus keeps working (switching *profiles* also swaps the embedder, which
means re-ingesting):

```bash
GRAPHRAG_LLM=ollama:gemma4:e4b-it-q4_K_M   # ← local
GRAPHRAG_LLM=gemini:gemini-3.5-flash       # ← API
docker compose up -d api                   # apply
```

See [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md#the-one-change-llm-toggle).

Ingest the sample corpus and ask a question:

```bash
docker compose exec api graphrag ingest data/sample.md
# then ask in the web UI: "How are Acme Robotics and Riverside University connected?"
```

Fully local? The `local` profile talks to **the Ollama already running on your
host** (`host.docker.internal:11434`), so it reuses models you've pulled instead
of downloading them again inside a container. Pull these:

```bash
ollama pull gemma4:e4b-it-q4_K_M   # chat + reranking + graph extraction
ollama pull gemma3:4b              # OCR (reads images; see note below)
ollama pull bge-m3                 # embeddings
make setup PROFILE=local && make up
```

No Ollama on the host? Bring up the bundled one instead — `docker compose
--profile local up -d ollama`, set `GRAPHRAG_OLLAMA_BASE_URL_INTERNAL=http://ollama:11434`,
and pull the models inside that container.

**Why two models:** OCR and reasoning have different requirements, and a model
that claims both doesn't always deliver both. `gemma4:e4b-it-q4_K_M` reports
`vision` in `ollama show` but ignores attached images, so OCR uses `gemma3:4b`.
Extraction stays on gemma4 — gemma3 is small enough that it fills the graph with
document-local labels. Every role is a separate config key, so pick per role and
verify rather than trust the capability list. See
[`docs/PROVIDERS.md`](docs/PROVIDERS.md).

### Option B — just the API (no Docker)

You need a Neo4j and a Redis reachable from your machine, plus Postgres if you
want accounts (`docker compose up neo4j redis postgres`).

```bash
pip install -e ".[dev,extras]"   # extras = Cohere/Voyage providers the cloud profiles use
cp .env.example .env          # point GRAPHRAG_NEO4J_* / _REDIS_URL / _DATABASE_URL at your services
alembic upgrade head          # only needed for the production profile
make serve PROFILE=local      # uvicorn on :8000, docs at /docs
graphrag ingest data/sample.md
graphrag query "How are Acme Robotics and Riverside University connected?"
```

---

## The API

With the `production` profile every request carries a **session cookie** (the
web UI) or an **API key** (`Authorization: Bearer grk_…`, for scripts). Each
account gets a fully isolated knowledge base — its own Neo4j corpus and its own
DuckDB vector file — while the heavy models are loaded once and shared across
everyone, so adding users costs almost no extra memory.

Within an account, **shelves** split that knowledge base by subject. A maths
textbook and a programming manual get their own corpus each, so their entities
and community summaries never merge — upload to a shelf, and a question searches
that shelf alone. Each shelf also remembers which of the ten **job presets**
(Study, Finance, Legal, Research, Medical, Code, Business, Writing, Teaching,
Summarize) it opens with. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) →
Shelves.

Development profiles have auth off and fall back to the `X-User-Id` header. See
[`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) → Multi-user.

```bash
# Sign up, verify, and use the session — or mint a key and skip the cookie:
graphrag apikey you@example.com
curl -H "Authorization: Bearer grk_…" -X POST localhost:8000/query \
     -H 'Content-Type: application/json' -d '{"question":"...","stream":false}'
```

| Method | Path | What it does |
| --- | --- | --- |
| `POST` | `/auth/signup` · `/auth/verify` · `/auth/login` · `/auth/logout` | Create an account, confirm the emailed code, sign in and out. |
| `GET`  | `/auth/me` · `/auth/limits` | Who am I; my allowances and what I've used. |
| `GET`/`POST`/`DELETE` | `/auth/keys` | Personal API keys (shown once). |
| `GET`/`POST`/`PATCH`/`DELETE` | `/threads` · `/threads/{id}` · `/threads/{id}/messages` | Conversations and their transcripts. |
| `GET`/`POST`/`PATCH`/`DELETE` | `/shelves` · `/shelves/{id}` | Subjects: a separate knowledge base per shelf. Deleting one takes its documents and graph with it. |
| `GET`  | `/shelves/presets` | The ten job presets the composer offers. |
| `GET`  | `/admin/users` · `/admin/users/{id}` | List, search and inspect accounts. |
| `PATCH`/`DELETE` | `/admin/users/{id}` | Suspend, change role, or purge everything they own. |
| `GET`/`PUT` | `/admin/limits` · `/admin/users/{id}/limits` · `/admin/limits/bulk` | Limits: global defaults, per user, or applied to everyone. |
| `GET`  | `/admin/usage` · `/admin/system` · `/admin/audit` | Usage over time, service status, admin action log. |
| `GET`  | `/admin/users/{id}/graph` · `/graph/sample` | A tenant's knowledge-graph stats and a slice to visualize. |
| `POST` | `/query` | Ask a question. Streams the answer (SSE) by default; returns sources used. Takes `preset` (the job) and, on a new thread, `shelf_id`. |
| `POST` | `/compare` | Side-by-side comparison of several subjects. |
| `POST` | `/search` | Raw hybrid retrieval, no LLM — see exactly what the retriever returns. |
| `POST` | `/ingest` | Ingest a server-side path (under `data/`) **or an http(s) URL** (background job). |
| `POST` | `/ingest/upload` | Upload and ingest a file. |
| `GET`  | `/ingest/{job_id}` | Check ingest progress. |
| `GET`  | `/ingest/files` | List your uploaded files and slots used. `?shelf_id=` narrows to one shelf. |
| `DELETE` | `/ingest/files/{file_id}` | Delete a file, its chunks and orphaned entities, and free its slot. |
| `GET`  | `/usage` | Per-user token usage (admin-gated). |
| `GET`  | `/health` · `/ready` · `/metrics` | Liveness / readiness / Prometheus metrics. |

Every endpoint is testable interactively at **`/docs`** — that page is generated
by FastAPI, so "write and test a request" needs no extra tool.

---

## Configuration in one breath

Settings are layered: `configs/default.yaml` → `configs/<profile>.yaml` → `.env`.
Three ready-made profiles ship with the repo:

- **`production.yaml`** — the deployment: accounts and limits **on**, Gemini for
  chat (DeepSeek and Qwen selectable), Cohere `embed-v4.0` + `rerank-v4.0-fast`,
  DuckDB vectors, Postgres for accounts and agent memory. No torch anywhere.
- **`local.yaml`** — everything on your host's Ollama, no keys and no model
  weights downloaded: gemma4 for chat/rerank/extraction, gemma3 for OCR, `bge-m3`
  for embeddings. Auth off. (`local-gemma.yaml` swaps in a larger Gemma 4.)
- **`api.yaml`** — Claude (`claude-opus-4-8`), Voyage embeddings, Gemini OCR.
  Auth off.

Every model role is its own key — `llm`, `ingestion.llm`, `ocr.vision_llm`,
`retrieval.rerank`, `embeddings` — so you can run a different model per role, or
the same one everywhere. See [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) and
[`docs/PROVIDERS.md`](docs/PROVIDERS.md). The embedding block exposes fine-grained
control (device, batch size, normalization, prefixes, Matryoshka dimensions,
caching).

**On a small GPU, `num_ctx` matters more than the model you pick** — twice over.

*It decides whether the model fits.* Ollama sizes its KV cache from the context
length, and that cache — not the weights — is usually what spills a model onto
the CPU: gemma4 at Ollama's default context is 9.9 GB and lands 71% on CPU, but
3.3 GB and 100% on GPU at `num_ctx: 8192`. `ollama ps` shows the split.

*And it decides how many models you load.* **Ollama keys a loaded runner by
(model, options), so the same model at two context sizes is two runners.** Give
chat 8192 and reranking 2048 and you haven't saved memory — you've told a 6 GB
card to hold two copies, so it evicts one to rerank and the other to answer,
twice per query. Measured: **33.4s split across two runners, 5.5s sharing one.**
That's why the local profile gives chat, reranking and extraction the *identical*
`num_ctx`, even where a smaller one would do — 4096 and 8192 are both 3.3 GB
anyway (16384 is the cliff), so the smaller number buys nothing and costs a swap.

The rule: **pick one context per model and use it everywhere that model appears.**
Different *models* (gemma3 for OCR, gemma4 for the rest) will still swap — that's
unavoidable on a small card — but the same model should never swap with itself.

---

## Which model runs what

Five roles, five independent config keys. The `local` profile fills them like
this — the choices aren't arbitrary, and the reasons generalise:

| Role | Config key | Default (`local`) | Why this one |
| --- | --- | --- | --- |
| **Chat / agent** | `llm.model` | `gemma4:e4b-it-q4_K_M` | Needs reliable tool calling; runs on every question |
| **Graph extraction** | `ingestion.llm.model` | `gemma4:e4b-it-q4_K_M` | Must resist naming notation; small models can't |
| **Reranking** | `retrieval.rerank.model` | `gemma4:e4b-it-q4_K_M` | Any model that returns a bare number |
| **OCR** | `ocr.vision_llm.model` | `gemma3:4b` | Needs vision that *works* — gemma4's doesn't |
| **Embeddings** | `embeddings.model` | `bge-m3:latest` | 1024-dim, multilingual, strong retrieval |

### Swapping a model

Change the key and restart — no code, no rebuild:

```yaml
llm:
  model: qwen2.5:7b-instruct     # ollama pull it first
  extra: { num_ctx: 8192 }       # then check `ollama ps` — see below
```

**Chat** — any Ollama model listing `tools` in `ollama show` (the agent binds
tools; without them it can't retrieve at all), or an API provider (`anthropic`,
`openai`, `gemini`, `deepseek`, `qwen`, `deepinfra`). Note `gemma3:4b` has **no**
`tools` — fine for OCR, useless as the agent. Every role also takes a
`fallbacks` list, so one dead key degrades the service instead of stopping it —
see [docs/PROVIDERS.md](docs/PROVIDERS.md).

**Extraction** — needs to follow instructions well enough to skip notation, which
is a real bar: on notes full of transition tables, gemma3:4b filled the graph with
state names (`Q0`..`Q16`) while gemma4 on the same prompt emitted none. If graph
expansion looks like noise, this is the knob.

**Reranking** — the least fussy: it only has to answer with a number. Any chat
model, `cross_encoder` (faster and better, needs the `local-models` extra),
`cohere` / `voyage`, or `none`. Worth knowing before you set it to `none`:
`retrieval.min_relevance` refuses off-topic questions using the reranker's
score, so turning reranking off also turns that gate off.

### A bigger model that doesn't fit is a slower model

Reach for a larger model and you'll usually get a worse system, not a better one.
The moment weights + KV cache exceed VRAM, Ollama runs the overflow on CPU and
the quality gain is spent paying for it. Measured on a 6 GB card (RTX 2060),
same prompt, both warm:

| Chat model | VRAM @ ctx 8192 | On GPU | Speed |
| --- | --- | --- | --- |
| `gemma4:e4b-it-q4_K_M` (8B) | 3.3 GB | **100%** | **41.2 tok/s** |
| `qwen2.5:7b-instruct` (7B) | 5.4 GB | 79% | 27.7 tok/s |

qwen2.5 is the stronger model and its tool calling works — but at 5.4 GB it never
fits alongside the ~1 GB the desktop already holds, so a fifth of it runs on CPU
and it ends up **~50% slower**. No context helps: it's 82% on GPU even at 4096.

So size the model to the card, not the benchmark. `ollama ps` is the whole test —
anything under `100% GPU` means you're paying for weights you can't use. On a
larger card the same comparison flips, which is exactly why this is a config key
and not a hardcoded default.

**OCR** — must be a *vision* model, and the capability list lies. `gemma3:4b`
works; `gemma4:e4b-it-q4_K_M` advertises `vision` and silently ignores images.
`gemini` + `gemini-2.5-flash` is the cloud option, and `ocr.engine: tesseract`
skips models entirely (good for clean printed text, poor for handwriting).

**Embeddings** — `bge-m3:latest` (1024), `nomic-embed-text` (768),
`mxbai-embed-large` (1024), or an API provider. Two catches: point
`embeddings.tokenizer` at the matching HF repo when the model name is an Ollama
tag (Ollama tags aren't HF repo ids, and chunk sizing silently degrades without
it), and **changing dimensions means re-ingesting** — the Neo4j vector index is
created with a fixed width, so a 768-dim model against a 1024-dim index will not
work. Drop the index and re-ingest when you switch.

### Test the model, don't trust the tag

`ollama show <model>` lists capabilities, and it's worth reading — it reliably
tells you what's ruled *out* (no `tools` means it can't be the agent; no
`embedding` means it can't embed). It does not tell you what works:

- `gemma4:e4b-it-q4_K_M` lists `vision`, then answers "please provide the image".
- A GGUF named `Qwen3-VL-Reranker` reports **no** `vision` at all despite the
  "VL", and scores an irrelevant document 10/10 — it reranks nothing.

Both failures are silent: ingest still reports `status=done`, queries still
return an answer. **Swap a model, then check one page, one chunk, one query**
before trusting it.

---

## Project layout

The shape follows the two things the system does: **put documents in** and **get
answers out**. Read it in that order and it explains itself.

```
src/graphrag/
├── core/           Domain vocabulary. Everything else imports FROM here.
│   ├── types.py      Document · Chunk · Entity · Relation · RetrievedChunk
│   ├── errors.py     ConfigError · ProviderError · IngestionError · StorageError
│   └── logging.py    structlog setup (structured key=value logs)
│
├── config/         Layered settings: default.yaml < <profile>.yaml < env
│   ├── settings.py   Typed models. Every knob in the system is a field here.
│   └── loader.py     Deep-merges the YAML layers, reads secrets from env
│
├── ─────────────── INGEST SIDE: file ──▶ graph + vectors ───────────────
│
├── ocr/            Picture of text ──▶ text
│   ├── vision_llm.py Sends the page image to a vision model
│   └── tesseract.py  Offline fallback, no model needed
│
├── ingestion/      The ingest half, in pipeline order
│   ├── loaders/      PDF · Word · HTML · CSV · text · image ──▶ Document (OCR on scans)
│   ├── chunking/     Document ──▶ Chunks. Three strategies:
│   │                   token     fixed windows, exact
│   │                   recursive split on structure, then size  (default)
│   │                   semantic  split where meaning shifts (uses the embedder)
│   ├── extraction/   Chunk ──▶ Entities + Relations, via an LLM
│   └── enrich.py     Post-ingest: entity resolution + community summaries
│
├── embeddings/     Text ──▶ vectors. One `Embedder` interface, several backends
│   ├── ollama.py     Reuses a model you've pulled (no weights downloaded)
│   ├── sentence_transformers.py  In-process, full control (optional extra)
│   ├── api_providers.py          OpenAI · Gemini · Voyage · Cohere
│   └── cache.py      Redis cache keyed by (model, text) — re-ingest is cheap
│
├── ─────────────── QUERY SIDE: question ──▶ answer ─────────────────────
│
├── retrieval/      Finding the right chunks. The interesting part.
│   ├── vector.py           Nearest-neighbour by meaning
│   ├── graph_augmented.py  Follows relationships out from matched entities
│   ├── hybrid.py           Runs several retrievers and combines them
│   ├── fusion.py           Reciprocal Rank Fusion — merges ranked lists fairly
│   └── reranker.py         Re-scores candidates (candidate_k ──▶ top_k)
│
├── agent/          Decides HOW to retrieve — this is the "agentic" part
│   ├── graph.py      LangGraph loop: think ──▶ call tool ──▶ look ──▶ repeat
│   │                   (run/arun/astream_events — sync CLI, async API, streaming)
│   ├── tools.py      What the agent may call: hybrid_search, graph_neighbors,
│   │                   global_search, …
│   ├── prompts.py    System prompts that steer the loop
│   └── styles.py     concise / detailed / technical / eli5
│
├── ─────────────── PLUMBING ────────────────────────────────────────────
│
├── llm/            factory.py (a chat model for any provider) · registry.py
│                     (which models a request may ask for)
├── storage/        GraphStore + VectorStore interfaces ──▶ swappable adapters
│   └── vector/       duckdb_store (one DB file per user) · neo4j_vector ·
│                     local_store (numpy files)
├── pipelines/      Wires the above into `ingest` and `query` flows
│
├── ─────────────── ACCOUNTS & CONTROL ─────────────────────────────────────
│
├── db/             SQLAlchemy models + engine — the Postgres system of record
├── accounts/       Signup · email codes · sessions · API keys · purge
├── limits/         Per-user quotas: what they are, and enforcing them
├── usage/          Recording what was used, for meters and admin charts
│
├── jobs.py         Ingest job status, persisted in Redis
├── worker.py       Arq worker — optional; ingest runs in-process by default
├── auth.py         The API-key format (SHA-256 hashed, shown once)
├── __main__.py     The `graphrag` CLI
├── api/            FastAPI: routers · SSE streaming · deps · /metrics
└── container.py    Composition root: reads config, builds everything, once

migrations/         Alembic — the Postgres schema, versioned
frontend/           React + Vite: the chat app (chat · documents · account)
admin/              React + Vite: the admin console, a separate app at /admin
configs/            The YAML profiles — default · production · local · api
docker/             API image + proxy image (builds both apps) + Caddyfile
scripts/eval.py     Score retrieval + answers against data/eval/qa.yaml (make eval)
tests/unit/         Fast, no services needed. Start here to learn the codebase.
tests/integration/  Against real Postgres — accounts, limits, admin (see below)
```

### How a request moves through it

**Ingesting** `report.pdf`:

```
api/routers/ingest.py   checks the caller's quota, stores the file, queues a job
        │                 (in-process by default — the DuckDB vector store needs
        │                  one owner per file; GRAPHRAG_USE_WORKER hands it to
        │                  worker.py instead)
pipelines/ingest.py     drives the rest:
  loaders/pdf.py          PDF ──▶ text   (ocr/ if a page is a scan)
  chunking/               text ──▶ chunks
  embeddings/             chunks ──▶ vectors ──▶ storage/vector  (the user's DuckDB file)
  extraction/             chunks ──▶ entities + relations ──▶ storage/graph
```

**Answering** *"How are Acme and Riverside connected?"*:

```
api/deps.py             resolves the caller: session cookie ──▶ API key ──▶ (dev header)
limits/                 is this message within their quota? 429 if not
        │
api/routers/query.py    receives it, opens an SSE stream
        │
pipelines/query.py      starts an agent session
        │
agent/graph.py          the loop: which tool would answer this?
        │                   └── calls a tool from agent/tools.py
retrieval/hybrid.py         vector + graph + keyword, in parallel
retrieval/fusion.py         merge the ranked lists (RRF)
retrieval/reranker.py       score candidate_k ──▶ keep top_k
        │
agent/graph.py          reads the chunks, answers (or calls another tool)
        │
api/streaming.py        tokens ──▶ browser, then the sources behind them
```

### Why it's shaped this way

**`core/` depends on nothing.** Types and errors sit at the bottom, so no two
modules ever need to import each other to agree on what a `Chunk` is.

**Every provider hides behind an interface.** `Embedder`, `GraphStore`,
`VectorStore`, `Reranker`, `OCREngine` — each is a small abstract class with a
concrete implementation per backend. Swapping Ollama for OpenAI is one new class
and one config line; nothing around it changes. That's why every model in this
README is a config key and not an `if` statement somewhere.

**`container.py` is the only place that knows how the pieces fit.** It reads the
config and builds the object graph once. Heavy things (models, drivers) are
`@cached_property`, so they load on first use and are shared by every user. That's
why adding users costs almost no memory — only the small per-tenant wrappers are
duplicated.

**Ingest runs off the request, not off the process.** The slow work (OCR,
embedding, extraction) runs as a background task under a semaphore, so a large
upload never blocks a chat stream and two uploads can't fight over the same two
cores. It stays *in* the API process because each user's DuckDB vector file
needs a single owner; set `GRAPHRAG_USE_WORKER=1` (with the Neo4j vector
provider) to hand it to a separately capped worker container instead.

**Reading it for the first time?** `core/types.py` (the vocabulary) ──▶
`configs/default.yaml` (every knob, commented) ──▶ `container.py` (how it's
wired) ──▶ `pipelines/` (the two flows end to end). `tests/unit/` runs in under a
second with no Neo4j or Ollama, and each test documents one real failure mode.

---

## Production hardening

This is built to run as a controllable service, not just a demo. What's in place:

- **Real accounts.** Email + password, with a 6-digit code proving the address
  exists before the account activates. Sessions are server-side: the cookie
  carries an opaque token, the database stores only its hash, and it's httpOnly
  + SameSite=Lax with the Secure flag following the request scheme. Suspending
  an account kills its sessions *and* its API keys on the next request — the
  reason sessions aren't stateless tokens. Signup, resend and login answer
  identically for known and unknown addresses so they can't be used to discover
  who has an account.
- **Enforced per-user limits.** Messages per minute and per day, token budgets
  per day and month, document count, per-file and total storage, indexed chunks,
  conversations. A refused request never reaches the model, and the 429 names
  the limit, the ceiling and the reset so the UI can say "12 of 12 today, resets
  in 4 hours". Rate windows live in Redis (fast, and fail *open* so a cache blip
  doesn't lock everyone out); the caps guarding durable resources read Postgres,
  so a deleted file returns its quota automatically.
- **Admin dashboard.** Users with search and status filters, per-user detail
  with usage and knowledge-graph stats, limits edited globally / per user / in
  bulk, usage charts, model availability, system status, and a full purge that
  clears a user from Postgres, Neo4j, DuckDB and disk. Every mutation is
  audited. Gated by an admin role, with `X-Admin-Key` as break-glass for
  bootstrap; **fails closed** when neither is configured.
- **Durable agent memory.** Conversation memory uses a LangGraph checkpointer in
  Postgres (Redis also supported, with redis-stack), so multi-turn context
  survives restarts. The API uses the async saver (its streaming needs it), the
  CLI the sync one — same keyspace, so both see the same threads. If the
  configured backend is unusable the API tries the other one before settling for
  in-process memory, rather than silently losing conversations.
- **Per-user isolation.** Each account gets its own Neo4j corpus *and* its own
  DuckDB vector file, so one user's data is a separate artifact that can be
  backed up, inspected or deleted on its own. Thread and file ids are checked
  against the owner on every access, and a resource you don't own returns 404,
  not 403 — ids can't be probed.
- **Resource controls.** Every container has RAM/CPU caps (`.env`:
  `API_MEM_LIMIT`, `NEO4J_MEM_LIMIT`, `NEO4J_HEAP`, `POSTGRES_MEM_LIMIT`, …),
  defaulting to a set that fits 4 vCPU / 12 GB with headroom, plus CPU-thread
  caps (`OMP_NUM_THREADS`) so local models can't starve the host.
- **Ports that stay shut.** Every container but the proxy binds to `127.0.0.1`,
  so from the network only the proxy exists; `make ports` fails the build if a
  prefix ever goes missing. That binding is the real control — `ufw deny` does
  **not** close a published container port, because Docker's iptables rules are
  evaluated before ufw's. [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) covers
  hardening the host underneath.
- **Confined server-side ingest.** `POST /ingest` with a path is fenced to
  `data/` (an attempt to reach `.env` or any other server file is a 400), and it
  also accepts an http(s) URL, fetched with a size cap. Uploads go through
  `/ingest/upload` regardless. Raising `max_upload_mb` also needs `MAX_UPLOAD_MB`
  in `.env` raised — Caddy rejects oversized bodies before the API sees them.
- **Grounded, closed-domain answers.** The assistant answers only from your
  documents. A relevance gate (`retrieval.min_relevance`) refuses questions the
  corpus doesn't cover before the model runs, a strict system prompt forbids
  outside knowledge and fabricated citations, and the safety layer's groundedness
  check blocks an ungrounded answer as a backstop — so *"what's 55 × 88?"* gets an
  honest refusal, not a confident number with your PDF cited under it.
- **Enforced safety verdicts.** With `safety.enabled`, the guard screens every
  question before the agent and every answer after; on the non-streaming path a
  block withholds the answer and a redaction swaps in the PII-clean text before
  the browser sees it, with the verdict surfaced in the UI. The judge is any
  OpenAI-compatible model (`GUARD_LLM_*`), or `mock` for an offline run.
- **Prompt-injection hardening.** Retrieved documents are attacker-controlled, so
  everything a tool returns is sanitized (control characters, chat-template
  special tokens) and wrapped in `<untrusted_data>` markers the system prompt
  defines as evidence, never instructions. A chunk can't forge the closing
  marker to escape its envelope. Tool output is capped, and nothing
  user-supplied reaches a prompt except the question: the answer style is
  enum-clamped, the model is validated against an allowlist, and the thread id
  is ownership-checked.
- **Observability.** Prometheus metrics at `/metrics` (request counts + latency
  histograms, labeled by route template so cardinality stays bounded), durable
  usage events in Postgres behind the admin charts, and structured request logs.
  A `make eval` target scores retrieval and answers against a golden set
  (`data/eval/qa.yaml`) so retrieval changes are measured, not guessed.
- **TLS + reverse proxy.** A single **Caddy** container is the public entrypoint
  (80/443): it terminates TLS, serves the chat app at `/` and the admin console
  at `/admin` from static files built into its image, and routes `/api/*` to the
  API with buffering off so SSE streams. One origin for everything, so CORS isn't
  needed and one session cookie covers both apps.
- **Accounts you can actually operate.** Password reset by emailed code, change
  password, a "signed-in devices" list you can revoke individually, per-IP limits
  on every unauthenticated auth endpoint, and a failed-login lockout with
  exponential backoff (`make unlock EMAIL=…` is the break-glass). Admins can
  invite users, force a password reset, and clear a lockout.

### Exposing it over HTTPS

The proxy handles certificates automatically — you just set `SITE_ADDRESS`:

| `SITE_ADDRESS` | Result |
|---|---|
| `:80` (default) | Plain HTTP on port 80 — local dev |
| `localhost` | HTTPS via Caddy's local self-signed CA |
| `rag.example.com` | **HTTPS via automatic Let's Encrypt** (also set `TLS_EMAIL`) |

For a public domain: point its DNS A record at the host, set `SITE_ADDRESS` and
`TLS_EMAIL` in `.env`, and `docker compose up` — Caddy provisions and renews the
cert on :443 and redirects HTTP→HTTPS. For a real deployment, also remove the
host `ports` from the `api` service so only the proxy is reachable.

Still on the roadmap: backups and restore drills for Postgres + the per-user
DuckDB files, distributed tracing, and the quantized ANN backends sketched in
[`docs/OPTIMIZATION-NOTES.md`](docs/OPTIMIZATION-NOTES.md) — the exact cosine
scan DuckDB uses is honest and fast at the per-user chunk ceiling, but it is a
scan.

## Development

New here? [`CONTRIBUTING.md`](CONTRIBUTING.md) covers what this repo
deliberately does not carry (credentials, one deployment's operational detail)
and how to supply your own, plus the easiest places to start — adding an answer
preset needs a single Markdown file and no Python.

```bash
make install     # editable install with dev + extra providers
make test        # unit tests (fast, no services)
make eval        # score retrieval + answers vs the golden set (needs the stack up)
make lint        # ruff + mypy
make fmt         # auto-format
```

Integration tests need a real Postgres. They create and drop the schema
themselves, so they refuse any database whose name doesn't look disposable:

```bash
docker compose up -d postgres
docker compose exec postgres psql -U graphrag -d graphrag -c "CREATE DATABASE graphrag_test OWNER graphrag;"
GRAPHRAG_TEST_DATABASE_URL=postgresql://graphrag:change-me@localhost:5432/graphrag_test \
  pytest -m integration -q
```

The two web apps are separate builds. Both proxy `/api` to the backend on
:8000, so run the API alongside them:

```bash
cd frontend && npm install && npm run dev   # chat app,      Vite on :5173
cd admin    && npm install && npm run dev   # admin console, Vite on :5174
npm run build                               # tsc + production bundle (in either)
make web                                    # or build both at once
```

In production neither dev server is involved: `make up` builds both into the
proxy image, which serves them at `/` and `/admin`.

Design decisions and the reasoning behind them live in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## License

MIT — see [LICENSE](LICENSE).
