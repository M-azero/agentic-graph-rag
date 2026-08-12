# Architecture

This document explains how the system is put together and, more importantly,
*why* it is put together that way. Read it top to bottom once; after that the code
should be self-explanatory.

## The core idea

Plain RAG embeds your documents and, at query time, returns the chunks whose
vectors are closest to the question. That works well until the answer depends on
a *relationship* rather than a *similarity* — "how are A and B connected?",
"what did the person who founded X do before?". Those questions need a graph.

This project keeps both representations of your data:

- a **vector index** for "find text that means roughly this", and
- a **knowledge graph** of entities and typed relationships for "follow the
  connections".

An **agent** sits on top and decides, per question, which to use — often both.

## The layers

Each arrow is an interface. Swapping an implementation (a different LLM, a
different store) means writing one adapter, not touching the layers around it.

```
  HTTP  ─────────────  FastAPI  (/query /search /ingest /compare + /docs)
                          │
  Agent ─────────────  LangGraph loop + tools  (the model chooses tools)
                          │
  Retrieval ─────────  Hybrid = vector ⊕ graph ⊕ keyword → RRF → rerank
        ┌─────────────────┼──────────────────┐
        │                 │                  │
  Embeddings         LLM providers       Storage
  (local | API)      (local | API)       GraphStore + VectorStore  → Neo4j
        ▲
  Ingestion:  load → chunk → embed → extract entities/relations → store
```

## Ingestion, step by step

1. **Load.** A loader turns a file into plain text. Images and scanned PDFs go
   through **OCR** first (a small vision model — Gemma 4 locally, Gemini in the
   cloud — reads the text out of the picture).
2. **Chunk.** Text is split into retrievable pieces. The chunker uses the
   *embedding model's own tokenizer* to measure size, so a chunk never exceeds
   what the embedder can encode. (See `docs/CONFIGURATION.md` → chunking.)
3. **Embed.** Each chunk becomes a vector. Vectors are cached in Redis so
   re-ingesting the same text is free.
4. **Extract.** The LLM reads each chunk and pulls out entities and the typed
   relationships between them. These become nodes and edges in Neo4j, and each
   chunk is linked to the entities it mentions (`Chunk -[:MENTIONS]-> Entity`).
   Extraction calls run concurrently (`ingestion.max_concurrency`).
5. **Store.** Chunk vectors live on `:Chunk` nodes with a native Neo4j vector
   index (or in the file-backed `local` store); entities and relations form the
   graph. One database holds both, so going from a matched chunk to its entities
   is a single hop.
6. **Enrich.** Once the document is stored, two passes run: **entity resolution**
   merges duplicates the per-chunk extractor couldn't ("Acme" / "Acme Robotics"),
   and **community detection** clusters the graph and LLM-summarizes each cluster.
   Those summaries answer whole-corpus questions that no single chunk can, via the
   agent's `global_search` tool.

Every node carries a `corpus` tag and every constraint, index, read, and write is
keyed on it — that (not a shared bare key) is what keeps one tenant's entities and
answers out of another's.

## Answering a question

1. The agent receives the question plus a style instruction.
2. It picks tools. `hybrid_search` is the default; `graph_neighbors` /
   `expand_subgraph` follow relationships; `compare` gathers evidence about
   several subjects at once. Tool descriptions live in `agent/tools.py`.
3. Every tool records the exact chunks it surfaced, so the API can return
   precise sources alongside the answer.
4. **Hybrid retrieval** runs vector, graph-augmented, and keyword search
   concurrently (a thread apiece), fuses the three rankings with Reciprocal Rank
   Fusion (which needs no comparable scores), then a reranker orders the
   finalists. Graph-augmented retrieval doesn't just re-find the seed entities —
   it follows relationships out to `graph_hops` and scores chunks by graph
   distance.
5. The agent writes the answer in the requested style, citing sources.

Multi-turn memory is handled by a LangGraph checkpointer keyed on a `thread_id`
(Redis-backed when reachable, in-process otherwise). The API and CLI use the
async and sync saver flavors respectively over one keyspace — the async one is
required because `/query` streams over `astream`. The streaming path also emits
`tool` events as the agent picks strategies, so the UI can show activity instead
of sitting silent through retrieval.

## Why these choices

- **Neo4j as the default store** — mature Cypher, a native vector index, and
  full-text search in one engine, so graph + vectors don't need two systems.
  It's behind a `GraphStore`/`VectorStore` interface, so an embedded backend can
  be added without touching retrieval.
- **LangGraph + LangChain integrations** — one `bind_tools` interface across
  Ollama, Claude, OpenAI, and Gemini is what makes "swap local ↔ API" a
  one-line config change. The trade-off is a heavier dependency; the domain
  layer stays framework-free so we're not locked in.
- **A composition root (`container.py`)** — all wiring in one place, lazily
  built. Nothing constructs its own dependencies, which keeps everything
  testable and swappable.

## Multi-user & memory

Each user has an **isolated knowledge base**, but that isolation is deliberately
cheap. The design splits into two objects:

- **`Container`** — the composition root, holding the *heavy, shared* singletons:
  the embedding model, the reranker model, the LLM client, the Neo4j driver, and
  Redis. Built once for the whole process.
- **`Tenant`** — a *lightweight, per-user* view. It binds thin store / retriever /
  agent wrappers to that user's namespace while **reusing the container's shared
  models**.

That split is the memory optimization: N users cost N sets of small wrappers, not
N copies of the models (which are what actually consume RAM/VRAM). The tenant
cache is an LRU bounded by `tenancy.max_active_tenants` — evicting a tenant frees
only wrappers; the models stay resident.

**How isolation works.** By default, every node a user ingests is tagged with a
`corpus`, and every query filters on it — so one Neo4j database
(Community-friendly) cleanly separates users. Set
`tenancy.per_tenant_database: true` to instead give each user a real Neo4j
database (Enterprise). Conversation memory threads are namespaced
`"{corpus}:{thread}"`, so history never leaks across accounts.

## Shelves — one account, several knowledge bases

A maths textbook and a programming manual are two subjects, not one corpus. Left
in the same namespace they actively corrupt each other: entity resolution merges
the "function" of each into a single node, community summaries describe an
incoherent blend of both, and graph traversal walks from integration into
closures. A **shelf** is the boundary that prevents it — and it reuses the
boundary already there rather than adding a second one:

```
corpus = tenant_id                  # the default shelf
corpus = tenant_id + "." + slug     # every other shelf
```

That one line is the whole mechanism. Neo4j's `(corpus, key)` constraints, the
DuckDB filename, and the checkpointer's thread namespace all key on the corpus
string, so each shelf gets its own entities, its own community summaries, its own
vector file and its own conversation memory — with no changes to a single Cypher
query.

Two properties are load-bearing:

- **The separator is a dot**, because `sanitize_user` and `sanitize_slug` both
  strip it. Neither a tenant id nor a slug can contain one, so `{tenant}.{slug}`
  is unambiguous. A dash would not have been: tenant ids legitimately contain
  dashes, so `alice-1-b-2` could be read as two different (tenant, shelf) pairs.
- **The default shelf's slug is empty**, so its corpus is the bare tenant id —
  which is exactly where everything ingested before shelves existed already
  lives. The migration therefore moves no data at all; it gives a name to what is
  already in place.

A question searches **one** shelf. Which one comes from the conversation, not the
request: a thread is pinned to its shelf at creation and the server reads
`Thread.shelf_id` on every turn, because the agent's memory for that thread is
keyed on the shelf's corpus — a thread that moved would lose its history and
start citing passages its own transcript never mentioned.

Quotas stay **per account**. Files and storage are counted over all shelves by
the query that reserves the slot; the chunk ceiling needs more care, since
`IngestPipeline._check_capacity` can only count the corpus it is writing to. The
API closes that by passing `max_chunks` already reduced by what the user's other
shelves hold, which makes the pipeline's per-corpus check equivalent to the
per-account rule. Otherwise four shelves would buy four times the allowance.

## Presets — ten jobs over the same retrieval

`SYSTEM_PROMPT` has exactly one substitution point, `{style}`, and it is filled
server-side from an enum. A **preset** fills that same slot with a working
method instead of a phrasing: what a finance question needs surfaced (figures
with their period and units, and never arithmetic of your own) is not what a
teaching question needs (build up from the definition, one flagged analogy).

The text lives in **`prompts/*.md`**, not in Python. Prompts are content: they
get revised far more often than the code around them, they are the thing a
non-Python reader most needs access to, and a diff of one should not be a diff
of a source file. The directory is resolved exactly as `configs/` is (checkout,
else cwd, else `GRAPHRAG_PROMPT_DIR`, which the image pins), read once and
cached — a prompt changing under a running process would mean two questions in
one conversation answered under different instructions.

Presets are chosen in the composer and stored as each shelf's default, so opening
the maths shelf selects Study without anyone re-picking it. Three properties keep
them safe to put inside a hardened prompt:

- **Server-side only.** A request names a preset *id*; `canonical_preset` clamps
  it to the enum before anything is looked up, so request text never reaches the
  prompt — and the clamp doubles as the agent cache key, so junk cannot mint
  entries.
- **They narrow, never widen.** No preset grants outside knowledge, relaxes the
  citation requirement, or offers an escape from the closed-domain refusal.
  `test_presets.py` asserts this against a list of permissive phrasings, because
  it is the tempting edit — telling the finance preset to "calculate the ratio"
  would quietly undo the contract above it.
- **`general` still carries `AnswerStyle`.** Its body ends on a
  `## Length and register` heading with nothing under it, and the style
  instruction is what goes there — so a caller that sends no preset (the CLI, an
  API key holder, an older UI build) still gets style-controlled phrasing
  exactly as before.

A job preset *replaces* the style instruction rather than stacking on it. Both
control the same axis, and "thorough, explain the reasoning" alongside "tight
bullets, front-loaded" is a contradiction the model resolves at random.

**The cost is real.** A preset body is ~550-650 tokens, and the agent's tool loop
resends the system prompt every turn — so a preset adds roughly 2-4k prompt
tokens to a question, against the ~10k a question already costs. The token
quotas in `db/models.GlobalLimit` were sized before this; a deployment that runs
close to them should expect to raise them.

**Request routing.** The API reads `X-User-Id`; `Container.tenant(user)` resolves
(and lazily prepares) that user's namespace, reusing the shared models. The CLI
takes `--user`; the UI has a user picker.

## Deployment — one command, end to end

`docker compose up` starts the services that form the whole workflow:

```
                    proxy (Caddy, TLS on 80/443)
                      │  /api/*   → api:8000
                      │  /admin/* → /srv/admin   (admin console SPA)
neo4j ─┐              │  /*       → /srv/app     (chat SPA)
redis ─┼─▶ api ◀──────┘
       │   (FastAPI)
       ├─▶ worker  (Arq — runs ingestion off the API, resource-capped)
       └── ollama  (optional, `local` profile — runs open models)
```

The **proxy** is the public entrypoint and the only container the network can
reach. It terminates TLS (automatic Let's Encrypt for a real domain,
self-signed for `localhost`), forwards `/api/*` to the API with response
buffering off so SSE streams, and serves both web apps' static files itself —
they are built into its image by `docker/Dockerfile.proxy`. Because everything
shares one origin, CORS isn't needed in that path and one session cookie
covers both apps without being widened to a domain scope.

Ingestion is **queued**: an upload enqueues a job on Redis, the `worker` picks it
up and runs the (blocking) embed + graph-extraction pipeline, and job status is
written back to Redis for the API to poll. That keeps the API responsive and puts
the heavy CPU/RAM work in a container you cap independently (`WORKER_MEM_LIMIT`,
`GRAPHRAG_WORKER_CONCURRENCY`). Every container has RAM/CPU limits set from `.env`.

The ordering is health-gated, so "the UI is up" means "the stack is up":

- `neo4j` and `redis` expose healthchecks; `api` waits for both to be healthy
  before starting.
- `api` exposes its own healthcheck (`/health`); `proxy` waits for `api` to be
  healthy. Its `start_period` is generous because the first request may download
  local models.

**Two web apps, one origin.** `frontend/` is the chat app served at `/`;
`admin/` is the operator console served at `/admin`. They are independent npm
projects with no shared code — the console has its own small API client
covering only the endpoints it calls, and its own components. They share the
CSS-variable token palette (duplicated, deliberately) so they read as one
product while remaining separately buildable and separately deployable.

Both are built into the proxy image and served as static files; navigating
between them is a full page load, which is what keeps the admin bundle out of
the end-user one. Everything a user needs is in the browser:

- a **status bar** that polls `/ready` and shows API / Neo4j / Redis health,
- a **drag-and-drop upload** that runs the ingest pipeline and streams per-file
  progress (chunks and entities extracted),
- a **streaming chat** that shows the exact sources behind each answer.

So the full loop — add documents → ask → get a grounded, cited answer, while
watching the system's health — is driven entirely from the served UI. Nothing
here is Docker-specific: the same API runs bare under `uvicorn`, and the UI runs
under `vite dev` (which proxies to `localhost:8000`).
