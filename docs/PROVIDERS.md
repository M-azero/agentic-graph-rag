# Providers

Every model role is swappable. This table shows the local and cloud options and
which config key selects them.

| Role         | Config key            | Local default              | Cloud options                          |
|--------------|-----------------------|----------------------------|----------------------------------------|
| Reply LLM    | `llm.provider`        | `ollama` (gemma4)          | `anthropic` (claude-opus-4-8), `openai`, `gemini`, `deepseek`, `qwen`, `deepinfra` |
| Extraction   | `ingestion.llm`       | `ollama` (gemma4) — defaults to `llm` | any chat provider |
| Embeddings   | `embeddings.provider` | `ollama` (bge-m3) or `sentence_transformers` (bge-m3) | `voyage`, `openai`, `gemini`, `cohere`, `deepinfra` |
| OCR          | `ocr.vision_llm.provider` | `ollama` (gemma3:4b)   | `gemini` (gemini-2.5-flash); or `tesseract` (offline) |
| Reranker     | `retrieval.rerank.provider` | `ollama` (any chat model) or `cross_encoder` (bge-reranker) | `voyage`, `cohere`, `anthropic`, `openai`, `gemini`, `deepseek`, `qwen`, `deepinfra` |

Any role can also be given a **failover chain** — see
[Failover](#failover-one-dead-key-should-not-stop-the-service) below.

## DeepInfra

Serverless open-weight models (Llama, Qwen, DeepSeek, Mistral, BGE embeddings)
behind one OpenAI-compatible endpoint, `https://api.deepinfra.com/v1/openai`.
Set `DEEPINFRA_TOKEN` in `.env` — `DEEPINFRA_API_KEY` is accepted too, since
that is what most people type first.

```yaml
llm:
  provider: deepinfra
  model: deepseek-ai/DeepSeek-V3

embeddings:
  provider: deepinfra
  model: BAAI/bge-m3
  dimensions: 1024

retrieval:
  rerank:
    provider: deepinfra          # generative reranking, see the table below
    model: Qwen/Qwen3-235B-A22B
```

**Model ids are the upstream ones, prefix and capitalisation included** —
`deepseek-ai/DeepSeek-V3`, not `deepseek-v3`. A wrong case is a 404 on a model
that exists. The catalogue changes as weights land and retire, so check
[deepinfra.com/models](https://deepinfra.com/models) rather than trusting a list
in a README.

Two things make it a good fit for self-hosting this project:

**It is the one cloud embedder you can safely fail over.** Embedding fallbacks
are normally refused, because a different model puts text in a different vector
space and silently corrupts retrieval (see below). DeepInfra serves models under
their *upstream* ids, so `BAAI/bge-m3` there is the same weights as `BAAI/bge-m3`
under `sentence_transformers` — same vectors, same space. That makes this legal,
where a Cohere → Voyage fallback is not:

```yaml
embeddings:
  provider: deepinfra
  model: BAAI/bge-m3
  fallbacks:
    - { provider: sentence_transformers, model: BAAI/bge-m3 }
```

**Chunk sizing already agrees with it.** Chunks are measured in the embedding
model's own tokens via an HF tokenizer, and `BAAI/bge-m3` is an HF repo id — so
`embeddings.tokenizer` needs no override, unlike an Ollama tag such as
`bge-m3:latest`.

## How swapping works

All chat models are built through one factory (`llm/factory.py`) and returned as
LangChain chat models, so they share the same `.invoke` / `.astream` /
`.bind_tools` interface. The agent never knows which provider it's talking to.

Embedders implement a tiny `Embedder` interface (`embed_documents`,
`embed_query`). Cloud models are wrapped from their LangChain integrations. Local
models have two routes:

- **`ollama`** — reuses a model you've already pulled, so nothing is downloaded
  twice and the weights stay out of the API/worker images. Ollama owns pooling,
  `max_seq_length`, and `device`; setting those in config logs a warning and has
  no effect. Prefixes, `normalize`, and Matryoshka `dimensions` still apply —
  they're applied to the text and vectors on our side.
- **`sentence_transformers`** — runs the model in-process for full control over
  every knob. Needs the `local-models` extra: it pulls torch, which ships ~2.7 GB
  of CUDA libraries on top of its own 1.2 GB. Install with
  `pip install '.[local-models]'`, or for Docker
  `docker compose build --build-arg EXTRAS='[local-models]'`. Without it the
  provider raises a `ProviderError` naming the extra.

Rerankers come in three shapes, because Ollama serves no cross-encoder endpoint
(`/api/rerank` is not a route — only generate/chat/embed):

| Provider | How it scores | Cost at `candidate_k: 24` |
|----------|---------------|---------------------------|
| `cross_encoder` | local HF model, all pairs in one batched pass | ~0.2s, needs the `local-models` extra + a 2.2 GB model download |
| `ollama` / `anthropic` / `openai` / `gemini` / `deepseek` / `qwen` / `deepinfra` | a chat model rates each pair 0-10 | ~6s, no download |
| `cohere` / `voyage` | hosted rerank endpoint | network round-trip, API key |

Generative reranking is pointwise — one call per candidate — so `candidate_k`
drives its cost directly, and `concurrency` (default 4) hides what it can.
Raising concurrency past Ollama's own parallelism buys nothing.

Quality depends on the model returning a bare number, so `rerank.prompt` is
config-exposed. Candidates the model fails to score keep their retrieval order
rather than being dropped. **Under `ollama` this sets `reasoning: false`**: a
thinking model spends its whole token budget reasoning and returns empty content,
scoring nothing at all — and reasoning is ~8x the latency for no benefit here.
Override via `rerank.extra` (raise `rerank.max_tokens` too if you do).

Not every model tagged "reranker" works this way. A generative reranker must
follow the scoring prompt; some GGUF builds don't, and score every document
identically — test yours before trusting it.

## Running fully local

1. Pull the models into the Ollama on your host:
   ```
   ollama pull gemma4:e4b-it-q4_K_M   # chat + reranking + extraction
   ollama pull gemma3:4b              # OCR
   ollama pull bge-m3                 # embeddings
   ```
2. `make setup PROFILE=local`
3. `make up`

The containers reach your host Ollama at `host.docker.internal:11434`. No API
keys, and no model weights downloaded — the `local` profile runs every role on
Ollama. The only Hugging Face fetch is the bge-m3 *tokenizer* (~17 MB), used to
count chunk sizes exactly; see `embeddings.tokenizer`.

No Ollama on the host? Bring up the bundled service with `--profile local`, set
`GRAPHRAG_OLLAMA_BASE_URL_INTERNAL=http://ollama:11434`, and pull the models
inside that container instead.

### Why the local profile uses two models

Roles have different requirements, and a model that claims to cover several
doesn't always deliver all of them:

- **OCR is `gemma3:4b`, not gemma4.** `gemma4:e4b-it-q4_K_M` lists `vision` in
  `ollama show` and then ignores attached images, replying "please provide the
  image" — indistinguishable, to the pipeline, from a page with no text.
  gemma3:4b transcribes the same handwritten page correctly in ~13s.
- **Extraction is gemma4, not gemma3.** On notes dense with transition tables,
  gemma3:4b ignores "skip document-local labels" and fills the graph with state
  names (`Q0`..`Q16`); gemma4 on the same prompt returns none.

They're 3.3 GB each and won't co-reside on a 6 GB card, so ingest swaps models
once per document. On a larger card both stay resident and it costs nothing.

The lesson generalises: **the capability list is a claim, not a guarantee.** Test
one page, one chunk, one query against the model you plan to use.

### Size the model to the card

A larger model that doesn't fit in VRAM is a *slower* model: Ollama runs the
overflow on CPU, and that costs more than the quality gain is worth. Measured on
a 6 GB card, same prompt, both warm:

| Chat model | VRAM @ ctx 8192 | On GPU | Speed |
|---|---|---|---|
| `gemma4:e4b-it-q4_K_M` | 3.3 GB | **100%** | **41.2 tok/s** |
| `qwen2.5:7b-instruct` | 5.4 GB | 79% | 27.7 tok/s |

qwen2.5 is the stronger model with working tool calling, and it still loses by
~50% here — it can't fit beside the ~1 GB the desktop holds, at any context (82%
on GPU even at 4096). `ollama ps` is the test: below `100% GPU` you're paying for
weights you can't use. On a bigger card this flips, which is why it's a config
key rather than a default.

### One context per model, everywhere it appears

Ollama keys a loaded runner by **(model, options)** — so the same model at two
context sizes is **two runners**, competing for the same VRAM. This is easy to do
by accident: giving reranking a small `num_ctx` looks like a saving, and is the
opposite. On a 6 GB card, chat at 8192 + reranking at 2048 means Ollama evicts
chat to rerank, then evicts reranking to answer — every query.

| Reranking config | rerank | then chat | total |
|---|---|---|---|
| e4b @2048 (separate runner) | 17.5s | 15.8s | **33.4s** |
| e2b @2048 (smaller — still separate) | 14.7s | 16.5s | **31.2s** |
| **e4b @8192 (same runner as chat)** | **3.6s** | **1.9s** | **5.5s** |

A **6x** difference, same rerank scores, same model. Note the middle row: a
*smaller* model barely helped, because the cost was never size — it was the swap.
Give every role that shares a model the identical `num_ctx`, even where a smaller
one would do; if it doesn't change the VRAM (4096 and 8192 are both 3.3 GB for
gemma4), the smaller number is pure cost.

### Using a bigger Gemma 4 for chat

Gemma 4 has native tool calling across the whole family (including the small
`e4b`), so it works in the normal agentic mode — no special handling. The
`local-gemma` profile swaps in a larger chat model and leaves everything else:

```
ollama pull gemma4:12b        # chat — steadier tool calling
make setup PROFILE=local-gemma
make up
```

## Running on cloud APIs

1. `make setup PROFILE=api`
2. Put the relevant keys in `.env` (see `docs/CONFIGURATION.md`).
3. `make up`

## Failover: one dead key should not stop the service

Any role that talks to an API takes a `fallbacks` list. A call that fails moves
to the next entry, transparently to the agent:

```yaml
llm:
  provider: deepseek
  model: deepseek-v4-flash
  fallbacks:
    - { provider: deepinfra, model: deepseek-ai/DeepSeek-V3 }
    - { provider: gemini,    model: gemini-3.5-flash }
  failover_max_failures: 2         # consecutive failures before a provider is benched
  failover_cooldown_seconds: 300   # how long it stays benched
```

A link whose API key is unset is dropped when the chain is built, so an unused
provider costs nothing. Failovers log `llm_failover` and increment
`graphrag_llm_failover_total{from_model,to_model,reason}`.

Three behaviours worth knowing, because each one is a decision rather than an
accident:

- **A stream that has already emitted tokens does not fail over.** Restarting on
  another provider would print the answer to the user twice, so once any token
  is out the error propagates instead.
- **A dead provider gets benched.** A revoked key fails in ~200 ms *per call*;
  without a circuit breaker that is a permanent latency tax. After
  `failover_max_failures` in a row a provider is skipped for the cooldown, then
  probed again — so one that comes back starts serving with no redeploy.
- **Request-shaped errors do not fail over.** A prompt over the context limit,
  or one a safety filter refused, fails identically on the next provider.
  Retrying only spends a second call to reach the same error.

### Embeddings are the exception

`embeddings.fallbacks` may only name **the same model** as the primary, and
config raises at load time otherwise. This is not caution for its own sake: two
embedding models place the same text at different points in vector space, so a
silent switch writes vectors that cannot be compared with the ones already
stored. Ingest would poison the index and queries would return nonsense
neighbours, with nothing raised anywhere — the worst kind of failure, one that
looks like it worked. If your embedder is down, ingest should fail and be
retried.

What *is* allowed is a second route to the same weights — see the DeepInfra
example above.

### The reranker decides whether questions get answered

`retrieval.min_relevance` refuses any question whose best chunk scores below it,
on the reranker's 0–1 scale. So a dead reranker is not merely worse ordering. If
every link in the rerank chain fails, scores fall back to raw retrieval
similarities on a different scale, and the gate **suspends itself** (logged as
`relevance_gate_bypassed`) rather than compare against numbers that do not mean
what it expects. Answering with citations beats refusing everything.

## Checking what actually works

A revoked or billing-blocked key still constructs a perfectly good client, so
nothing looks wrong until a user's question 500s. `scripts/preflight.py` makes
one real call per provider:

```bash
python scripts/preflight.py            # everything
python scripts/preflight.py --quick    # skip the paid calls
docker compose exec -T api python scripts/preflight.py
```

It exits non-zero only when a role has *no* working provider, so a chain running
on its second link reports `WARN` and still passes.

## Adding a new provider

- **LLM:** add a branch in `llm/factory.py`. If it is OpenAI-compatible — as
  DeepSeek, Qwen and DeepInfra all are — that is a `ChatOpenAI` with a
  `base_url`, plus the key in `Secrets` and an entry in `_CREDENTIAL` so
  failover can tell a configured provider from an unconfigured one.
- **Embeddings:** add a branch in `embeddings/api_providers.py` (or a new class
  implementing `Embedder`). Add its output size to `_KNOWN_DIMS` — the vector
  store is created with that number, so a wrong guess costs a re-ingest.
- **Reranker:** add the name to `_LLM_PROVIDERS` in `retrieval/reranker.py` to
  use it generatively.
- **Store:** implement `GraphStore` / `VectorStore` and register it in
  `storage/__init__.py`.

Nothing else changes — the container wires whatever the config selects.
