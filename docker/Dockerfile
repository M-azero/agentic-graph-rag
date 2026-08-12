# API image. Includes Tesseract (OCR fallback) and poppler (PDF rendering).
#
# Two stages so the compiler toolchain needed to build wheels doesn't ship to
# production: the builder produces a virtualenv, the runtime copies it. Saves
# roughly 300 MB and removes gcc from the deployed image.
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Dependencies first for better layer caching.
COPY pyproject.toml README.md ./
COPY src ./src

# `[extras]` ships the API embedders/rerankers (Cohere, Voyage) the cloud
# profiles select — without them those profiles cannot embed or rerank at all.
# They're small. What stays opt-in is `[local-models]` (torch, ~3.9 GB of it
# CUDA libraries this container has no GPU for): add it via
# --build-arg EXTRAS='[extras,local-models]' if you want the in-process
# embedder/cross-encoder; the `local` profile serves both roles from Ollama.
ARG EXTRAS="[extras]"
RUN pip install ".${EXTRAS}"

# llmlens observability SDK (integrated feature). Tiny — its only dependency is
# httpx, already installed — so it ships by default and `observability.enabled:
# true` works out of the box; the feature stays OFF unless enabled in config.
# Build with --build-arg WITH_LLMLENS=0 to skip it.
ARG WITH_LLMLENS=1
# Self-contained: the SDK carries its own README, so it installs from a copied
# directory with nothing alongside it. It used to declare
# `readme = "../README.md"` and needed the llmlens README staged one level
# above — hatchling 1.32 rejects that path outright, file present or not.
COPY integrations/llmlens/sdk /tmp/llmlens-sdk
RUN if [ "$WITH_LLMLENS" = "1" ]; then pip install /tmp/llmlens-sdk; fi


FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/app/.cache/huggingface \
    GRAPHRAG_CONFIG_DIR=/app/configs \
    GRAPHRAG_PROMPT_DIR=/app/prompts

# gosu lets the entrypoint drop from root to the app user cleanly (the pattern
# the official Postgres image uses).
RUN apt-get update && apt-get install -y --no-install-recommends \
      tesseract-ocr poppler-utils gosu \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY pyproject.toml README.md alembic.ini ./
COPY src ./src
COPY configs ./configs
# The agent's job presets. Authored as Markdown and read at runtime, so they are
# image content like configs are — not packaged inside the wheel.
COPY prompts ./prompts
COPY migrations ./migrations
# The preflight check ships with the image so it can be run where it means
# something — against the deployed config, from inside the network:
#   docker compose exec -T api python scripts/preflight.py
COPY scripts/preflight.py ./scripts/preflight.py
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

# Create the non-root user and own the image's paths. `data/` is owned at
# runtime by the entrypoint, because it's a host bind mount whose ownership the
# build can't set.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data /app/.cache/huggingface \
    && chown -R app:app /app \
    && chmod +x /usr/local/bin/entrypoint.sh

# Stay root so the entrypoint can chown the mount; it drops to `app` before
# exec'ing the server, so the actual process is unprivileged.
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --retries=30 --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# One worker process, and it stays one however many cores the box has: the
# duckdb vector provider takes an exclusive lock per tenant file, so a second
# uvicorn worker would not share the load — it would fail to open the database.
# Extra cores go to ingest fan-out (ingestion.max_concurrency) instead.
CMD ["uvicorn", "graphrag.api.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
