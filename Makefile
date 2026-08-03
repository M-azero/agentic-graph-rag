# Agentic Graph RAG — common tasks.
# Run `make help` to see everything.

PROFILE ?= production
COMPOSE := docker compose
# The integrations overlay (Guardrails + llmlens) layered on the base stack.
FEATURES := -f docker-compose.yml -f docker-compose.integrations.yml

.DEFAULT_GOAL := help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install the package with dev extras (editable)
	pip install -e ".[dev,extras]"

setup: ## Pick a config profile: make setup PROFILE=production (or local / api)
	@test -f .env || cp .env.example .env
	@grep -v '^GRAPHRAG_PROFILE=' .env > .env.tmp && \
		echo "GRAPHRAG_PROFILE=$(PROFILE)" >> .env.tmp && mv .env.tmp .env
	@echo "Profile set to '$(PROFILE)' in .env. Edit .env for secrets, then: make up"

migrate: ## Apply database migrations (needs GRAPHRAG_DATABASE_URL)
	alembic upgrade head

serve: ## Run the API bare (no Docker) with uvicorn
	GRAPHRAG_PROFILE=$(PROFILE) uvicorn graphrag.api.app:create_app --factory --reload --port 8000

worker: ## Run the optional ingest worker (needs Redis; not for the duckdb provider)
	GRAPHRAG_PROFILE=$(PROFILE) arq graphrag.worker.WorkerSettings

# FILE, not PATH: overriding PATH would clobber the shell's executable search
# path and the recipe couldn't find `graphrag` at all.
ingest: ## Ingest a file or folder: make ingest FILE=./data/mydoc.pdf
	GRAPHRAG_PROFILE=$(PROFILE) graphrag ingest $(FILE)

admin: ## Grant an account the admin role: make admin EMAIL=you@example.com
	$(COMPOSE) exec api graphrag promote-admin $(EMAIL)

# The break-glass for a lockout the admin console cannot fix — because the
# locked-out account is the only admin, so nobody is left who can sign in.
unlock: ## Clear a login lockout: make unlock EMAIL=you@example.com
	$(COMPOSE) exec api graphrag unlock $(EMAIL)

up: ## One command: bring up the whole stack, then apply migrations
	$(COMPOSE) up -d --build
	$(COMPOSE) exec -T api alembic upgrade head
	@echo "App:      http://localhost         (admin console: http://localhost/admin)"
	@echo "API:      http://localhost:8000  (docs: http://localhost:8000/docs)"
	@echo "Neo4j:    http://localhost:7474"

up-features: ## RAG + Guardrails safety, one stack (no llmlens)
	$(COMPOSE) $(FEATURES) up -d --build
	$(COMPOSE) $(FEATURES) exec -T api alembic upgrade head
	@echo "App:        http://localhost   Admin: http://localhost/admin"
	@echo "API:        http://localhost:8000/docs"
	@echo "Guardrails: http://localhost:8080/health"

deploy: ## Everything in ONE command: RAG + Guardrails + llmlens observability
	$(COMPOSE) $(FEATURES) --profile observability up -d --build
	$(COMPOSE) $(FEATURES) exec -T api alembic upgrade head
	@echo "App:        http://localhost   Admin: http://localhost/admin"
	@echo "API:        http://localhost:8000/docs"
	@echo "Guardrails: http://localhost:8080/health  (verdict service)"
	@echo "llmlens UI: http://localhost:5273         (traces / cost / alerts)"
	@echo "NOTE: enable the features in configs/$(PROFILE).yaml — safety.enabled / observability.enabled"

deploy-down: ## Stop the full integrated stack (incl. llmlens)
	$(COMPOSE) $(FEATURES) --profile observability down

down: ## Stop the stack
	$(COMPOSE) down

logs: ## Tail all container logs
	$(COMPOSE) logs -f

test: ## Run unit tests (fast, no services)
	pytest -m "not integration"

# The integration fixtures create and drop the schema, so they refuse any
# database whose name doesn't look disposable.
test-integration: ## Run integration tests (needs Postgres; set GRAPHRAG_TEST_DATABASE_URL)
	pytest -m integration

test-all: ## Run every test (needs Postgres + Neo4j + Redis up)
	pytest

eval: ## Score retrieval + answers against the golden set (needs the stack up)
	GRAPHRAG_PROFILE=$(PROFILE) python scripts/eval.py

web: ## Build both web apps (type-checks them too)
	cd frontend && npm ci && npm run build
	cd admin && npm ci && npm run build

frontend: web   ## Deprecated alias for `web`

# The Caddyfile is the only way into the deployment. Check it before you ship
# it, not after the container refuses to start.
proxy-check: ## Validate docker/Caddyfile without deploying it
	docker run --rm -v "$(CURDIR)/docker/Caddyfile:/etc/caddy/Caddyfile:ro" 	  -e SITE_ADDRESS=:80 -e TLS_EMAIL= -e MAX_UPLOAD_MB=25 	  caddy:2-alpine caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile

lint: ## Lint & type-check
	ruff check src tests migrations
	mypy src

fmt: ## Auto-format & fix
	ruff check --fix src tests migrations
	ruff format src tests migrations

.PHONY: help install setup migrate serve worker ingest admin up down logs \
        up-features deploy deploy-down \
        test test-integration test-all eval frontend lint fmt
