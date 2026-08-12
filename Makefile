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

# Host security: SSH, firewall, fail2ban, port exposure. AUDIT ONLY — it
# reports and changes nothing. Applying is deliberately not a make target: it
# needs SSH_USER and takes the box's remote access with it, so it wants the
# arguments typed out.
#
# The script is operator material for a specific box and is not committed (see
# .gitignore), so this target explains itself rather than failing with a bare
# "No such file" on a fresh clone. `make ports` is the part everyone needs and
# it works everywhere.
harden: ## Audit the host's SSH / firewall / port exposure (changes nothing)
	@if [ -x ./scripts/harden-host.sh ]; then \
	    sudo ./scripts/harden-host.sh; \
	else \
	    echo "scripts/harden-host.sh is not in this checkout — it is operator"; \
	    echo "material for one deployment, not part of the project."; \
	    echo; \
	    echo "For host hardening guidance see docs/DEPLOYMENT.md."; \
	    echo "For the check that matters most here, run:  make ports"; \
	fi

# The compose files bind everything but the proxy to 127.0.0.1. This is the
# check that it stayed that way: a dropped prefix is a database on the public
# internet, and `ufw deny` does not close a published container port.
ports: ## Fail if a container port is published outside 127.0.0.1 (except 80/443)
	@exposed=`$(COMPOSE) ps --format '{{.Name}}|{{.Ports}}' 2>/dev/null \
	    | awk -F'|' '{n=split($$2,m,", "); for(i=1;i<=n;i++) if(m[i]!="") print $$1 "  " m[i]}' \
	    | grep -E '(0\.0\.0\.0|\[::\]):' | grep -vE ':(80|443)->'`; \
	if [ -n "$$exposed" ]; then \
	  echo "EXPOSED - these answer from the network and should be 127.0.0.1-bound:"; \
	  echo "$$exposed"; \
	  echo "Fix the ports: entry in docker-compose.yml, then: docker compose up -d"; \
	  exit 1; \
	fi; \
	echo "OK - only the proxy's 80/443 are reachable from the network"

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

.PHONY: help install setup migrate serve worker ingest admin unlock up down logs \
        up-features deploy deploy-down harden ports proxy-check web \
        test test-integration test-all eval frontend lint fmt
