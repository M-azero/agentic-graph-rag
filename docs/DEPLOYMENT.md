# Deployment

Target: a single VPS — **4 vCPU, 12 GB RAM, ~100 GB SSD**, any provider.
Everything below assumes that shape; on a different box the only thing that
changes is the memory caps.

Nothing here is host-specific: it is Docker Compose plus a reverse proxy, so it
runs the same on a VPS, a home server, or a laptop. The only external
requirements are a domain whose A record points at the box, ports 80 and 443
open, and outbound HTTPS to whichever model providers you configure.

Securing the box the stack runs on is a separate job from securing the stack,
and this document only covers the second. Before a deployment takes traffic,
harden the host itself: SSH to key-only with no root login, a firewall that
allows 80/443 and SSH and nothing else, unattended security updates, and swap
so a build cannot OOM-kill the database.

**One trap is worth stating here, because it bites almost everyone:** a plain
`ufw deny 5432` does **not** close a container port. Docker writes its own
iptables rules in the `DOCKER` chain, which is evaluated before ufw's, so a
service published to `0.0.0.0` stays reachable no matter what ufw says. The
control that actually works is the one this repo already uses — the `127.0.0.1:`
prefix on every `ports:` entry except the proxy's. `make ports` fails if one
ever goes missing. If you want a firewall backstop as well, it has to be a rule
in the `DOCKER-USER` chain, which *is* evaluated first.

## What runs

| Service | Memory cap | CPU cap | Why it's there |
| --- | --- | --- | --- |
| `api` | 3 GB | 3 | FastAPI + the agent. Also runs ingest (see below). |
| `neo4j` | 2.5 GB | 2 | The knowledge graph. Heap 1g + pagecache 768m fits inside with JVM headroom. |
| `postgres` | 1 GB | 1 | Accounts, limits, usage, chat history, agent memory. |
| `redis` | 384 MB | 0.5 | Caches, rate-limit windows, live job status. All rebuildable. |
| `guardrails` | 512 MB | 1 | Screens every `/query` in and out — a full core because it is in the path of every answer. |
| `proxy` | 256 MB | 1 | Caddy: TLS, both web apps' static files, and the API reverse proxy. The public entrypoint. |

That's **7.6 GB** of limits across 6 containers. Measured idle draw is about
0.9 GB; the caps are headroom for ingest, not steady state.

The remaining ~4 GB is deliberately unallocated, and it is not slack. Each
tenant's vectors live in a `data/vectors/*.duckdb` file read through the **host
page cache**, so memory the containers do not claim is exactly what keeps a
vector scan off the disk. A `docker compose build` — two `npm ci` plus two
`vite build` — wants around a GB of its own on top. Handing the last 4 GB out
as container caps would make queries slower, not faster.

The CPU caps sum to 8.5 on 4 cores on purpose: they are ceilings that stop one
service starving the others under load, not reservations. **The API still runs
a single uvicorn worker** and more cores do not change that — the DuckDB vector
provider takes an exclusive lock per tenant file, so a second worker process
would fight the first for it. The extra cores go to ingest fan-out
(`ingestion.max_concurrency`, now 3) and to keeping Neo4j, Postgres and the
guard off each other's toes.

Three services are off by default: `ollama` (profile `local`), `worker`
(profile `worker`), and the whole llmlens observability stack (profile
`observability`). **llmlens is not part of this deployment** — it adds
ClickHouse plus its own Redis, api, worker and dashboard, roughly another
4.6 GB. Even at 12 GB that does not fit: 7.6 + 4.6 is 12.2 GB of caps with
nothing left for the OS. Adding it means a 16 GB box, or trimming the caps
above first. `.env.example` therefore sets `COMPOSE_FILE` to the base plus the
guardrails overlay and leaves `COMPOSE_PROFILES` unset, so a plain
`docker compose up -d` brings up exactly the six above. See
[INTEGRATIONS.md](INTEGRATIONS.md) for what it costs to add it later.

**No torch.** The production profile uses Cohere for embeddings and reranking,
so the ~3.9 GB of CUDA libraries `sentence-transformers` would pull in are never
installed. Adding `--build-arg EXTRAS='[extras,local-models]'` puts them back if
you want in-process models — on a CPU-only box you almost certainly don't.

## Every model API has a fallback chain

One dead API key should degrade this deployment, not stop it. The production
profile chains providers per surface, and a call that fails moves to the next:

| Surface | Vendors | Chain |
| --- | --- | --- |
| chat, extraction | 3 | `deepseek` → `deepinfra` ×2 → `cohere` |
| OCR | 3 + local | `deepinfra` ×2 → `cohere` → `gemini` → **`tesseract`** (no key) |
| rerank | 3 | `cohere` → `deepseek` → `deepinfra` |
| guard judge | 3 | set in `GUARD_LLM_FALLBACKS` — see below |
| embeddings | 1 | `cohere` **only — deliberately no fallback** |

Two consecutive failures take a provider out of rotation for five minutes; a
denied key answers in ~200 ms and paying that on every request adds up. It
rejoins automatically once a probe succeeds, so a provider that comes back — a
billing block lifted, say — starts serving again with no redeploy. Failovers
are logged as `llm_failover` and counted in `graphrag_llm_failover_total`.

**Count vendors, not links.** A five-deep chain behind one API key survives
nothing, because the outages that actually happen are account-level — a billing
block, a revoked key, a vendor down — and they take out every model behind that
key at once. Two links at the same vendor are still worth having (they cover a
single model being rate-limited or withdrawn), but they are not redundancy. So
each surface above reaches at least two vendors, and the guard's own chain is
configured separately because it runs in a different process:

```bash
docker compose exec -T api python scripts/preflight.py --vendors
```

```
PASS  vendors  chat        3 vendors: deepseek, deepinfra, cohere
PASS  vendors  guard       3 vendors: custom, cohere, deepseek
SKIP  vendors  embeddings  cohere - single vendor by design
```

**The guard is the one where losing the provider is silent.** It fails open: if
the judge stops answering, every verdict comes back `judge_unavailable`,
requests pass unscreened, and the service reports healthy throughout. Nothing
errors. Set `GUARD_LLM_FALLBACKS` (`provider:model,provider:model`) so that
cannot be one dead key, and pair it with `GUARD_LLM_TOTAL_TIMEOUT_S` — without
an overall ceiling the budget is per-link, so three links can outlast
`safety.timeout_s`, at which point the caller has already given up and failed
open while the chain keeps spending tokens. Keep
`safety.timeout_s > GUARD_LLM_TOTAL_TIMEOUT_S`.

Judge links are worth *testing*, not just configuring. Models that classify
prompt injection perfectly will happily score a flatly contradicted answer as
grounded — the verdict is well-formed, so nothing looks wrong. Check candidates
against your own policy on both halves of the job before trusting them.

**Embeddings are the exception, and it matters.** Two embedding models place the
same text at different points in vector space. Falling back to a different one
would write vectors that cannot be compared with the ones already stored:
ingest would poison the index and queries would return nonsense neighbours,
with nothing raised anywhere. So if Cohere is down, ingest fails and should be
retried. Config rejects an `embeddings.fallbacks` entry naming a different
model rather than letting that happen quietly.

One consequence worth knowing: `retrieval.min_relevance` is calibrated on the
reranker's 0–1 scale. If every reranker in the chain is down, scores are raw
retrieval similarities on a different scale, so the closed-domain gate suspends
itself (logged as `relevance_gate_bypassed`) instead of refusing every question.

## First run

```bash
cp .env.example .env
# Set at minimum:
#   GRAPHRAG_NEO4J_PASSWORD, GRAPHRAG_POSTGRES_PASSWORD  (before the first up!)
#   COHERE_API_KEY       (embeddings + rerank — no fallback, so required)
#   DEEPSEEK_API_KEY     (chat default; add DEEPINFRA_TOKEN for the chain)
#   GUARD_LLM_API_KEY    (the guardrails judge — it has no env_file of its own)
#   GUARD_LLM_FALLBACKS  (judge failover; without it the guard silently stops
#                         screening when its one provider goes down)
#   GRAPHRAG_ADMIN_EMAIL (the address you'll sign up with)
#   RESEND_API_KEY + GRAPHRAG_EMAIL_FROM on a domain you verified in Resend
#   SITE_ADDRESS=your.domain, TLS_EMAIL=you@example.com

make up                             # builds, starts, and applies migrations
docker compose exec -T api python scripts/preflight.py
```

**Run `preflight.py` before opening the deployment to traffic.** It makes one
real call per provider, because a revoked or billing-blocked key still
constructs a perfectly good client — nothing looks wrong until a user's
question returns a 500. It exits non-zero only when a surface has *no* working
provider, so a chain running on its second link doesn't block a deploy:

```
PASS  chat        deepseek:deepseek-v4-flash
WARN  chat        gemini:gemini-3.5-flash      403 PERMISSION_DENIED ...
PASS  chat        (chain)                      1/3 links healthy
PASS  guardrails  cohere:command-a-03-2025     judge fallback configured
WARN  guardrails  (judge chain)                single judge, no failover
FAIL  email       ... <onboarding@resend.dev>  shared sender only delivers to your own address
```

It also reports which judge link *answered*, not just which is configured: a
verdict served by link 3 means the two in front of it are down.

> **Set the database passwords *before* this first `make up`.** Postgres and
> Neo4j bake their password in when their data volume is first created;
> changing `.env` afterwards doesn't change the stored password and every
> connection then fails auth. If you hit that, see
> [Operations & troubleshooting](#operations--troubleshooting) below.

Then sign up in the browser with the `GRAPHRAG_ADMIN_EMAIL` address, enter the
code, and claim admin (no restart needed):

```bash
make admin EMAIL=you@example.com
```

Without an email provider configured, codes are written to the log instead of
sent — enough for a first boot or a single admin:

```bash
docker compose logs api | grep "code is"
```

For real users to receive codes, wire up [email delivery](#email-delivery-verification-codes).

## The two settings that decide whether you're exposed

**`GRAPHRAG_PROFILE`.** `local` and `api` disable authentication — any caller
can act as any user via the `X-User-Id` header. Only `production` turns accounts
on. The API logs a warning at startup when auth is off; if you see
`auth_disabled` in a deployed server's log, that server is open.

**`SITE_ADDRESS`.** Set it to your domain and Caddy provisions a Let's Encrypt
certificate and redirects HTTP to HTTPS. Left at `:80`, everything is plaintext
on the wire, including session cookies and passwords.

Every published port except the proxy's 80/443 is bound to `127.0.0.1`, so
from the network only the proxy exists. Neo4j browser, Postgres, Redis and the
raw API are reachable from the box itself (or over an SSH tunnel, e.g.
`ssh -L 7474:localhost:7474 you@host`), never from outside. That binding is the
control, and `make ports` is the check that it stayed that way — a dropped
prefix is a database on the public internet. A `DOCKER-USER` firewall rule is
the backstop for the day one goes missing anyway (see the note at the top of
this document on why `ufw deny` alone is not).

## What the deployment does not publish

The admin API has always been locked — `require_admin_user` fails *closed*, so
with auth on and neither an admin account nor `GRAPHRAG_ADMIN_KEY` it returns
403 rather than opening up. What used to be public was everything *describing*
it:

| Path | Before | Now |
| --- | --- | --- |
| `/api/docs`, `/api/redoc` | 200 — full admin API documented | 404 |
| `/api/openapi.json` | 200 — the machine-readable contract | 404 |
| `/api/metrics` | 200 — every route, its traffic and error rates | 403 without `X-Admin-Key` |
| `/api/admin/*` | 403 | 403 |

Two independent controls, because "the admin API documents itself publicly" is
not a mistake worth risking on one: `api.docs_enabled: false` and
`api.metrics_public: false` in `configs/production.yaml`, plus a `respond 404`
for those paths in the Caddyfile. A profile edit or a changed default cannot
quietly republish them.

Both stay reachable from the box itself on `127.0.0.1:8000`, where the proxy is
not in the path — so `curl localhost:8000/docs` over an SSH tunnel still works
when you need the schema, and a Prometheus scraper can either send
`X-Admin-Key` or scrape the container directly in-network.

**The `/admin` URL itself is not a secret, and shouldn't be treated as one.**
It serves a second static bundle — the admin console is its own app, built from
`admin/` and hosted by the same Caddy — and static files are downloadable by
anyone who asks for them. What protects the admin area is the 403 that
`require_admin_user` returns on every `/api/admin/*` call, which fails closed.
The console's own gate only decides what to *render*: a visitor without the
admin role gets a UI that can fetch nothing. Renaming the path would buy
nothing against anyone who opens developer tools.

Splitting the console out does buy one real thing, though: the end-user bundle
no longer contains the admin route names, its API client, or its charting
dependencies. Someone reading the shipped JavaScript learns nothing about the
admin surface from it.

Caddy also sets HSTS, `X-Content-Type-Options`, `X-Frame-Options: DENY`, a
referrer policy, and strips the `Server` header.

## What one account can reach of another's

With open registration, every anonymous visitor is one signup away from being
an authenticated caller, so tenant isolation is the boundary that matters. Each
account gets its own Neo4j scope, its own DuckDB vector file and its own rows;
the ids that appear in URLs are all checked against the caller:

| Surface | Control |
| --- | --- |
| documents, files | looked up `WHERE user_id = caller`, so a foreign id 404s |
| threads, messages | same, and a thread you don't own is **404, not 403** |
| retrieval, agent memory | scoped per tenant, keyed `tenant:thread` |
| ingest job status | authenticated, and matched against the job's owner |
| server-side paths | **admin only, single file** — see below |
| `/admin/*` | fail-closed 403; a user API key is not an admin key |

404 rather than 403 throughout, deliberately: a 403 for a real id and a 404 for
an invented one lets a caller enumerate which ids exist.

**Server-side path ingest is an operator tool, not a user one.** `POST /ingest`
accepts a path under `data/` as well as a URL, and `data/uploads` is where every
tenant's documents land. Two controls, because either alone leaves a hole:

- the caller must be an **admin** — checked before the path is resolved, so the
  400/404 difference cannot be used to map the disk;
- the path must be a **single file**. Containment in `data/` was the whole check
  once, and it accepted a directory — which `iter_documents` walks recursively,
  so `path=data/uploads` copied every other account's documents into the
  caller's own corpus, queryable afterwards through `/query` and `/search`.

A URL, by contrast, is the caller's own content and needs no special role. It is
metered exactly like an upload: the same per-file cap (`min(api.max_upload_mb,
max_file_mb)`), the same file and storage slots, the same chunk ceiling, and a
`files` row so it can be deleted again. Previously it had none of those, which
made a URL the cheap way past every document quota at once.

**URL ingest is restricted to public addresses.** `POST /ingest` takes a URL and
the server fetches it, which without a destination check is a server-side
request forgery primitive — and an unusually complete one, since the response is
*ingested*, so anything the server can reach becomes a document the caller can
query. Every hop is resolved and checked by address, not name, and redirects are
re-checked rather than followed, so this is refused:

```
POST /ingest?path=http://169.254.169.254/metadata/v1.json
400  169.254.169.254 resolves to a private or link-local address.
```

That address is the cloud metadata service — on DigitalOcean `/metadata/v1.json`
includes the droplet's `user_data`, which is where provisioning secrets live.
Service names on the compose network (`neo4j:7474`, `guardrails:8080`) and
loopback are refused the same way. One residual risk is named rather than
papered over: a host whose DNS answer changes between the check and the
connection (DNS rebinding) can still slip through, because the standard library
resolves again when it connects.

**Errors are scrubbed before they leave the process.** Provider SDKs put the
request URL in their messages and some providers carry the API key in that URL,
so anything client-facing — the SSE `error` event, a failed ingest job's
`detail` — goes through `graphrag.core.redact` first. The unredacted text stays
in the server log. Questions are logged by SHA-256 fingerprint and length, not
text: logs get shipped, backed up and pasted into tickets, and a question put to
a private knowledge base is the user's content.

## Accounts and sign-in

**Password reset.** `/forgot-password` emails a 6-digit code; `/reset-password`
takes the code and a new password. It reuses the same `email_otps` table as
verification, distinguished by a `purpose` column — every read and every
invalidation is scoped by it, because an unscoped lookup would let a reset code
activate a pending account. A completed reset revokes every session the account
holds, since "someone else is signed in as me" is one of the reasons to reset.
No session is opened by the reset itself: a stolen code is not a login.

**Signed-in devices.** The account page lists live sessions with their address,
browser and last-active time, and can revoke one or all-but-this-one. Changing
your password rotates the current session and revokes the rest.

**Rate limits and lockout.** Every `/auth/*` endpoint reachable without
credentials carries a per-IP limit on top of the global bucket
(`auth.rate_limits` in `configs/default.yaml` — login 10/min, signup 5/min,
verify 10/min, resend 3/min, reset 5/min). These key on `X-Real-IP`, which Caddy
*sets*; `X-Forwarded-For` is appended to and so its first entry is
attacker-controlled.

On top of that, `auth.lockout_threshold` consecutive failed logins (default 10)
lock an account, with the wait doubling from 60s to a 1h ceiling. The counter
lives in Postgres, not Redis: the Redis counters elsewhere in this system fail
*open* by design, which is right for a quota and exactly wrong for a lockout —
it would let an attacker remove the control by breaking the cache.

To clear one:

```bash
make unlock EMAIL=someone@example.com     # or the Unlock action in the console
```

That CLI exists for the case the console cannot fix: the locked-out account is
the only admin, so nobody is left who can sign in to unlock it.
`GRAPHRAG_ADMIN_KEY` bypasses login entirely but only as a header, so it cannot
drive the cookie-based console.

**Admin-created accounts.** With `auth.open_registration: false`, the console's
**Invite** button is the way in: it creates a verified, active account and
emails a code to set a password. No password is ever chosen by one person on
behalf of another, and the invite runs through the same reset flow everyone else
uses. Admins can also force a password reset (revokes sessions first, then
mails a code) and clear a lockout. Every one of these writes an audit row.

Note the email budget: reset codes and invites share the same transactional
sending quota as verification (Resend's free tier is 100/day). The per-IP limits
above and the one-outstanding-code-per-purpose rule are what bound it.

## Email delivery (verification codes)

Signup emails a 6-digit code that the account must enter before it activates.
Where that code goes depends on config:

- **No provider key** → the code is written to the API log
  (`docker compose logs api | grep "code is"`). Fine for a first boot or a lone
  admin; useless for real users.
- **Resend or Brevo key set** → the code is emailed.

Sending is best-effort: a signup never fails because the email API had a bad
minute — the user is told to request a resend instead. An `email_send_failed`
line in the log carries the provider's reason when one is rejected.

### Resend

1. Create a key at your Resend dashboard → **API Keys → Create API Key** (a
   `re_…` string). Put it in `.env` as `RESEND_API_KEY=…`.
2. Choose a sender address — this decides who can receive mail:
   - **Testing** — `GRAPHRAG_EMAIL_FROM=Graph RAG <onboarding@resend.dev>`.
     Works with just the key, no DNS. But Resend's shared sender only delivers
     to the address your Resend account is registered under — enough to receive
     your *own* admin code, not anyone else's.
   - **Production** — verify a domain you own (Resend → **Domains → Add Domain**,
     then add the DNS records it shows), and set
     `GRAPHRAG_EMAIL_FROM=Graph RAG <noreply@yourdomain.com>`. Now codes reach
     any address.
3. Apply it — recreate, don't restart (see [troubleshooting](#operations--troubleshooting)):
   ```bash
   docker compose up -d --force-recreate api
   ```

On startup the API logs `email_provider_unconfigured` (falling back to console)
if the key wasn't picked up — usually because the container was `restart`ed
rather than recreated.

Verify delivery without creating an account (swap in an address your sender can
reach):

```bash
docker compose exec -T api python - <<'PY'
import asyncio
from graphrag.accounts.emails import build_email_sender
from graphrag.config.loader import load_settings
settings, secrets = load_settings()
sender = build_email_sender(settings, secrets)
print("sender:", type(sender).__name__, "from:", secrets.email_from)
print("sent :", asyncio.run(sender.send("you@example.com", "Test", "It works.")))
PY
```

`sender: ConsoleSender` means the key isn't active — check `RESEND_API_KEY` and
that you recreated the container.

## Why ingest runs inside the API

The DuckDB vector store gives each user their own database file. DuckDB takes an
exclusive lock on an open file, so exactly one OS process may hold a given
tenant's database — and the API needs it for every query.

Rather than coordinate two processes over a lock, the deployment removes the
second one: ingest runs as a background task in the API under a semaphore, so
one upload at a time, off the request path. With cloud embedding and extraction
this work is I/O-bound, so it doesn't fight the chat stream for CPU.

The Arq worker still exists for deployments using the Neo4j vector provider:

```bash
GRAPHRAG_USE_WORKER=1 docker compose --profile worker up -d
```

It refuses to start against the `duckdb` provider rather than corrupt a file.

## Backups

Two things hold user data, and both need backing up:

```bash
# Postgres: accounts, limits, usage, chat history
docker compose exec -T postgres pg_dump -U graphrag graphrag | gzip > pg-$(date +%F).sql.gz

# Neo4j: the knowledge graph
docker compose exec neo4j neo4j-admin database dump neo4j --to-path=/data/backups

# Vectors + uploads: plain files on the host
tar czf data-$(date +%F).tar.gz data/
```

`data/vectors/` holds one `.duckdb` file per tenant, so a single user can be
restored without touching anyone else's. Restore has to be rehearsed to count —
untested backups are a belief, not a policy.

## Operations & troubleshooting

### `.env` needs a recreate; `configs/*.yaml` needs a rebuild

Three different commands, and picking the weakest one fails silently — the
container comes up healthy, serving the old settings.

| Changed | Command |
| --- | --- |
| nothing, just stuck | `docker compose restart api` |
| `.env` — a key, a password, the sender | `docker compose up -d --force-recreate api` |
| `configs/*.yaml` — a model, a chain, a flag | `docker compose up -d --build api` |
| `docker/Caddyfile` — a route, a header | `docker compose restart proxy` |
| `frontend/` or `admin/` — any UI change | `docker compose up -d --build proxy` |
| `src/graphrag/db/models.py` or a new migration | `docker compose up -d --build api` **then** `docker compose exec -T api alembic upgrade head` |

`restart` reuses the container's existing environment, so a new `.env` value is
ignored. And `configs/` is **copied into the image** at build time with no
volume mount, so a recreate re-reads `.env` but still runs the *old* YAML —
which looks exactly like your config change having no effect. `preflight.py`
prints the chains it actually loaded, which is the fastest way to tell whether
the container is running what you think it is.

The Caddyfile is both baked into the proxy image *and* bind-mounted, so a config
fix is a restart rather than a rebuild — worth having when the thing you are
fixing is the only way into the deployment. Validate it before you ship it:

```bash
make proxy-check     # caddy validate, without touching the running stack
```

The web apps are the opposite: their built files live inside the image, so a UI
change needs `--build`. That build runs two `npm ci` + `vite build` stages,
which is about a minute on 4 vCPU — `make up` does it for you. Run
`docker builder prune` occasionally; node_modules layers accumulate.

**A schema change needs both steps.** Rebuilding the API without migrating leaves
new columns missing, and the first request that selects one returns a 500 — the
API looks healthy and every signed-in page breaks at once. `make up` runs
`alembic upgrade head` for you, which is the reason to prefer it over a bare
`docker compose up -d --build`.

### "password authentication failed" for user graphrag / neo4j

Postgres and Neo4j read their password **only when their data volume is first
created**. Changing `GRAPHRAG_POSTGRES_PASSWORD` / `GRAPHRAG_NEO4J_PASSWORD`
afterwards doesn't touch the stored password — the app sends the new one and the
database rejects it. Set them before the first `make up`. To fix a volume that's
already out of sync:

**Postgres** — reset the stored password in place (non-destructive):

```bash
docker compose exec -T postgres psql -U graphrag -d graphrag \
  -c "ALTER USER graphrag WITH PASSWORD 'the-value-from-your-env';"
docker compose up -d --force-recreate api
```

(The socket connection inside the container uses trust auth, so this works
without knowing the old password.)

**Neo4j** stores credentials in its system database — there's no in-place reset
without the old password. If the graph holds nothing you need (for instance you
never completed an ingest), recreate its volume so it reinitializes from `.env`:

```bash
docker compose rm -sf neo4j
docker volume rm agentic-graph-rag_neo4j_data   # wipes the graph ONLY
docker compose up -d neo4j
docker compose up -d --force-recreate api
```

Postgres (accounts) and the DuckDB vectors are separate volumes and are
untouched. If you *do* have graph data to keep, instead restart Neo4j once with
`NEO4J_dbms_security_auth__enabled=false`, run `ALTER USER neo4j SET PASSWORD`,
then remove that override and restart.

### Uploads fail with a permission error

The API runs as a non-root user; its entrypoint chowns the mounted `data/`
directory on start so it can write uploads and the per-user DuckDB files. If
uploads 500 with `PermissionError`, the entrypoint didn't run — check the image
was built from the current `docker/Dockerfile` (it must have an `ENTRYPOINT`),
and that `docker/entrypoint.sh` has LF line endings (a `.gitattributes` pins
this; a Windows checkout without it can reintroduce CRLF).

### Becoming admin / reaching the admin panel

The admin area at `/admin` needs an account with the admin role.

1. Sign up in the browser with the address in `GRAPHRAG_ADMIN_EMAIL`.
2. Promote it (no restart needed):
   ```bash
   make admin EMAIL=you@example.com
   # or: docker compose exec api graphrag promote-admin you@example.com
   ```
   Restarting the API also auto-promotes `GRAPHRAG_ADMIN_EMAIL` once that account
   exists.
3. Go to `/admin` directly — e.g. `https://your.domain/admin`.

There is deliberately **no link to the console from the chat app**. An operator
knows the URL; putting an entry point in the end-user chrome would advertise the
surface to everyone who reads the markup for no benefit. A visitor who guesses
the path still gets nothing: `require_admin_user` 403s every `/api/admin/*` call
and fails closed, and the console renders a refusal for a non-admin account.

Break-glass: `GRAPHRAG_ADMIN_KEY` in `.env` reaches admin endpoints with an
`X-Admin-Key` header even when no admin account exists — for bootstrap or
recovery when you're locked out.

## Scaling past one box

The design has room, in roughly this order:

1. **Postgres and Neo4j move to managed services.** Both are already reached by
   URL; nothing in the app assumes they're local.
2. **More API replicas.** Rate limits and caches are already Redis-backed and
   shared. The blocker is DuckDB's single-writer file: replicas need either the
   Neo4j vector provider, or sticky routing per tenant, or a networked vector
   store behind the same `VectorStore` interface.
3. **An ANN index.** The exact cosine scan is milliseconds at the per-user chunk
   ceiling and always exact. Past that, see
   [OPTIMIZATION-NOTES.md](OPTIMIZATION-NOTES.md).
