"""The relational schema: accounts, limits, usage, chat history, files.

Postgres is the system of record for everything that must survive a restart and
be queried by the admin panel. Redis keeps only derived, expendable state —
rate-limit windows, caches, live job status.

Two identity columns matter. `users.id` is the primary key everything references.
`users.tenant_id` is the storage namespace: it names the Neo4j corpus and the
DuckDB file for that user, so it must be a filesystem- and Cypher-safe token
(see `container.sanitize_user`) and must never be reused after deletion.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import INET, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# The two kinds of emailed code. Constants rather than literals because every
# read of `email_otps` has to filter on one of them (see EmailOTP).
PURPOSE_VERIFY = "verify"
PURPOSE_RESET = "reset"


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def _now() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


# --- identity ----------------------------------------------------------------

class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = _uuid_pk()
    # citext would be tidier, but it needs an extension; lowercasing on write
    # keeps the unique index meaningful without one.
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    tenant_id: Mapped[str] = mapped_column(String(48), unique=True, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False, default="user")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    email_verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    password_changed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Lockout state. It lives here rather than in Redis for one reason: the
    # Redis counters in `limits/service.py` fail *open* by design, which is
    # right for a quota and exactly wrong for a lockout — an attacker who can
    # break the counter would be removing the control. Postgres is already read
    # and written on every login (`_by_email`, `last_login_at`), so this costs
    # no extra round trip and survives a restart.
    failed_logins: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _now()

    __table_args__ = (
        CheckConstraint("role in ('user','admin')", name="ck_users_role"),
        CheckConstraint(
            "status in ('pending','active','suspended','deleted')", name="ck_users_status"
        ),
    )

    limits: Mapped[UserLimit | None] = relationship(
        back_populates="user", cascade="all, delete-orphan", uselist=False
    )


class EmailOTP(Base):
    """Short-lived verification codes. Stored hashed: a leaked database dump
    must not let someone verify another person's address.

    Two purposes share this table, and every query MUST scope by `purpose`:

      verify  proves a new address is real, then activates the account
      reset   proves control of a known address, then sets a new password
              (also used for an admin invite — "set your password" is the
              same operation as "reset it")

    They share a lifecycle exactly — hashed code, TTL, attempt cap, one
    outstanding code per user — but they must never be interchangeable. An
    unscoped lookup would let a reset code activate a pending account, and
    issuing one would silently invalidate the other.
    """

    __tablename__ = "email_otps"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    code_hash: Mapped[str] = mapped_column(Text, nullable=False)
    purpose: Mapped[str] = mapped_column(
        String(16), nullable=False, default=PURPOSE_VERIFY, server_default=PURPOSE_VERIFY
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _now()

    __table_args__ = (Index("ix_email_otps_user_purpose", "user_id", "purpose"),)


class Session(Base):
    """Server-side sessions. The cookie carries an opaque token; only its hash
    is stored, so the table is useless to an attacker who reads it — and a
    suspended user can be cut off immediately, which a stateless JWT can't do
    without a denylist that costs the same lookup anyway."""

    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    token_hash: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ip: Mapped[str | None] = mapped_column(INET)
    user_agent: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _now()


class APIKey(Base):
    """Programmatic access. Same `grk_` + SHA-256 scheme the Redis KeyStore
    used, so existing clients keep working; only the storage moved."""

    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    key_hash: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    label: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _now()


# --- limits ------------------------------------------------------------------

_LIMIT_COLUMNS = (
    "messages_per_minute", "messages_per_day", "tokens_per_day", "tokens_per_month",
    "max_files", "max_file_mb", "max_storage_mb", "max_chunks", "max_threads",
)


class GlobalLimit(Base):
    """Defaults for every user. Single row (id=1) so the admin panel edits one
    object rather than a scattered config.

    The token ceilings count PROMPT AND COMPLETION together, which is what
    actually costs money. In RAG the prompt dominates: one measured question on
    a small corpus spent 7,652 prompt tokens against 594 completion, because
    every turn of the agent's tool loop resends the system prompt, the
    conversation and the retrieved chunks. Budget roughly 10k tokens per
    question, not the few hundred the visible answer suggests.

    **Three places state these numbers and they must agree**: the defaults
    below, the `server_default`s in the migrations, and `limits.service._DEFAULTS`
    (used when there is no row at all). They did not — the ORM defaults here were
    5x the other two, so a `global_limits` row created through the ORM rather
    than by Alembic silently shipped a five-fold quota. `test_limit_defaults.py`
    now holds all three together.
    """

    __tablename__ = "global_limits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    messages_per_minute: Mapped[int] = mapped_column(Integer, nullable=False, default=6)
    messages_per_day: Mapped[int] = mapped_column(Integer, nullable=False, default=100)
    # Raised from 150k/2M by migration 0003 to cover the extra model calls answer
    # review makes; changing either needs a matching migration.
    tokens_per_day: Mapped[int] = mapped_column(BigInteger, nullable=False, default=300_000)
    tokens_per_month: Mapped[int] = mapped_column(BigInteger, nullable=False, default=4_000_000)
    max_files: Mapped[int] = mapped_column(Integer, nullable=False, default=10)
    max_file_mb: Mapped[int] = mapped_column(Integer, nullable=False, default=15)
    max_storage_mb: Mapped[int] = mapped_column(Integer, nullable=False, default=100)
    max_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=20_000)
    max_threads: Mapped[int] = mapped_column(Integer, nullable=False, default=25)
    updated_at: Mapped[datetime] = _now()

    __table_args__ = (CheckConstraint("id = 1", name="ck_global_limits_singleton"),)


class UserLimit(Base):
    """Per-user overrides. Every column is nullable: NULL means "inherit the
    global default", so raising a default lifts everyone who wasn't
    deliberately pinned."""

    __tablename__ = "user_limits"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    messages_per_minute: Mapped[int | None] = mapped_column(Integer)
    messages_per_day: Mapped[int | None] = mapped_column(Integer)
    tokens_per_day: Mapped[int | None] = mapped_column(BigInteger)
    tokens_per_month: Mapped[int | None] = mapped_column(BigInteger)
    max_files: Mapped[int | None] = mapped_column(Integer)
    max_file_mb: Mapped[int | None] = mapped_column(Integer)
    max_storage_mb: Mapped[int | None] = mapped_column(Integer)
    max_chunks: Mapped[int | None] = mapped_column(Integer)
    max_threads: Mapped[int | None] = mapped_column(Integer)
    updated_at: Mapped[datetime] = _now()

    user: Mapped[User] = relationship(back_populates="limits")


class UsageEvent(Base):
    """Append-only usage log — the source for admin charts and for any
    accounting that must survive a Redis flush."""

    __tablename__ = "usage_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    amount: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1)
    meta: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = _now()

    __table_args__ = (
        Index("ix_usage_user_time", "user_id", "created_at"),
        Index("ix_usage_time", "created_at"),
    )


# --- shelves -----------------------------------------------------------------

class Shelf(Base):
    """A subject: one shelf of documents with its own knowledge graph.

    A user who uploads a maths textbook and a programming manual wants two
    knowledge bases, not one — otherwise entity resolution merges the "function"
    of each into a single node and graph traversal walks from calculus into
    closures. A shelf is that boundary, and it reuses the one the storage layer
    already has: `slug` becomes the suffix of the Neo4j `corpus` tag and of the
    DuckDB filename (see `container.corpus_for`).

    Two columns carry the weight:

    `slug` is storage-visible and therefore **immutable** once written. Renaming
    a shelf changes `name` only; changing the slug would leave every chunk,
    entity and community summary stranded under a corpus nothing queries. It is
    unique per user, never globally — two people may both have a "physics".

    `is_default` marks the one shelf whose slug is empty, whose corpus is
    therefore the bare tenant id. That is where everything ingested before
    shelves existed already lives, so the default shelf is not a new container
    to migrate data into — it is the name for the data that is already there.
    Exactly one per user, and it cannot be deleted.
    """

    __tablename__ = "shelves"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(80), nullable=False)
    slug: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    # Which job preset this shelf opens with (see agent/presets.py). Stored as a
    # plain string, not an enum column: presets are an application concern and a
    # release that adds one should not need a migration. Unknown values clamp to
    # `general` on read.
    preset: Mapped[str] = mapped_column(String(24), nullable=False, default="general")
    is_default: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    created_at: Mapped[datetime] = _now()

    __table_args__ = (
        UniqueConstraint("user_id", "slug", name="uq_shelves_user_slug"),
        # One default shelf per user. Declared here as well as in migration 0004
        # because `alembic check` compares the models against the database: an
        # index only the migration knows about reads as drift, and CI fails on
        # it. Partial rather than UNIQUE(user_id, is_default), which would
        # instead forbid a user from having two *non*-default shelves.
        Index(
            "uq_shelves_one_default", "user_id",
            unique=True, postgresql_where=text("is_default"),
        ),
    )


# --- chat --------------------------------------------------------------------

class Thread(Base):
    """A conversation. The agent's own memory lives in the checkpointer keyed by
    `{corpus}:{thread.id}`; this table is the transcript the UI renders and
    the list it browses."""

    __tablename__ = "threads"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    # Which shelf this conversation is asking about. Pinned at creation and not
    # changed afterwards: the agent's memory for the thread is keyed on the
    # shelf's corpus, so moving a thread would strand its history — and the
    # transcript would cite passages the new shelf does not contain. NULL means
    # the default shelf, which is what every thread predating shelves is.
    #
    # `index=True` yields ix_threads_shelf_id, the index migration 0004 creates.
    # It has to be declared here too or `alembic check` reports drift.
    shelf_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("shelves.id", ondelete="SET NULL"), index=True
    )
    title: Mapped[str] = mapped_column(String(120), nullable=False, default="New chat")
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _now()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (Index("ix_threads_user", "user_id", "updated_at"),)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    thread_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("threads.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(12), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[dict | None] = mapped_column(JSONB)
    tool_calls: Mapped[dict | None] = mapped_column(JSONB)
    model: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = _now()

    __table_args__ = (
        CheckConstraint("role in ('user','assistant')", name="ck_messages_role"),
        Index("ix_messages_thread", "thread_id", "id"),
    )


# --- documents ---------------------------------------------------------------

class File(Base):
    """Uploaded documents. Replaces the Redis hash that tracked file slots: a
    quota you can only enforce while the cache is up isn't a quota."""

    __tablename__ = "files"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # The shelf this document was ingested into. NULL is the default shelf, both
    # for files that predate shelves and for a deleted shelf's leftovers —
    # SET NULL rather than CASCADE because deleting a shelf must not silently
    # delete the uploads, which still count against the user's file quota and
    # still exist on disk.
    #
    # `index=True` yields ix_files_shelf_id, the index migration 0004 creates.
    # It has to be declared here too or `alembic check` reports drift.
    shelf_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("shelves.id", ondelete="SET NULL"), index=True
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    mime: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="uploaded")
    chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    job_id: Mapped[str | None] = mapped_column(String(16))
    created_at: Mapped[datetime] = _now()


class IngestJob(Base):
    """Terminal state of an ingest, for the admin view. Live progress stays in
    Redis, where the polling endpoint can read it cheaply."""

    __tablename__ = "ingest_jobs"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )
    file_id: Mapped[str | None] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    stats: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


# --- operations --------------------------------------------------------------

class AppSetting(Base):
    """Runtime settings an admin can change without a redeploy (e.g. which chat
    models are offered)."""

    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[dict] = mapped_column(JSONB, nullable=False)
    updated_at: Mapped[datetime] = _now()


class AuditLog(Base):
    """Every admin mutation. Actor is nullable so entries survive the deletion
    of the admin who made them."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    actor_user_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    action: Mapped[str] = mapped_column(String(48), nullable=False)
    target_user_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    detail: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = _now()

    __table_args__ = (Index("ix_audit_time", "created_at"),)


LIMIT_COLUMNS = _LIMIT_COLUMNS
IS_ACTIVE = "active"

__all__ = [
    "APIKey", "AppSetting", "AuditLog", "Base", "EmailOTP", "File", "GlobalLimit",
    "IngestJob", "LIMIT_COLUMNS", "Message", "PURPOSE_RESET", "PURPOSE_VERIFY",
    "Session", "Shelf", "Thread", "UsageEvent", "User", "UserLimit",
]
