"""Shelves: per-subject knowledge bases inside one account.

A shelf's `slug` is the suffix of its Neo4j corpus and its DuckDB filename, so
this migration is the point where a storage-visible name comes into existence.
Two consequences shape it:

- **Every user gets a default shelf, with an empty slug.** An empty slug means
  the corpus is the bare tenant id — which is exactly where everything ingested
  before this migration already lives. So the backfill does not move a single
  chunk, vector or entity: it gives a name to data that is already in place. Any
  scheme that gave the first shelf a non-empty slug would have orphaned every
  existing document behind a corpus no query names.

- **Existing files and threads stay NULL.** NULL reads as "the default shelf"
  everywhere in the application, so an unmigrated row and a deliberately-default
  row behave identically. Backfilling them to point at the new default shelf row
  would be equivalent and slower, and it would make the FK the source of truth
  for something the corpus already decides.

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-07
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "shelves",
        sa.Column(
            "id", UUID(as_uuid=True), primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "user_id", UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False,
        ),
        sa.Column("name", sa.String(80), nullable=False),
        sa.Column("slug", sa.String(32), nullable=False, server_default=""),
        sa.Column("preset", sa.String(24), nullable=False, server_default="general"),
        sa.Column(
            "is_default", sa.Boolean(), nullable=False, server_default=sa.text("false")
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True),
            server_default=sa.func.now(), nullable=False,
        ),
        sa.UniqueConstraint("user_id", "slug", name="uq_shelves_user_slug"),
    )
    op.create_index("ix_shelves_user_id", "shelves", ["user_id"])

    # One default shelf per user, and the partial unique index is what enforces
    # "one". A plain UNIQUE(user_id, is_default) would instead forbid a user from
    # having two *non*-default shelves, which is the opposite of the rule.
    op.create_index(
        "uq_shelves_one_default", "shelves", ["user_id"],
        unique=True, postgresql_where=sa.text("is_default"),
    )

    for table in ("files", "threads"):
        op.add_column(table, sa.Column("shelf_id", UUID(as_uuid=True), nullable=True))
        op.create_foreign_key(
            f"fk_{table}_shelf_id", table, "shelves",
            ["shelf_id"], ["id"], ondelete="SET NULL",
        )
    op.create_index("ix_files_shelf_id", "files", ["shelf_id"])
    op.create_index("ix_threads_shelf_id", "threads", ["shelf_id"])

    # The name is what a user sees before they have renamed anything, so it
    # reads as a place rather than as a migration artifact.
    op.execute(
        sa.text(
            """
            INSERT INTO shelves (user_id, name, slug, preset, is_default)
            SELECT id, 'My documents', '', 'general', true FROM users
            """
        )
    )


def downgrade() -> None:
    for table in ("files", "threads"):
        op.drop_constraint(f"fk_{table}_shelf_id", table, type_="foreignkey")
        op.drop_index(f"ix_{table}_shelf_id", table_name=table)
        op.drop_column(table, "shelf_id")
    op.drop_index("uq_shelves_one_default", table_name="shelves")
    op.drop_index("ix_shelves_user_id", table_name="shelves")
    op.drop_table("shelves")
