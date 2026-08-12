"""Raise the shipped token budget to cover answer review.

Review adds model calls to a question that previously made only the agent's own:
a critic when the free citation check cannot settle the answer, a reviser when
it finds something to repair, and — when `agent.review.max_rounds` allows it — a
second research pass. A hard question can therefore cost roughly twice what it
did, so the out-of-the-box ceiling doubles with it.

Two things this deliberately does NOT do:

- It does not touch `user_limits`. Those rows are per-user overrides an
  administrator set on purpose; a migration that "helpfully" raised them would
  silently undo deliberate decisions.
- It does not raise a `global_limits` row that has been moved off the shipped
  default. A deployment that lowered its ceiling did so for a reason — a small
  VPS, a metered provider key — and doubling its spend without being asked is
  not a migration's call. Such deployments raise it themselves, from the admin
  console or with a direct UPDATE, once they have decided review is worth it.

The practical effect is that this changes the default for fresh installs and
for deployments still on it, and leaves every deliberate choice alone.

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-05
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

# (column, shipped-until-now, new shipped default)
_BUDGETS = (
    ("tokens_per_day", 150_000, 300_000),
    ("tokens_per_month", 2_000_000, 4_000_000),
)


def _set_defaults(pairs) -> None:
    for column, value in pairs:
        op.alter_column(
            "global_limits", column, server_default=str(value),
            existing_type=sa.BigInteger, existing_nullable=False,
        )


def upgrade() -> None:
    _set_defaults((col, new) for col, _old, new in _BUDGETS)
    for column, old, new in _BUDGETS:
        # `WHERE = old` is the whole safety mechanism: it moves deployments
        # still on the shipped value and no others.
        op.execute(
            sa.text(
                f"UPDATE global_limits SET {column} = :new WHERE {column} = :old"  # noqa: S608
            ).bindparams(new=new, old=old)
        )


def downgrade() -> None:
    _set_defaults((col, old) for col, old, _new in _BUDGETS)
    for column, old, new in _BUDGETS:
        op.execute(
            sa.text(
                f"UPDATE global_limits SET {column} = :old WHERE {column} = :new"  # noqa: S608
            ).bindparams(new=new, old=old)
        )
