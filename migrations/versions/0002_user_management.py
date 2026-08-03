"""Password reset, login lockout, and purpose-scoped verification codes.

Additive only. Every column is nullable or carries a server default, so it
applies to a populated database without a backfill and without a lock beyond
the catalogue update — and `downgrade()` is an exact inverse.

Password reset needs no new table: `email_otps.purpose` has existed since 0001
with `server_default='verify'`, so a reset code is the same row with a
different purpose. What it *does* need is the index below, because every OTP
lookup becomes purpose-scoped (an unscoped read would let a reset code verify
an account) and `ix_email_otps_user_id` alone would leave that filter to a
recheck on every candidate row.

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-03
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Read by the account page ("last changed ...") and by the admin table; also
    # the marker that a forced reset actually happened.
    op.add_column(
        "users", sa.Column("password_changed_at", sa.DateTime(timezone=True))
    )
    # Lockout state. In Postgres rather than Redis deliberately: the Redis
    # counters elsewhere in this codebase fail open, which is correct for a
    # quota and wrong for a lockout — it would let an attacker remove the
    # control by breaking the cache. NOT NULL with a default so existing rows
    # start at zero rather than needing a backfill pass.
    op.add_column(
        "users",
        sa.Column("failed_logins", sa.Integer, nullable=False, server_default="0"),
    )
    op.add_column("users", sa.Column("locked_until", sa.DateTime(timezone=True)))

    op.create_index(
        "ix_email_otps_user_purpose", "email_otps", ["user_id", "purpose"]
    )


def downgrade() -> None:
    op.drop_index("ix_email_otps_user_purpose", table_name="email_otps")
    op.drop_column("users", "locked_until")
    op.drop_column("users", "failed_logins")
    op.drop_column("users", "password_changed_at")
