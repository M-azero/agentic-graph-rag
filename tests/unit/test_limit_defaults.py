"""The shipped limits are stated in three places; they have to agree.

  1. `GlobalLimit`'s ORM column defaults — used when a row is created through
     SQLAlchemy, which `PUT /admin/limits` does when the row is missing.
  2. the migrations' `server_default`s — used when Alembic creates the table.
  3. `limits.service._DEFAULTS` — used when there is no row at all.

They had drifted: the ORM said 1.5M/20M tokens where the other two said
300k/4M, so which quota a deployment got depended on which code path created
the row. The comment in `_DEFAULTS` asserted they mirrored each other, which is
how a 5x discrepancy survives review.

This test is the mirror, mechanically. It reads the migrations rather than
restating their numbers, so raising a ceiling in one place and not the others
fails here instead of in someone's provider bill.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from graphrag.db.models import LIMIT_COLUMNS, GlobalLimit
from graphrag.limits.service import _DEFAULTS

_MIGRATIONS = pathlib.Path("migrations/versions")


def _orm_defaults() -> dict[str, int]:
    out = {}
    for column in LIMIT_COLUMNS:
        default = GlobalLimit.__table__.columns[column].default
        assert default is not None, f"{column} has no ORM default"
        out[column] = default.arg
    return out


# One column per line, so the pattern must not cross a newline: with DOTALL a
# non-greedy `.*?` happily runs from `sa.Column("id"` past the end of that line
# to the *next* column's server_default, pairing a name with someone else's
# number. Line-anchored is both correct and easier to be sure of.
_COLUMN = re.compile(r'sa\.Column\(\s*"(\w+)"[^\n]*?server_default="(\d+)"')

# 0003 drives its change from a table of (column, old, new) tuples.
_REBUDGET = re.compile(r'\(\s*"(\w+)",\s*([\d_]+),\s*([\d_]+)\s*\)')


def _migration_defaults() -> dict[str, int]:
    """The effective server_default per column, applying migrations in order.

    0001 creates them; later migrations may raise one. Read from the files
    rather than restated, so this cannot drift the way the thing it checks did.
    """
    initial = (_MIGRATIONS / "0001_initial.py").read_text(encoding="utf-8")
    # Only the global_limits block: user_limits repeats every column name with
    # no defaults at all, and matching those would quietly poison the result.
    block = initial.split('"global_limits"', 1)[1].split("op.create_table", 1)[0]
    defaults = {name: int(value) for name, value in _COLUMN.findall(block)}

    for path in sorted(_MIGRATIONS.glob("0*.py"))[1:]:
        text = path.read_text(encoding="utf-8")
        if "global_limits" not in text:
            continue
        for name, _old, new in _REBUDGET.findall(text):
            if name in defaults:
                defaults[name] = int(new.replace("_", ""))
    return defaults


def test_the_migrations_are_parsed_at_all():
    """Guard the guard: a regex that silently matches nothing would make every
    assertion below vacuously true."""
    found = _migration_defaults()
    assert set(found) == set(LIMIT_COLUMNS), f"parsed only {sorted(found)}"


@pytest.mark.parametrize("column", LIMIT_COLUMNS)
def test_orm_defaults_match_the_migrations(column):
    assert _orm_defaults()[column] == _migration_defaults()[column], (
        f"{column}: the ORM default and the migration's server_default disagree, "
        "so the quota depends on which code path created the row"
    )


@pytest.mark.parametrize("column", LIMIT_COLUMNS)
def test_fallback_defaults_match_the_migrations(column):
    assert _DEFAULTS[column] == _migration_defaults()[column], (
        f"{column}: limits.service._DEFAULTS disagrees with the migration, so a "
        "deployment with no global_limits row gets a different quota"
    )


def test_the_review_budget_is_the_raised_one():
    """0003 doubled these for answer review. A revert would be a quiet
    regression to a ceiling that no longer fits the work."""
    assert _DEFAULTS["tokens_per_day"] == 300_000
    assert _DEFAULTS["tokens_per_month"] == 4_000_000
