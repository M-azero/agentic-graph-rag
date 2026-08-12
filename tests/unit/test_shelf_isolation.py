"""Shelves are a storage boundary, so the tests are about the corpus name.

Everything downstream — Neo4j's `(corpus, key)` constraints, the DuckDB
filename, the checkpointer's thread namespace — keys on the string
`corpus_for` returns. If two shelves can ever produce the same one, or a shelf
can produce another *tenant's*, that is a cross-corpus leak with no other line
of defence behind it.

The compatibility case matters just as much: the default shelf must resolve to
the bare tenant id, because that is where every document ingested before shelves
existed already lives.
"""

import pytest

from graphrag.config.settings import Settings
from graphrag.container import (
    SHELF_SEPARATOR,
    Container,
    corpus_for,
    sanitize_slug,
    sanitize_user,
)

# --------------------------------------------------------------------------
# The corpus name
# --------------------------------------------------------------------------

def test_the_default_shelf_is_the_bare_tenant_id():
    """The migration guarantee. If this changes, every pre-shelves document
    becomes unreachable — stored under a corpus nothing queries."""
    for empty in ("", None, "   ", "---"):
        assert corpus_for("alice-ab12", empty) == "alice-ab12"


def test_a_named_shelf_suffixes_the_tenant():
    assert corpus_for("alice-ab12", "maths") == "alice-ab12.maths"


def test_the_separator_cannot_occur_in_a_tenant_or_a_slug():
    """This is what makes `{tenant}.{slug}` unambiguous. A dash or underscore
    would not have been: tenant ids may legitimately contain both, so
    `a-1--b` could be read as two different (tenant, shelf) pairs."""
    nasty = "a.b.c-d_e..f"
    assert SHELF_SEPARATOR not in sanitize_user(nasty)
    assert SHELF_SEPARATOR not in sanitize_slug(nasty)
    # Therefore: exactly one separator in a shelf corpus, none in a default one.
    assert corpus_for(sanitize_user(nasty), sanitize_slug(nasty)).count(SHELF_SEPARATOR) == 1
    assert corpus_for(sanitize_user(nasty)).count(SHELF_SEPARATOR) == 0


@pytest.mark.parametrize(
    ("tenant_a", "slug_a", "tenant_b", "slug_b"),
    [
        # The collision a dash separator would have allowed.
        ("alice-1", "b-2", "alice-1-b", "2"),
        # A shelf must never be able to name another tenant's default corpus.
        ("alice-1", "bob-2", "alice-1-bob-2", ""),
        ("alice", "bob", "bob", "alice"),
    ],
)
def test_no_two_scopes_collide(tenant_a, slug_a, tenant_b, slug_b):
    assert corpus_for(tenant_a, slug_a) != corpus_for(tenant_b, slug_b)


def test_slugs_are_bounded_and_storage_safe():
    """The corpus becomes a filename under the DuckDB provider, so a slug that
    escaped the character class would be a path, not a name."""
    slug = sanitize_slug("../../etc/passwd" + "x" * 200)
    assert "/" not in slug and "." not in slug and ".." not in slug
    assert len(slug) <= 32


# --------------------------------------------------------------------------
# Tenant resolution and caching
# --------------------------------------------------------------------------

def _container() -> Container:
    from graphrag.config.settings import Secrets

    return Container(Settings(), Secrets())


def test_scope_puts_shelves_in_one_database_per_user():
    """Only the corpus varies per shelf. Under `per_tenant_database` a database
    per shelf would multiply an Enterprise-only resource by however many
    subjects someone keeps, for isolation the corpus tag already gives."""
    c = _container()
    c.settings.tenancy.per_tenant_database = True

    db_default, corpus_default = c._resolve_scope("alice", "")
    db_shelf, corpus_shelf = c._resolve_scope("alice", "maths")

    assert db_default == db_shelf
    assert corpus_default != corpus_shelf


def test_evict_tenant_drops_every_shelf_of_that_user_and_nobody_elses():
    """Purge relies on this: a cached tenant holds store wrappers pointing at
    data that no longer exists, and under DuckDB an open file handle that has to
    be released before the file can be unlinked."""
    c = _container()
    c._tenants.update(
        {
            "alice": object(),
            "alice.maths": object(),
            "alice.code": object(),
            # Shares a prefix but is a different account — the separator is
            # what keeps `alice-2` from being read as a shelf of `alice`.
            "alice-2": object(),
            "alice-2.maths": object(),
            "bob": object(),
        }
    )

    assert c.evict_tenant("alice") == 3
    assert set(c._tenants) == {"alice-2", "alice-2.maths", "bob"}
