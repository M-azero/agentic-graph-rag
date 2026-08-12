"""Request fields that reach a model or a retriever are bounded.

None of these were. `question` had `min_length=1` and no maximum, so a
multi-megabyte body was embedded, screened by the guard, and then resent on
every turn of the agent's tool loop — for one unit of message quota. `subjects`
was worse than large: it *fans out*, one full hybrid retrieval per entry.

The rate limiter counts requests, not what a request costs, so amplification
inside a single request is a hole it cannot see.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from graphrag.api.schemas import CompareRequest, QueryRequest, SearchRequest


def test_a_reasonable_question_is_accepted():
    assert QueryRequest(question="what does the contract say about renewal?")


def test_an_enormous_question_is_refused():
    with pytest.raises(ValidationError):
        QueryRequest(question="x" * 100_000)


def test_an_empty_question_is_still_refused():
    with pytest.raises(ValidationError):
        QueryRequest(question="")


@pytest.mark.parametrize("field", ["thread_id", "model"])
def test_identifier_fields_are_bounded(field):
    """`thread_id` becomes a checkpointer key and `model` a cache key; neither
    should be caller-sized."""
    with pytest.raises(ValidationError):
        QueryRequest(question="hi", **{field: "x" * 5000})


def test_search_k_is_bounded_on_both_sides():
    assert SearchRequest(query="q", k=8).k == 8
    with pytest.raises(ValidationError):
        SearchRequest(query="q", k=0)
    with pytest.raises(ValidationError):
        SearchRequest(query="q", k=-1)   # would slice from the wrong end
    with pytest.raises(ValidationError):
        SearchRequest(query="q", k=100_000)


def test_compare_still_needs_two_subjects():
    with pytest.raises(ValidationError):
        CompareRequest(subjects=["only one"])


def test_compare_accepts_a_realistic_comparison():
    assert CompareRequest(subjects=["Postgres", "MySQL"], aspects=["licensing"])


def test_compare_refuses_a_fan_out():
    """The amplification: each subject is its own retrieval pass."""
    with pytest.raises(ValidationError):
        CompareRequest(subjects=[f"subject {i}" for i in range(500)])


def test_compare_refuses_one_enormous_subject():
    """Capping the list length alone leaves the same cost in fewer entries."""
    with pytest.raises(ValidationError):
        CompareRequest(subjects=["ok", "x" * 50_000])


def test_compare_refuses_an_enormous_aspect():
    with pytest.raises(ValidationError):
        CompareRequest(subjects=["a", "b"], aspects=["x" * 50_000])
