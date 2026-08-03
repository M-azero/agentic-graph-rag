"""Entity extraction: keeping notation out of the knowledge graph.

Technical notes are full of symbols and grammar rules. Small local models happily
return "$", "AB" and "11" as entities, which clutters graph expansion. The prompt
asks for restraint; this is the floor that doesn't depend on the model obeying.
"""

import json
from types import SimpleNamespace

import pytest

from graphrag.ingestion.extraction.graph_extractor import LLMGraphExtractor, _is_nameable


class _FakeLLM:
    def __init__(self, payload) -> None:
        self._payload = payload

    def invoke(self, messages):
        if isinstance(self._payload, Exception):
            raise self._payload
        body = self._payload if isinstance(self._payload, str) else json.dumps(self._payload)
        return SimpleNamespace(content=body)


def _extract(payload):
    return LLMGraphExtractor(_FakeLLM(payload)).extract("some text")


@pytest.mark.parametrize("name", ["$", "!", "A", "D", "11", "0140", "->", "  "])
def test_notation_is_not_an_entity(name):
    assert not _is_nameable(name)


@pytest.mark.parametrize(
    "name", ["TM", "CFG", "Turing Machine", "Chomsky Normal Form", "Emil Eifrem", "A5 tape"]
)
def test_real_names_survive(name):
    # "TM" must pass: a length rule that dropped "AB" would drop this too.
    assert _is_nameable(name)


def test_symbol_entities_are_dropped():
    result = _extract(
        {
            "entities": [
                {"name": "Turing Machine", "type": "CONCEPT"},
                {"name": "$", "type": "symbol"},
                {"name": "A", "type": "variable"},
                {"name": "11", "type": "string"},
            ],
            "relations": [],
        }
    )
    assert [e.name for e in result.entities] == ["Turing Machine"]


def test_relations_to_dropped_entities_go_too():
    # A relation is only meaningful if both endpoints survived.
    result = _extract(
        {
            "entities": [
                {"name": "Chomsky Normal Form", "type": "CONCEPT"},
                {"name": "Context Free Grammar", "type": "CONCEPT"},
                {"name": "BC", "type": "rule"},
            ],
            "relations": [
                {
                    "source": "Chomsky Normal Form",
                    "target": "Context Free Grammar",
                    "type": "PART_OF",
                },
                {"source": "Chomsky Normal Form", "target": "$", "type": "USES"},
            ],
        }
    )
    assert len(result.entities) == 3  # "BC" has a letter -> the model's call, not ours
    assert [r.type for r in result.relations] == ["PART_OF"]


def test_names_are_stripped():
    result = _extract({"entities": [{"name": "  Turing Machine \n", "type": "CONCEPT"}]})
    assert result.entities[0].name == "Turing Machine"


def test_empty_extraction_is_fine():
    result = _extract({"entities": [], "relations": []})
    assert result.entities == [] and result.relations == []
    assert result.failed is False


def test_unparseable_reply_does_not_raise():
    result = _extract("I cannot produce JSON for this.")
    assert result.entities == [] and result.relations == []
    # The model answered, it just answered badly. That is a poor chunk, not a
    # broken provider, so it must not be reported as an ingest failure.
    assert result.failed is False


def test_a_dead_provider_is_distinguishable_from_an_empty_chunk():
    """Both give back no entities. Only one of them means the knowledge graph
    is missing data the user should be told about."""
    result = _extract(RuntimeError("403 PERMISSION_DENIED"))
    assert result.entities == [] and result.relations == []
    assert result.failed is True
