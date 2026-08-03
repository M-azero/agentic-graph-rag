"""Domain types passed between layers. Plain dataclasses — cheap and framework-free."""

from __future__ import annotations

import enum
import hashlib
from dataclasses import dataclass, field
from typing import Any


class AnswerStyle(enum.StrEnum):
    """How the agent should phrase its final answer."""

    CONCISE = "concise"
    DETAILED = "detailed"
    TECHNICAL = "technical"
    ELI5 = "eli5"


def _stable_id(*parts: str) -> str:
    return hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()[:16]


@dataclass(slots=True)
class Document:
    """A source document before chunking."""

    source: str  # path or URL
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return _stable_id(self.source)


@dataclass(slots=True)
class Chunk:
    """A retrievable unit of text with its embedding."""

    doc_id: str
    index: int
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    @property
    def id(self) -> str:
        return _stable_id(self.doc_id, str(self.index))


@dataclass(slots=True)
class Entity:
    """A node in the knowledge graph."""

    name: str
    type: str = "Concept"
    description: str = ""

    @property
    def key(self) -> str:
        # Case-insensitive identity so "OpenAI" and "openai" merge.
        return self.name.strip().lower()


@dataclass(slots=True)
class Relation:
    """A typed edge between two entities."""

    source: str
    target: str
    type: str
    description: str = ""


@dataclass(slots=True)
class RetrievedChunk:
    """A chunk returned by a retriever, with provenance and score."""

    chunk_id: str
    text: str
    source: str
    score: float
    retriever: str = "vector"  # which tool produced it
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QueryResult:
    """The agent's final answer plus the evidence it used."""

    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    # What this one question cost, prompt side included. Zero when nothing
    # metered the run (the CLI, or a caller that passed no meter) — never a
    # claim that the run was free.
    input_tokens: int = 0
    output_tokens: int = 0
