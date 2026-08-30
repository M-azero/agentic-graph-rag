"""The canonical step registry: every block the pipeline inspector can draw.

This module is the contract between the instrumentation and the UI. Code calls
`step(GUARD_INPUT, ...)` with a name from here; `GET /api/trace/pipelines`
serves the nodes and edges built from the same table. So a step renamed in one
place fails `tests/unit/test_trace_steps_registry.py` rather than silently
dropping off the diagram.

`source` is the file the block lives in, so the inspector can point a reader at
the code rather than describing it. A path, not a path:line — line numbers rot
on the first edit and nothing would check them.
"""

from __future__ import annotations

from dataclasses import dataclass

QUERY = "query"
INGEST = "ingest"
REVIEW = "review"

PIPELINES = (QUERY, INGEST, REVIEW)


@dataclass(frozen=True)
class StepSpec:
    """One block on the diagram."""

    name: str
    label: str
    kind: str
    pipeline: str
    source: str
    summary: str = ""
    # One call per item (a tool call, a chunk extraction). The diagram draws
    # these as a stack rather than a single box.
    repeats: bool = False
    # Drawn but not always visited. The UI greys these rather than marking them
    # missing, which for an optional stage is not a fault.
    optional: bool = False


# -- query ---------------------------------------------------------------------

GUARD_INPUT = "guard.input"
GATE_PROBE = "gate.probe"
AGENT_RUN = "agent.run"
TOOL_CALL = "tool.call"
RETRIEVAL_VECTOR = "retrieval.vector"
RETRIEVAL_GRAPH = "retrieval.graph"
RETRIEVAL_KEYWORD = "retrieval.keyword"
RETRIEVAL_FUSE = "retrieval.fuse"
RETRIEVAL_RERANK = "retrieval.rerank"
CHECK_CITATIONS = "check.citations"
GUARD_OUTPUT = "guard.output"
BILLING = "billing"

# -- review --------------------------------------------------------------------

REVIEW_RESEARCH = "review.research"
REVIEW_CHECK = "review.check"
REVIEW_WIDEN = "review.widen"
REVIEW_REVISE = "review.revise"
REVIEW_FINALIZE = "review.finalize"

# -- ingest --------------------------------------------------------------------

INGEST_LOAD = "ingest.load"
INGEST_CHUNK = "ingest.chunk"
INGEST_EMBED = "ingest.embed"
INGEST_VECTOR_UPSERT = "ingest.vector_upsert"
INGEST_GRAPH_CHUNKS = "ingest.graph_chunks"
INGEST_LINK_SEQUENCE = "ingest.link_sequence"
INGEST_EXTRACT = "ingest.extract"
INGEST_GRAPH_WRITE = "ingest.graph_write"
INGEST_RESOLVE = "ingest.resolve_entities"
INGEST_COMMUNITIES = "ingest.communities"

# Emitted by the recorder itself when a run exceeds `trace.max_steps`.
TRUNCATED = "trace.truncated"


_SPECS: tuple[StepSpec, ...] = (
    # -- query -----------------------------------------------------------------
    StepSpec(
        GUARD_INPUT, "Input guard", "guard", QUERY,
        "src/graphrag/safety/guardrails.py",
        "Screens the question before a token is spent: prompt injection, "
        "jailbreak, off-topic, pasted secrets.",
        optional=True,
    ),
    StepSpec(
        GATE_PROBE, "Closed-domain gate", "gate", QUERY,
        "src/graphrag/api/routers/query.py",
        "One probe retrieval. If nothing clears retrieval.min_relevance the "
        "question is refused as not covered by the knowledge base.",
        optional=True,
    ),
    StepSpec(
        AGENT_RUN, "Agent (ReAct loop)", "agent", QUERY,
        "src/graphrag/agent/graph.py",
        "The tool-using loop: the model reads the question, decides which "
        "retrieval strategies to run, then writes the answer.",
    ),
    StepSpec(
        TOOL_CALL, "Tool", "tool", QUERY,
        "src/graphrag/agent/tools.py",
        "One of the nine tools running: hybrid_search, vector_search, "
        "graph_neighbors, expand_subgraph, get_entity, fulltext_search, "
        "compare, read_around, global_search.",
        repeats=True,
    ),
    StepSpec(
        RETRIEVAL_VECTOR, "Vector search", "retrieval", QUERY,
        "src/graphrag/retrieval/vector.py",
        "Embed the query, nearest-neighbour over the chunk index.",
        repeats=True,
    ),
    StepSpec(
        RETRIEVAL_GRAPH, "Graph-augmented search", "retrieval", QUERY,
        "src/graphrag/retrieval/graph_augmented.py",
        "Seed entities from the query, then pull the chunks that mention them "
        "and their graph neighbourhood.",
        repeats=True,
    ),
    StepSpec(
        RETRIEVAL_KEYWORD, "Keyword search", "retrieval", QUERY,
        "src/graphrag/storage/graph/base.py",
        "Exact fulltext lookup over chunk nodes.",
        repeats=True,
    ),
    StepSpec(
        RETRIEVAL_FUSE, "RRF fusion", "retrieval", QUERY,
        "src/graphrag/retrieval/fusion.py",
        "Merge the three ranked lists without needing comparable scores: "
        "score = sum of 1/(60 + rank + 1).",
        repeats=True,
    ),
    StepSpec(
        RETRIEVAL_RERANK, "Rerank", "rerank", QUERY,
        "src/graphrag/retrieval/reranker.py",
        "Order the fused candidates by true relevance. These are the scores "
        "the closed-domain gate is calibrated against.",
        repeats=True,
    ),
    StepSpec(
        CHECK_CITATIONS, "Citation check", "check", QUERY,
        "src/graphrag/agent/review/citations.py",
        "Deterministic, no model call: which sources the answer actually "
        "cited, which it invented, whether it refused.",
    ),
    StepSpec(
        GUARD_OUTPUT, "Output guard", "guard", QUERY,
        "src/graphrag/safety/guardrails.py",
        "PII and secret redaction, groundedness, system-prompt leak. Can block "
        "or rewrite the answer before it reaches the browser.",
        optional=True,
    ),
    StepSpec(
        BILLING, "Metering", "billing", QUERY,
        "src/graphrag/usage/meter.py",
        "Prompt and completion tokens across every model call in the run - the "
        "agent turns, the reranker, the critic.",
    ),
    # -- review ----------------------------------------------------------------
    StepSpec(
        REVIEW_RESEARCH, "Research", "agent", REVIEW,
        "src/graphrag/agent/review/graph.py",
        "Run the ReAct agent and collect its draft plus its evidence.",
        repeats=True,
    ),
    StepSpec(
        REVIEW_CHECK, "Critic", "check", REVIEW,
        "src/graphrag/agent/review/critic.py",
        "Free citation check first; a model call only when that settles "
        "nothing. Decides ship / revise / retrieve more.",
        repeats=True,
    ),
    StepSpec(
        REVIEW_WIDEN, "Widen", "retrieval", REVIEW,
        "src/graphrag/agent/review/graph.py",
        "Cheap escalation: walk the :NEXT chain from what we already have, and "
        "double the retrieval budget for the next pass.",
        optional=True,
    ),
    StepSpec(
        REVIEW_REVISE, "Revise", "llm", REVIEW,
        "src/graphrag/agent/review/revise.py",
        "Repair the draft against what the critic found wrong.",
        optional=True,
    ),
    StepSpec(
        REVIEW_FINALIZE, "Finalize", "check", REVIEW,
        "src/graphrag/agent/review/graph.py",
        "Re-check citations against the text actually being sent, order the "
        "sources, report the outcome.",
    ),
    # -- ingest ----------------------------------------------------------------
    StepSpec(
        INGEST_LOAD, "Load", "loader", INGEST,
        "src/graphrag/ingestion/loaders/__init__.py",
        "PDF, Word, HTML, CSV, text or image to plain text. A page whose text "
        "layer is thinner than ocr.min_text_chars goes through OCR first.",
        repeats=True,
    ),
    StepSpec(
        INGEST_CHUNK, "Chunk", "chunking", INGEST,
        "src/graphrag/ingestion/chunking/router.py",
        "Split the document. With route_per_document on, the strategy is "
        "picked per file: token for tabular, recursive for headed prose, "
        "semantic for OCR output.",
        repeats=True,
    ),
    StepSpec(
        INGEST_EMBED, "Embed", "embedding", INGEST,
        "src/graphrag/embeddings/base.py",
        "One vector per chunk.",
        repeats=True,
    ),
    StepSpec(
        INGEST_VECTOR_UPSERT, "Vector upsert", "store", INGEST,
        "src/graphrag/storage/vector",
        "Write the chunks and their vectors into this shelf's index.",
        repeats=True,
    ),
    StepSpec(
        INGEST_GRAPH_CHUNKS, "Graph chunk nodes", "store", INGEST,
        "src/graphrag/storage/graph/base.py",
        "Chunk nodes in the graph, so fulltext search and MENTIONS edges have "
        "something to attach to when vectors live elsewhere.",
        repeats=True, optional=True,
    ),
    StepSpec(
        INGEST_LINK_SEQUENCE, "Link :NEXT chain", "store", INGEST,
        "src/graphrag/storage/graph/base.py",
        "Chain the chunks in reading order - what read_around and the review "
        "loop widen step walk.",
        repeats=True,
    ),
    StepSpec(
        INGEST_EXTRACT, "Extract entities", "llm", INGEST,
        "src/graphrag/ingestion/extraction/graph_extractor.py",
        "One LLM call per chunk, run concurrently. By far the slowest part of "
        "an ingest.",
        repeats=True, optional=True,
    ),
    StepSpec(
        INGEST_GRAPH_WRITE, "Graph write", "store", INGEST,
        "src/graphrag/storage/graph/base.py",
        "Merge the entities and relations, and link each chunk to what it "
        "mentions. Serial: concurrent MERGEs on the same key only fight for "
        "locks.",
        repeats=True, optional=True,
    ),
    StepSpec(
        INGEST_RESOLVE, "Resolve duplicates", "enrich", INGEST,
        "src/graphrag/ingestion/enrich.py",
        "Fold entities the per-chunk extractor could not see were the same "
        "thing - Acme and Acme Robotics.",
        optional=True,
    ),
    StepSpec(
        INGEST_COMMUNITIES, "Community summaries", "enrich", INGEST,
        "src/graphrag/ingestion/enrich.py",
        "Cluster the graph and describe each cluster, so corpus-wide questions "
        "have something to read.",
        optional=True,
    ),
    # -- recorder-emitted ------------------------------------------------------
    StepSpec(
        TRUNCATED, "Truncated", "meta", QUERY,
        "src/graphrag/trace/recorder.py",
        "The run produced more steps than trace.max_steps; the rest were "
        "dropped.",
        optional=True,
    ),
)

SPECS: dict[str, StepSpec] = {spec.name: spec for spec in _SPECS}

# Every canonical name, for the registry test.
NAMES = frozenset(SPECS)


# Edges are declared per pipeline rather than derived from parentage: the
# diagram is the *designed* flow, which includes paths a given run did not take
# (a blocked question never reaches the agent). Parentage describes one run;
# this describes the system.
EDGES: dict[str, tuple[tuple[str, str], ...]] = {
    QUERY: (
        (GUARD_INPUT, GATE_PROBE),
        (GATE_PROBE, AGENT_RUN),
        (AGENT_RUN, TOOL_CALL),
        (TOOL_CALL, RETRIEVAL_VECTOR),
        (TOOL_CALL, RETRIEVAL_GRAPH),
        (TOOL_CALL, RETRIEVAL_KEYWORD),
        (RETRIEVAL_VECTOR, RETRIEVAL_FUSE),
        (RETRIEVAL_GRAPH, RETRIEVAL_FUSE),
        (RETRIEVAL_KEYWORD, RETRIEVAL_FUSE),
        (RETRIEVAL_FUSE, RETRIEVAL_RERANK),
        (RETRIEVAL_RERANK, AGENT_RUN),
        (AGENT_RUN, CHECK_CITATIONS),
        (CHECK_CITATIONS, GUARD_OUTPUT),
        (GUARD_OUTPUT, BILLING),
    ),
    REVIEW: (
        (REVIEW_RESEARCH, REVIEW_CHECK),
        (REVIEW_CHECK, REVIEW_WIDEN),
        (REVIEW_CHECK, REVIEW_REVISE),
        (REVIEW_CHECK, REVIEW_FINALIZE),
        (REVIEW_WIDEN, REVIEW_RESEARCH),
        (REVIEW_REVISE, REVIEW_FINALIZE),
    ),
    INGEST: (
        (INGEST_LOAD, INGEST_CHUNK),
        (INGEST_CHUNK, INGEST_EMBED),
        (INGEST_EMBED, INGEST_VECTOR_UPSERT),
        (INGEST_VECTOR_UPSERT, INGEST_GRAPH_CHUNKS),
        (INGEST_GRAPH_CHUNKS, INGEST_LINK_SEQUENCE),
        (INGEST_CHUNK, INGEST_EXTRACT),
        (INGEST_EXTRACT, INGEST_GRAPH_WRITE),
        (INGEST_GRAPH_WRITE, INGEST_RESOLVE),
        (INGEST_LINK_SEQUENCE, INGEST_RESOLVE),
        (INGEST_RESOLVE, INGEST_COMMUNITIES),
    ),
}


def pipeline_graph(pipeline: str) -> dict:
    """Nodes + edges for one pipeline, in the shape the frontend draws."""
    nodes = [
        {
            "name": s.name,
            "label": s.label,
            "kind": s.kind,
            "source": s.source,
            "summary": s.summary,
            "repeats": s.repeats,
            "optional": s.optional,
        }
        for s in _SPECS
        if s.pipeline == pipeline and s.name != TRUNCATED
    ]
    return {
        "pipeline": pipeline,
        "nodes": nodes,
        "edges": [{"from": a, "to": b} for a, b in EDGES.get(pipeline, ())],
    }


def all_graphs() -> list[dict]:
    return [pipeline_graph(p) for p in PIPELINES]


__all__ = [
    "EDGES", "NAMES", "PIPELINES", "SPECS", "StepSpec", "all_graphs", "pipeline_graph",
]
