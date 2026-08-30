"""The agent's tools. Each is a real capability over the store, not a wrapper
around one vector call — that's what makes this an *agentic* graph RAG: the model
chooses among genuinely different retrieval strategies.

Tools return text for the model to read, and also record the chunks they surfaced
into a shared collector so the API can report exact sources.

Everything a tool returns is document-derived and therefore untrusted: chunk
text obviously, but also entity names, descriptions, and community summaries —
those were extracted *by an LLM from user documents* and can carry injected
instructions just as easily. So every output is sanitized and wrapped in the
untrusted-data envelope the system prompt tells the model to treat as data only,
and capped so a hostile document can't flood the context."""

from __future__ import annotations

import functools
import inspect
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from langchain_core.tools import StructuredTool

from graphrag.agent.prompts import wrap_untrusted
from graphrag.core.sanitize import sanitize_untrusted
from graphrag.core.types import RetrievedChunk
from graphrag.embeddings.base import Embedder
from graphrag.retrieval.hybrid import HybridRetriever
from graphrag.retrieval.plan import active_plan
from graphrag.retrieval.vector import VectorRetriever
from graphrag.storage.graph.base import GraphStore
from graphrag.trace import step
from graphrag.trace import steps as trace_steps

_MAX_CHUNK_CHARS = 4000
_MAX_TOOL_OUTPUT_CHARS = 8000

# Citable provenances for answers that come from the graph rather than from one
# chunk. `SYSTEM_PROMPT` names these, and citation validation accepts them.
GRAPH_LABEL = "knowledge-graph"
COMMUNITY_LABEL = "community-summaries"


@dataclass(frozen=True, eq=False)
class ToolContext:
    """What the tools read. Every field is tenant-scoped and immutable.

    Nothing here varies per query, which is what lets one compiled agent serve
    a tenant's whole traffic instead of being rebuilt per question. The part
    that *is* per-query — which chunks the tools surfaced — travels in a
    ContextVar instead; see `collect_sources`.
    """

    vector: VectorRetriever
    hybrid: HybridRetriever
    graph: GraphStore
    embedder: Embedder
    top_k: int = 8
    graph_hops: int = 2


class SourceSink:
    """What one query's tools surfaced: chunks (deduplicated by chunk id) and
    the non-chunk source labels.

    Labels exist because three tools — `graph_neighbors`, `get_entity` and
    `global_search` — answer from the knowledge graph rather than from any
    single chunk. They still have a citable provenance (`knowledge-graph`,
    `community-summaries`), and without recording it, citation validation would
    read every graph-derived citation as fabricated.
    """

    __slots__ = ("_chunks", "_labels", "_lock", "_seen")

    def __init__(self) -> None:
        self._chunks: list[RetrievedChunk] = []
        self._seen: set[str] = set()
        self._labels: set[str] = set()
        # Tools can run concurrently — the model may issue parallel tool calls,
        # and LangGraph runs sync tools on executor threads — so the "have I
        # seen this chunk" check and the append have to happen together.
        self._lock = threading.Lock()

    def add(self, chunks: list[RetrievedChunk]) -> None:
        with self._lock:
            for chunk in chunks:
                if chunk.chunk_id not in self._seen:
                    self._seen.add(chunk.chunk_id)
                    self._chunks.append(chunk)

    def note_label(self, label: str) -> None:
        """Record a non-chunk provenance the model is allowed to cite."""
        with self._lock:
            self._labels.add(label)

    @property
    def chunks(self) -> list[RetrievedChunk]:
        """A snapshot — safe to read while the query is still running."""
        with self._lock:
            return list(self._chunks)

    @property
    def labels(self) -> set[str]:
        with self._lock:
            return set(self._labels)


_SINK: ContextVar[SourceSink | None] = ContextVar("graphrag_source_sink", default=None)


@contextmanager
def collect_sources(sink: SourceSink | None = None) -> Iterator[SourceSink]:
    """Scope a collector to one agent run — a fresh one, or a caller's.

    Passing an existing sink lets several runs accumulate into one citation
    list, which is what a review loop's second research pass needs: the answer
    it finally ships cites evidence from both passes.

    A ContextVar rather than an argument because the tools are baked into a
    compiled graph that outlives the query. Both of LangGraph's executors
    submit work under `copy_context()`, and LangChain does the same when it
    runs a sync tool in a thread, so the sink set here is visible inside every
    tool call — while two queries running as separate asyncio tasks each see
    their own. One session per task is assumed, which is how the API drives it:
    one request, one session.
    """
    sink = sink if sink is not None else SourceSink()
    token = _SINK.set(sink)
    try:
        yield sink
    finally:
        _SINK.reset(token)


def _src(source: str) -> str:
    """Source names come from uploaded filenames — sanitize and keep them to
    one attribute-safe line."""
    return sanitize_untrusted(source, 200).replace('"', "'").replace("\n", " ")


def _cap(text: str) -> str:
    if len(text) > _MAX_TOOL_OUTPUT_CHARS:
        return text[:_MAX_TOOL_OUTPUT_CHARS] + " …[truncated]"
    return text


def _format(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No results."
    # The [source: ...] tag stays outside the envelope: it is ours (the model
    # cites it), while the text inside the markers is the document's.
    #
    # The [chunk: ...] line is the handle `read_around` takes. It sits on its
    # own line rather than inside the source tag so citation parsing is
    # unaffected — and it is emitted at all because a tool the model cannot
    # call correctly costs schema tokens on every turn while being dead weight.
    return _cap(
        "\n\n".join(
            f"[source: {_src(c.source)}]\n[chunk: {_src(c.chunk_id)}]\n"
            + wrap_untrusted(_src(c.source), sanitize_untrusted(c.text.strip(), _MAX_CHUNK_CHARS))
            for c in chunks
        )
    )


def _graph_data(label: str, text: str) -> str:
    """Wrap graph-derived text (entity names/descriptions extracted from user
    documents by an LLM — untrusted like everything else).

    Carries a `[source: ...]` tag outside the envelope exactly as `_format`
    does, and records the label, so an answer grounded in the graph can cite a
    provenance that citation validation will recognise.
    """
    _note_label(label)
    return _cap(
        f"[source: {label}]\n"
        + wrap_untrusted(label, sanitize_untrusted(text, _MAX_TOOL_OUTPUT_CHARS))
    )


def _collect(chunks: list[RetrievedChunk]) -> None:
    """Record chunks against the query in flight, if there is one. Tools stay
    callable outside a session (tests, scripts) — they just don't record."""
    sink = _SINK.get()
    if sink is not None:
        sink.add(chunks)


def _note_label(label: str) -> None:
    sink = _SINK.get()
    if sink is not None:
        sink.note_label(label)


def _chunk_ids() -> list[str]:
    """Chunk ids the query in flight has collected so far, or [] outside one."""
    sink = _SINK.get()
    return [c.chunk_id for c in sink.chunks] if sink is not None else []


def build_tools(ctx: ToolContext) -> list[StructuredTool]:
    def _k() -> int:
        """How many results this call should return.

        The plan wins when one is bound, so an escalated pass can widen the net
        without recompiling the agent; otherwise the tenant's configured
        `top_k` stands.
        """
        plan = active_plan()
        return plan.top_k if plan else ctx.top_k

    def _hops() -> int:
        plan = active_plan()
        return plan.graph_hops if plan else ctx.graph_hops

    def hybrid_search(query: str) -> str:
        """Search the knowledge base with the strong hybrid retriever (vector +
        graph + keyword, reranked). Use this for most questions."""
        chunks = ctx.hybrid.retrieve(query, _k())
        _collect(chunks)
        return _format(chunks)

    def vector_search(query: str) -> str:
        """Semantic similarity search over text chunks. Use to find passages about a topic."""
        chunks = ctx.vector.retrieve(query, _k())
        _collect(chunks)
        return _format(chunks)

    def graph_neighbors(entity: str) -> str:
        """List the relationships around one entity in the knowledge graph. Use for
        'how is X connected?' questions."""
        return _graph_data(GRAPH_LABEL, ctx.graph.neighbors(entity, _hops()))

    def expand_subgraph(entities: str) -> str:
        """Explore the graph around several entities at once. Pass a comma-separated
        list of entity names."""
        names = [e.strip() for e in entities.split(",") if e.strip()]
        parts = [ctx.graph.neighbors(name, _hops()) for name in names]
        chunks = ctx.graph.chunks_for_entities(names, limit=_k())
        _collect(chunks)
        return _cap(_graph_data(GRAPH_LABEL, "\n\n".join(parts)) + "\n\n" + _format(chunks))

    def get_entity(name: str) -> str:
        """Look up what the graph knows about a single entity: its type, description,
        and directly connected entities."""
        info = ctx.graph.get_entity(name)
        if not info:
            return f"No entity named '{sanitize_untrusted(name, 200)}' found."
        return _graph_data(
            GRAPH_LABEL,
            f"{info['name']} ({info['type']})\n{info.get('description', '')}\n"
            f"Connected to: {', '.join(info.get('connected', []))}",
        )

    def fulltext_search(text: str) -> str:
        """Exact keyword search over chunks. Use when you know a specific term."""
        chunks = ctx.graph.fulltext_chunks(text, _k())
        _collect(chunks)
        return _format(chunks)

    def compare(subjects: str) -> str:
        """Gather evidence about several subjects for a side-by-side comparison.
        Pass a comma-separated list of subjects (e.g. 'Postgres, MySQL')."""
        names = [s.strip() for s in subjects.split(",") if s.strip()]
        blocks = []
        for name in names:
            chunks = ctx.hybrid.retrieve(name, max(3, _k() // 2))
            _collect(chunks)
            blocks.append(f"### {name}\n{_format(chunks)}")
        return _cap("\n\n".join(blocks))

    def read_around(chunk_id: str) -> str:
        """Read the passages immediately before and after a chunk you already
        retrieved. Pass the chunk_id from a [source: ...] result. Use when a
        passage is cut off mid-thought, or an answer looks incomplete because it
        starts or ends abruptly."""
        ids = [i.strip() for i in chunk_id.split(",") if i.strip()]
        if not ids:
            return "No results."
        chunks = ctx.graph.chunk_window(ids, before=1, after=1)
        _collect(chunks)
        return _format(chunks)

    def global_search(question: str) -> str:
        """Answer corpus-wide questions ('what are the main themes?', 'give an
        overview') from community summaries of the whole knowledge graph. Use when
        the question is about the collection as a whole, not one specific fact."""
        from graphrag.ingestion.enrich import global_search as _global

        return _graph_data(COMMUNITY_LABEL, _global(ctx.graph, ctx.embedder, question))

    def traced(fn):
        """Record one `tool.call` step around a tool.

        Wrapped here rather than inside each function so the nine stay readable
        and none of them can forget. `functools.wraps` is not enough on its own:
        `StructuredTool.from_function` derives the tool's *name*, *description*
        and *argument schema* from the wrapped callable, so the wrapper has to
        carry the docstring and the signature or the model is handed a tool
        called "wrapper" taking **kwargs. Hence the explicit copy of
        `__signature__` below.

        Inert off a traced run: `step()` is one ContextVar read and the call
        forwards unchanged.
        """

        @functools.wraps(fn)
        def call(*args, **kwargs):
            bound = inspect.signature(fn).bind(*args, **kwargs)
            # `args` as one dict rather than splatted: a tool parameter is
            # named `name` (get_entity) and would collide with step()'s own.
            with step(
                trace_steps.TOOL_CALL,
                label=fn.__name__,
                tool=fn.__name__,
                args=dict(bound.arguments),
            ) as s:
                before = len(_chunk_ids())
                out = fn(*args, **kwargs)
                after = _chunk_ids()
                s.output(
                    text=out,
                    chars=len(out),
                    # What this call added to the citation pool, which is the
                    # tool's real effect and is not visible in its return text.
                    chunks_added=len(after) - before,
                )
                s.meta(k=_k(), graph_hops=_hops())
                return out

        call.__signature__ = inspect.signature(fn)
        return call

    return [
        StructuredTool.from_function(traced(hybrid_search)),
        StructuredTool.from_function(traced(vector_search)),
        StructuredTool.from_function(traced(graph_neighbors)),
        StructuredTool.from_function(traced(expand_subgraph)),
        StructuredTool.from_function(traced(get_entity)),
        StructuredTool.from_function(traced(fulltext_search)),
        StructuredTool.from_function(traced(compare)),
        StructuredTool.from_function(traced(read_around)),
        StructuredTool.from_function(traced(global_search)),
    ]
