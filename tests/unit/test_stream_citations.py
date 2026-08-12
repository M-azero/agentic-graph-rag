"""Citation review on the streamed path.

Streaming cannot repair an answer — the tokens are already on the wire by the
time anything can look at them. But `sources` has *not* been sent yet, and that
is the part this can still get right: a streamed refusal must not ship the
chunks the model just rejected, which is the same bug that was fixed on the
non-streaming path.

No critic call here on purpose: ~1.6k tokens and a second of latency to produce
an advisory nobody can act on is a bad trade.
"""

import json

import pytest

from graphrag.agent.prompts import CLOSED_DOMAIN_REFUSAL
from graphrag.agent.tools import GRAPH_LABEL
from graphrag.api.streaming import sse_answer
from graphrag.core.types import RetrievedChunk


def _chunk(cid, source):
    return RetrievedChunk(
        chunk_id=cid, text=f"body {cid}", source=source, score=1.0, retriever="test"
    )


class _Service:
    """Stands in for QueryService, replaying a scripted stream."""

    def __init__(self, answer, sources, labels=()):
        self._answer, self._sources, self._labels = answer, sources, list(labels)

    async def stream(self, question, **kw):
        yield "tool", "hybrid_search", self._sources, self._labels
        for piece in self._answer.split(" "):
            yield "token", piece + " ", self._sources, self._labels


async def _events(service):
    out = []
    async for ev in sse_answer(service, "q", "concise", "t"):
        out.append(ev)
    return out


def _sources_event(events):
    for ev in events:
        if ev["event"] == "sources":
            return json.loads(ev["data"])
    raise AssertionError("no sources event")


def _event_names(events):
    return [e["event"] for e in events]


# ── the refusal fix, on the streaming path ─────────────────────────────────


async def test_a_streamed_refusal_ships_no_sources():
    service = _Service(CLOSED_DOMAIN_REFUSAL, [_chunk("c1", "a.pdf")])
    assert _sources_event(await _events(service)) == []


async def test_a_normal_streamed_answer_keeps_its_sources():
    service = _Service("The answer. [source: a.pdf]", [_chunk("c1", "a.pdf")])
    payload = _sources_event(await _events(service))
    assert [s["source"] for s in payload] == ["a.pdf"]


# ── cited marking and ordering ─────────────────────────────────────────────


async def test_cited_sources_are_marked_and_sorted_first():
    service = _Service(
        "Per the memo. [source: b.pdf]",
        [_chunk("c1", "a.pdf"), _chunk("c2", "b.pdf"), _chunk("c3", "c.pdf")],
    )
    payload = _sources_event(await _events(service))
    assert payload[0]["source"] == "b.pdf"
    assert payload[0]["cited"] is True
    assert all(s["cited"] is False for s in payload[1:])


async def test_graph_labels_reach_the_stream_check():
    """Without labels on the stream, a graph-grounded answer would be flagged
    as citing something that was never retrieved."""
    service = _Service(
        f"They are linked. [source: {GRAPH_LABEL}]",
        [_chunk("c1", "a.pdf")],
        labels=(GRAPH_LABEL,),
    )
    events = await _events(service)
    assert "verification" not in _event_names(events)


# ── advisory flagging ──────────────────────────────────────────────────────


async def test_a_fabricated_citation_is_flagged_not_blocked():
    """The text already reached the client, so the client is told rather than
    protected."""
    service = _Service("Claim. [source: ghost.pdf]", [_chunk("c1", "a.pdf")])
    events = await _events(service)
    names = _event_names(events)

    assert "verification" in names
    verification = next(e for e in events if e["event"] == "verification")
    assert json.loads(verification["data"])["reasons"] == ["ghost.pdf"]
    # The answer still went out in full.
    assert "".join(e["data"] for e in events if e["event"] == "token").strip() == (
        "Claim. [source: ghost.pdf]"
    )


async def test_a_clean_answer_emits_no_verification_event():
    service = _Service("Fine. [source: a.pdf]", [_chunk("c1", "a.pdf")])
    assert "verification" not in _event_names(await _events(service))


# ── event ordering and shape ───────────────────────────────────────────────


async def test_verification_lands_after_sources_and_before_done():
    service = _Service("Claim. [source: ghost.pdf]", [_chunk("c1", "a.pdf")])
    names = _event_names(await _events(service))
    assert names.index("sources") < names.index("verification") < names.index("done")


async def test_an_empty_stream_does_not_blow_up():
    """`labels` must be initialised — a stream that yields nothing would
    otherwise reference it unbound."""

    class _Empty:
        async def stream(self, question, **kw):
            return
            yield  # pragma: no cover - makes this an async generator

    assert _sources_event(await _events(_Empty())) == []


@pytest.mark.parametrize("event", ["token", "sources", "done"])
async def test_the_event_contract_is_unchanged(event):
    service = _Service("Fine. [source: a.pdf]", [_chunk("c1", "a.pdf")])
    assert event in _event_names(await _events(service))
