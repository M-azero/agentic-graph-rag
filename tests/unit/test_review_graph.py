"""The review loop: routing, bounds, and the gate that makes it affordable.

Two properties carry the design and are asserted directly rather than inferred:

- **The critic is skipped on clean, cheap answers.** That gate is the whole
  affordability argument — if it stops firing, every question pays for a model
  call it did not need.
- **The loop terminates.** `rounds` bounds research and `revise` has no path
  back to `check`, so the worst case is fixed rather than hoped for.
"""

import pytest

from graphrag.agent.prompts import CLOSED_DOMAIN_REFUSAL
from graphrag.agent.review.citations import verify_citations
from graphrag.agent.review.critic import needs_critic, parse_verdict
from graphrag.agent.review.graph import ReviewRunner
from graphrag.agent.review.state import (
    ESCALATED,
    FLAGGED,
    OK,
    REFUSED,
    RETRIEVE_MORE,
    REVISE,
    REVISED,
    SHIP,
)
from graphrag.core.types import QueryResult, RetrievedChunk

# `asyncio_mode = "auto"` in pyproject.toml runs the async tests here; a module
# level asyncio marker would also be applied to the sync ones and warn.


def _chunk(cid, source="a.pdf"):
    return RetrievedChunk(
        chunk_id=cid, text=f"body {cid}", source=source, score=1.0, retriever="test"
    )


class _Session:
    def __init__(self, runner, answer, sources, tool_calls):
        self._runner, self._answer = runner, answer
        self._sources, self._tool_calls = sources, tool_calls

    async def arun(self):
        for c in self._sources:
            self._runner.sink.add([c])
        return QueryResult(
            answer=self._answer, sources=list(self._sources),
            tool_calls=list(self._tool_calls),
        )


class _Agents:
    """Stands in for AgentRunner. Returns a scripted draft per research pass."""

    def __init__(self, drafts, sources=None, tool_calls=None):
        self._drafts = list(drafts)
        self._sources = sources or [[_chunk("c1")]]
        self._tool_calls = tool_calls or [[{"tool": "hybrid_search", "args": {}}]]
        self.calls: list[dict] = []
        self.sink = None

    def session(self, question, **kw):
        i = min(len(self.calls), len(self._drafts) - 1)
        self.calls.append({"question": question, **kw})
        self.sink = kw["sink"]
        return _Session(
            self, self._drafts[i],
            self._sources[min(i, len(self._sources) - 1)],
            self._tool_calls[min(i, len(self._tool_calls) - 1)],
        )


class _LLM:
    """Scripted critic/reviser. Counts calls so the gate can be asserted."""

    def __init__(self, *replies):
        self._replies = list(replies)
        self.calls = 0

    async def ainvoke(self, prompt, config=None):
        self.calls += 1
        reply = self._replies[min(self.calls - 1, len(self._replies) - 1)]
        if isinstance(reply, Exception):
            raise reply

        class _R:
            content = reply

        return _R()


class _Store:
    def __init__(self, window=None):
        self.window = window or []
        self.calls = 0

    def chunk_window(self, ids, before=1, after=1):
        self.calls += 1
        return self.window


def _runner(agents, llm, store=None, **kw):
    return ReviewRunner(agents, llm, store or _Store(), **kw)


# ── the gate ───────────────────────────────────────────────────────────────


async def test_a_clean_cheap_answer_costs_no_critic_call():
    """The affordability argument, asserted directly."""
    llm = _LLM('{"action":"ship"}')
    agents = _Agents(["Short answer. [source: a.pdf]"])
    out = await _runner(agents, llm).arun("q")

    assert llm.calls == 0
    assert out.critic_called is False
    assert out.outcome == OK


async def test_a_long_answer_gets_reviewed():
    llm = _LLM('{"action":"ship","complete":true}')
    agents = _Agents(["word " * 400 + "[source: a.pdf]"])
    out = await _runner(agents, llm).arun("q")

    assert llm.calls == 1
    assert out.critic_called is True


async def test_a_fabricated_citation_always_gets_reviewed():
    llm = _LLM('{"action":"revise"}', "Fixed. [source: a.pdf]")
    agents = _Agents(["Claim. [source: ghost.pdf]"])
    out = await _runner(agents, llm).arun("q")

    assert llm.calls == 2  # critic, then reviser
    assert out.outcome == REVISED


@pytest.mark.parametrize(
    ("report_kw", "tool_calls", "draft", "expected"),
    [
        ({}, [{}], "short", False),
        ({}, [{}, {}], "short", True),          # more than one tool call
        ({}, [{}], "x" * 2000, True),           # long draft
    ],
)
async def test_needs_critic_gate(report_kw, tool_calls, draft, expected):
    report = verify_citations("Answer. [source: a.pdf]", {"a.pdf"})
    assert needs_critic(report, tool_calls, draft) is expected


# ── refusal ────────────────────────────────────────────────────────────────


async def test_a_refusal_ships_immediately_with_no_sources():
    llm = _LLM('{"action":"revise"}')
    agents = _Agents([CLOSED_DOMAIN_REFUSAL])
    out = await _runner(agents, llm).arun("q")

    assert out.outcome == REFUSED
    assert out.sources == []
    assert llm.calls == 0  # a correct refusal is a good answer


# ── routing and bounds ─────────────────────────────────────────────────────


async def test_retrieve_more_runs_a_second_research_pass():
    llm = _LLM(
        '{"action":"retrieve_more","missing":"the 2024 figures"}',
        '{"action":"ship"}',
    )
    agents = _Agents(["word " * 400 + "[source: a.pdf]"])
    store = _Store(window=[_chunk("c2")])
    out = await _runner(agents, llm, store, max_rounds=1).arun("q")

    assert out.rounds == 2
    assert store.calls == 1  # the cheap window walk happened first
    assert "the 2024 figures" in agents.calls[1]["question"]


async def test_the_second_pass_stays_out_of_conversation_memory():
    """Otherwise round one's whole ReAct transcript is appended to the user's
    history and resent on every later turn."""
    llm = _LLM('{"action":"retrieve_more","missing":"x"}', '{"action":"ship"}')
    agents = _Agents(["word " * 400 + "[source: a.pdf]"])
    await _runner(agents, llm, max_rounds=1).arun("q")

    assert agents.calls[0]["remember"] is True
    assert agents.calls[1]["remember"] is False


async def test_both_passes_share_one_sink():
    """The shipped answer must be able to cite evidence the first pass found."""
    llm = _LLM('{"action":"retrieve_more","missing":"x"}', '{"action":"ship"}')
    agents = _Agents(
        ["word " * 400 + "[source: a.pdf]"],
        sources=[[_chunk("c1", "a.pdf")], [_chunk("c2", "b.pdf")]],
    )
    await _runner(agents, llm, max_rounds=1).arun("q")

    assert agents.calls[0]["sink"] is agents.calls[1]["sink"]


async def test_the_second_pass_widens_the_plan():
    llm = _LLM('{"action":"retrieve_more","missing":"x"}', '{"action":"ship"}')
    agents = _Agents(["word " * 400 + "[source: a.pdf]"])
    await _runner(agents, llm, max_rounds=1).arun("q")

    assert agents.calls[1]["plan"].candidate_k > agents.calls[0]["plan"].candidate_k


async def test_research_is_bounded_by_max_rounds():
    """A critic that always escalates must not loop forever."""
    llm = _LLM('{"action":"retrieve_more","missing":"more"}')
    agents = _Agents(["word " * 400 + "[source: a.pdf]"])
    out = await _runner(agents, llm, max_rounds=1).arun("q")

    assert out.rounds == 2  # the initial pass plus exactly one escalation
    assert out.outcome == ESCALATED


async def test_max_rounds_zero_never_escalates():
    llm = _LLM('{"action":"retrieve_more","missing":"more"}')
    agents = _Agents(["word " * 400 + "[source: a.pdf]"])
    out = await _runner(agents, llm, max_rounds=0).arun("q")

    assert out.rounds == 1


async def test_revise_does_not_loop_back_into_check():
    """One repair attempt. A revise → check → revise cycle would be unbounded
    exactly when it triggers most: a model that cannot stop inventing sources."""
    llm = _LLM('{"action":"revise"}', "Still bad. [source: ghost.pdf]")
    agents = _Agents(["Claim. [source: ghost.pdf]"])
    out = await _runner(agents, llm).arun("q")

    assert llm.calls == 2  # critic + one revise, never a second critic
    # The tag survived the repair, so it is stripped *and* flagged — dropping
    # it silently would hide the fabrication instead of surfacing it.
    assert out.outcome == FLAGGED
    assert "ghost.pdf" not in out.answer


# ── failing open ───────────────────────────────────────────────────────────


async def test_a_dead_critic_ships_the_draft():
    llm = _LLM(RuntimeError("provider down"))
    agents = _Agents(["word " * 400 + "[source: a.pdf]"])
    out = await _runner(agents, llm).arun("q")

    assert out.answer.startswith("word")
    assert out.outcome == OK


async def test_unparseable_critic_output_ships_the_draft():
    llm = _LLM("I think the answer looks fine, honestly.")
    agents = _Agents(["word " * 400 + "[source: a.pdf]"])
    out = await _runner(agents, llm).arun("q")

    assert out.outcome == OK


async def test_a_failed_revise_keeps_the_original_answer():
    """The draft is a real answer that was already paid for."""
    llm = _LLM('{"action":"revise"}', RuntimeError("provider down"))
    agents = _Agents(["Claim. [source: ghost.pdf]"])
    out = await _runner(agents, llm).arun("q")

    assert out.answer == "Claim. [source: ghost.pdf]"


async def test_a_broken_chunk_window_does_not_fail_the_answer():
    class _Broken:
        def chunk_window(self, *a, **k):
            raise RuntimeError("neo4j down")

    llm = _LLM('{"action":"retrieve_more","missing":"x"}', '{"action":"ship"}')
    agents = _Agents(["word " * 400 + "[source: a.pdf]"])
    out = await _runner(agents, llm, _Broken(), max_rounds=1).arun("q")
    assert out.answer


# ── verdict parsing ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "action"),
    [
        ('{"action":"ship"}', SHIP),
        ('{"action":"revise"}', REVISE),
        ('{"action":"retrieve_more"}', RETRIEVE_MORE),
        ('here you go:\n```json\n{"action":"revise"}\n```', REVISE),
        ('prose {"action":"revise"} more prose', REVISE),
        ('{"action":"SHIP"}', SHIP),
        ('{"action":"explode"}', SHIP),      # unknown action fails open
        ("not json at all", SHIP),
        ("", SHIP),
        ('{"action": "revise", "note": "a } brace in a string"}', REVISE),
    ],
)
def test_verdict_parsing(raw, action):
    assert parse_verdict(raw).action == action


def test_verdict_fields_are_bounded():
    v = parse_verdict(
        '{"action":"revise","missing":"' + "x" * 900 + '",'
        '"unsupported":["a","b","c","d","e","f","g","h","i","j","k","l"]}'
    )
    assert len(v.missing) <= 500
    assert len(v.unsupported) <= 10


def test_a_string_unsupported_field_is_accepted():
    """Models return a bare string where a list was asked for."""
    assert parse_verdict('{"action":"revise","unsupported":"one claim"}').unsupported == (
        "one claim",
    )
