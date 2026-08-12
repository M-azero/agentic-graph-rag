"""The review loop: research → check → {ship | revise | retrieve more}.

A real `StateGraph`, because `create_react_agent` cannot express it — the loop
needs nodes that are not the model deciding what to do next, and a bounded edge
back into research.

Why the shape is this and not a conversation between agents: every node that
costs tokens sees the smallest input that lets it do its job. The critic sees a
draft and a list of source *names*; the reviser sees the same plus what was
wrong. Neither ever sees a raw chunk. A group chat would hand every participant
the whole transcript, which is how multi-agent systems become expensive without
becoming more accurate.

Termination is structural rather than hoped for:
  - `rounds` only increments in `research`, and `check` forces a ship once it
    reaches `max_rounds`;
  - `revise` has exactly one outgoing edge, to `finalize`.
So the worst case is fixed: two research passes, two critic calls, one revise.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from graphrag.agent.review.citations import (
    available_sources,
    order_by_citation,
    verify_citations,
)
from graphrag.agent.review.critic import needs_critic, run_critic
from graphrag.agent.review.revise import revise as run_revise
from graphrag.agent.review.state import (
    ESCALATED,
    FLAGGED,
    OK,
    REFUSED,
    RETRIEVE_MORE,
    REVISE,
    REVISED,
    SHIP,
    ReviewOutcome,
    ReviewState,
)
from graphrag.agent.tools import SourceSink
from graphrag.core.logging import get_logger
from graphrag.core.types import RetrievedChunk
from graphrag.retrieval.plan import RetrievalPlan

log = get_logger(__name__)


class ReviewRunner:
    """Compiles and drives the review graph for one tenant.

    Holds no per-query state: like `AgentRunner`, the graph is compiled once and
    everything about a single question travels in the state dict.
    """

    def __init__(
        self,
        agent_runner,
        llm,
        graph_store,
        *,
        max_rounds: int = 1,
        window_before: int = 1,
        window_after: int = 1,
        critic_free_tool_calls: int = 1,
        critic_free_chars: int = 1200,
        base_plan: RetrievalPlan | None = None,
    ) -> None:
        self._agents = agent_runner
        self._llm = llm
        self._store = graph_store
        self._max_rounds = max(0, int(max_rounds))
        self._window = (max(0, window_before), max(0, window_after))
        self._free_tool_calls = critic_free_tool_calls
        self._free_chars = critic_free_chars
        self._base_plan = base_plan or RetrievalPlan()
        self._compiled = self._build()

    # -- nodes ---------------------------------------------------------------

    async def _research(self, state: ReviewState) -> dict[str, Any]:
        """Run the ReAct agent and collect its draft plus its evidence."""
        rounds = state.get("rounds", 0)
        escalating = rounds > 0

        question = state["question"]
        if escalating and (verdict := state.get("verdict")) and verdict.missing:
            # Ask for the gap, not the original question again — repeating it
            # verbatim tends to reproduce the same retrieval and the same
            # answer, at full price.
            question = f"{question}\n\nFocus specifically on: {verdict.missing}"

        session = self._agents.session(
            question,
            style=state.get("style"),
            preset=state.get("preset"),
            thread_id=state["thread_id"],
            model=state.get("model"),
            meter=state.get("meter"),
            plan=state.get("plan") or self._base_plan,
            # One sink across both passes, so the answer can cite evidence the
            # first pass found even after the second pass widened the net.
            sink=state["sink"],
            # Round two stays out of conversation memory — see `_agent_for`.
            remember=not escalating,
        )
        result = await session.arun()

        # The sink accumulates, so it — not this result — is the whole picture.
        sink = state["sink"]
        merged = list(sink.chunks) + [
            c for c in state.get("window_gain", []) if c not in sink.chunks
        ]
        return {
            "draft": result.answer,
            "tool_calls": (state.get("tool_calls") or []) + result.tool_calls,
            "sources": merged,
            "labels": sorted(sink.labels),
            "rounds": rounds + 1,
        }

    async def _check(self, state: ReviewState) -> dict[str, Any]:
        """Free citation check first; a model call only if it settles nothing."""
        draft = state.get("draft", "")
        available = available_sources(state.get("sources", []), state.get("labels", []))
        report = verify_citations(draft, available)

        if report.refusal:
            return {"report": report, "action": SHIP, "outcome": REFUSED}

        out: dict[str, Any] = {"report": report}
        if not needs_critic(
            report, state.get("tool_calls", []), draft,
            free_tool_calls=self._free_tool_calls, free_chars=self._free_chars,
        ):
            out["action"] = SHIP
            return out

        verdict = await run_critic(
            self._llm, state["question"], draft, available, report,
            config=state.get("llm_config"),
        )
        out["verdict"] = verdict
        action = verdict.action

        # Structural bound: once the research budget is spent, escalation is no
        # longer on the table whatever the critic thinks.
        if action == RETRIEVE_MORE and not self._can_escalate(state):
            action = REVISE if not report.clean or not verdict.complete else SHIP
            out["outcome"] = ESCALATED
        out["action"] = action
        return out

    def _widen(self, state: ReviewState) -> dict[str, Any]:
        """Cheap escalation first: walk the :NEXT chain from what we already
        have. A boundary that cut an answer in half costs one graph hop to
        repair, against a full retrieval pass with an embedding call and a
        rerank."""
        sources = state.get("sources", [])
        ids = [c.chunk_id for c in sources[:8] if c.chunk_id]
        gained: list[RetrievedChunk] = []
        if ids and any(self._window):
            try:
                gained = self._store.chunk_window(
                    ids, before=self._window[0], after=self._window[1]
                )
            except Exception as exc:  # a missing window must not fail the answer
                log.warning("chunk_window_failed", error=str(exc))
        known = {c.chunk_id for c in sources}
        fresh = [c for c in gained if c.chunk_id not in known]
        return {
            "plan": (state.get("plan") or self._base_plan).widened(),
            "window_gain": fresh,
        }

    async def _revise(self, state: ReviewState) -> dict[str, Any]:
        available = available_sources(state.get("sources", []), state.get("labels", []))
        answer, outcome = await run_revise(
            self._llm, state.get("draft", ""), available,
            state["report"], state.get("verdict"),
            config=state.get("llm_config"),
        )
        return {"answer": answer, "outcome": outcome}

    def _finalize(self, state: ReviewState) -> dict[str, Any]:
        answer = state.get("answer") or state.get("draft", "")
        outcome = state.get("outcome") or OK
        report = state.get("report")

        if outcome == REFUSED:
            sources: list[RetrievedChunk] = []
        elif report is not None:
            # Re-check when the text changed under us, so `cited` describes the
            # answer actually being sent.
            final = (
                report
                if outcome in (OK, REFUSED)
                else verify_citations(
                    answer,
                    available_sources(state.get("sources", []), state.get("labels", [])),
                )
            )
            sources = order_by_citation(state.get("sources", []), final)
            report = final
        else:
            sources = state.get("sources", [])

        verdict = state.get("verdict")
        log.info(
            "review_decision",
            rounds=state.get("rounds", 0),
            action=state.get("action", SHIP),
            outcome=outcome,
            critic_called=verdict is not None,
            cited=len(report.cited) if report else 0,
            fabricated=len(report.fabricated) if report else 0,
        )
        return {"answer": answer, "outcome": outcome, "sources": sources, "report": report}

    # -- wiring --------------------------------------------------------------

    def _can_escalate(self, state: ReviewState) -> bool:
        """`rounds` counts completed research passes; `max_rounds` is how many
        *extra* ones are allowed. So `max_rounds=0` means one pass and no
        escalation, and `max_rounds=1` allows a single retry.

        One definition, read by both `_check` and `_route` — two copies of this
        comparison drifting apart is how a bounded loop stops being bounded.
        """
        return state.get("rounds", 0) <= self._max_rounds

    def _route(self, state: ReviewState) -> str:
        action = state.get("action", SHIP)
        if action == RETRIEVE_MORE and self._can_escalate(state):
            return "widen"
        if action == REVISE:
            return "revise"
        return "finalize"

    def _build(self):
        g = StateGraph(ReviewState)
        g.add_node("research", self._research)
        g.add_node("check", self._check)
        g.add_node("widen", self._widen)
        g.add_node("revise", self._revise)
        g.add_node("finalize", self._finalize)

        g.add_edge(START, "research")
        g.add_edge("research", "check")
        g.add_conditional_edges(
            "check", self._route,
            {"widen": "widen", "revise": "revise", "finalize": "finalize"},
        )
        g.add_edge("widen", "research")   # the only cycle, bounded by `rounds`
        g.add_edge("revise", "finalize")  # deliberately no path back to check
        g.add_edge("finalize", END)
        return g.compile()

    async def arun(
        self,
        question: str,
        *,
        style: str | None = None,
        preset: str | None = None,
        thread_id: str = "default",
        user_id: str | None = None,
        model: Any = None,
        meter: Any = None,
        plan: RetrievalPlan | None = None,
    ) -> ReviewOutcome:
        """Answer one question with review. Async only — the API's checkpointer
        is the async saver, which refuses a sync invoke."""
        state: ReviewState = {
            "question": question,
            "style": style,
            "preset": preset,
            "thread_id": thread_id,
            "user_id": user_id,
            "model": model,
            "meter": meter,
            # One meter for the whole loop, so the critic's and reviser's tokens
            # land on the same bill as the answer they were spent reviewing.
            "llm_config": {"callbacks": [meter]} if meter is not None else {},
            "plan": plan or self._base_plan,
            "sink": SourceSink(),
            "rounds": 0,
        }
        final = await self._compiled.ainvoke(
            state, {"recursion_limit": self._recursion_limit()}
        )
        return self.outcome_from(final)

    def _recursion_limit(self) -> int:
        """Supersteps the loop can take. Derived from the same bound the routing
        enforces, so a routing bug surfaces as a clear recursion error rather
        than an unbounded spend."""
        # research + check per round, then widen between rounds, then
        # revise + finalize, with slack.
        return 3 * (self._max_rounds + 1) + 4

    def outcome_from(self, state: ReviewState) -> ReviewOutcome:
        report = state.get("report")
        return ReviewOutcome(
            answer=state.get("answer", ""),
            sources=state.get("sources", []),
            tool_calls=state.get("tool_calls", []),
            labels=state.get("labels", []),
            outcome=state.get("outcome", OK),
            rounds=state.get("rounds", 0),
            critic_called=state.get("verdict") is not None,
            report=report,
            verdict=state.get("verdict"),
        )


__all__ = ["ReviewRunner", "ESCALATED", "FLAGGED", "OK", "REFUSED", "REVISED"]
