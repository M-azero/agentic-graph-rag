"""Every model call the review loop makes is billed to the request.

The trap this guards: `TokenMeter` is attached per invocation via
`config["callbacks"]`. A critic or reviser call made outside that config spends
real provider tokens that never reach metering or quota — free to the user, and
invisible in `usage_events`. That is the same class of bug `record_answer_tokens`
was written to close on the streaming path, so it gets the same kind of test.
"""

from graphrag.agent.review.citations import verify_citations
from graphrag.agent.review.critic import run_critic
from graphrag.agent.review.revise import revise


class _Recorder:
    """A chat model that records the config each call was made with."""

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.configs: list[dict] = []

    async def ainvoke(self, prompt, config=None):
        self.configs.append(config)

        class _R:
            content = self._reply

        return _R()


class _Meter:
    """Stands in for TokenMeter — identity is all these assertions need."""


async def test_the_critic_is_billed_to_the_request():
    llm = _Recorder('{"action":"ship"}')
    meter = _Meter()
    report = verify_citations("Answer. [source: a.pdf]", {"a.pdf"})

    await run_critic(
        llm, "q", "draft", {"a.pdf"}, report, config={"callbacks": [meter]}
    )

    assert llm.configs == [{"callbacks": [meter]}]


async def test_the_reviser_is_billed_to_the_request():
    llm = _Recorder("Fixed. [source: a.pdf]")
    meter = _Meter()
    report = verify_citations("Claim. [source: ghost.pdf]", {"a.pdf"})

    await revise(llm, "Claim. [source: ghost.pdf]", {"a.pdf"}, report,
                 config={"callbacks": [meter]})

    assert llm.configs == [{"callbacks": [meter]}]


async def test_the_whole_loop_shares_one_meter():
    """Critic and reviser must carry the *same* meter instance as the answer
    they reviewed — two meters would split one question across two bills."""
    from graphrag.agent.review.graph import ReviewRunner
    from graphrag.core.types import QueryResult

    seen: list[dict] = []

    class _LLM:
        async def ainvoke(self, prompt, config=None):
            seen.append(config)

            class _R:
                content = '{"action":"revise"}' if len(seen) == 1 else "Fixed. [source: a.pdf]"

            return _R()

    class _Session:
        async def arun(self):
            return QueryResult(answer="Claim. [source: ghost.pdf]", tool_calls=[{}])

    class _Agents:
        def __init__(self):
            self.meters: list = []

        def session(self, question, **kw):
            self.meters.append(kw.get("meter"))
            return _Session()

    class _Store:
        def chunk_window(self, *a, **k):
            return []

    agents, meter = _Agents(), _Meter()
    await ReviewRunner(agents, _LLM(), _Store()).arun("q", meter=meter)

    assert agents.meters == [meter]                 # research
    assert [c["callbacks"] for c in seen] == [[meter], [meter]]  # critic + revise


async def test_no_meter_still_works():
    """The CLI passes no meter; the loop must not require one."""
    from graphrag.agent.review.graph import ReviewRunner
    from graphrag.core.types import QueryResult

    class _LLM:
        async def ainvoke(self, prompt, config=None):
            class _R:
                content = '{"action":"ship"}'

            return _R()

    class _Session:
        async def arun(self):
            return QueryResult(answer="Answer. [source: a.pdf]", tool_calls=[{}])

    class _Agents:
        def session(self, question, **kw):
            return _Session()

    class _Store:
        def chunk_window(self, *a, **k):
            return []

    out = await ReviewRunner(_Agents(), _LLM(), _Store()).arun("q")
    assert out.answer == "Answer. [source: a.pdf]"
