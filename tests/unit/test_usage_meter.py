"""Per-run token metering.

Input is the larger half of a RAG bill, and it has two ways to go wrong that
both look fine at runtime: counting it from the checkpointed message list
re-charges every earlier turn of a conversation, and a streamed call that
reports no usage block counts nothing at all.
"""

from __future__ import annotations

from types import SimpleNamespace

from graphrag.usage import TokenMeter


def _gen(text: str = "", usage: dict | None = None, info: dict | None = None):
    message = SimpleNamespace(content=text, usage_metadata=usage)
    return SimpleNamespace(text=text, message=message, generation_info=info)


def _response(*generations, llm_output: dict | None = None):
    return SimpleNamespace(generations=[list(generations)], llm_output=llm_output)


def _msg(text: str):
    return SimpleNamespace(content=text)


# ── provider-reported usage ─────────────────────────────────────────────────
async def test_reported_usage_is_used_verbatim():
    meter = TokenMeter()
    await meter.on_chat_model_start({}, [[_msg("hello")]])
    await meter.on_llm_end(
        _response(_gen("hi", {"input_tokens": 1500, "output_tokens": 200}))
    )
    assert meter.totals == (1500, 200)
    assert meter.reported is True


async def test_usage_is_read_from_llm_output_when_the_message_lacks_it():
    """Some OpenAI-compatible endpoints only fill the top-level token_usage."""
    meter = TokenMeter()
    await meter.on_chat_model_start({}, [[_msg("hello")]])
    await meter.on_llm_end(
        _response(_gen("hi"), llm_output={"token_usage": {
            "prompt_tokens": 900, "completion_tokens": 40,
        }})
    )
    assert meter.totals == (900, 40)


async def test_usage_is_read_from_generation_info():
    meter = TokenMeter()
    await meter.on_chat_model_start({}, [[_msg("x")]])
    await meter.on_llm_end(
        _response(_gen("hi", info={"token_usage": {
            "prompt_tokens": 700, "completion_tokens": 30,
        }}))
    )
    assert meter.totals == (700, 30)


async def test_every_turn_of_the_tool_loop_accumulates():
    """The agent re-sends the whole prompt on each turn; that is most of the
    prompt cost and none of it shows up in the visible answer."""
    meter = TokenMeter()
    for _ in range(3):
        await meter.on_chat_model_start({}, [[_msg("context " * 100)]])
        await meter.on_llm_end(
            _response(_gen("partial", {"input_tokens": 1200, "output_tokens": 50}))
        )
    assert meter.totals == (3600, 150)
    assert meter.calls == 3


# ── the estimate floor ──────────────────────────────────────────────────────
async def test_a_call_with_no_usage_block_still_costs_something():
    """A streamed call usually reports no usage unless stream_options asked for
    it — and that flag isn't accepted everywhere, so it isn't sent. Unmetered
    is the one outcome worth ruling out."""
    meter = TokenMeter()
    await meter.on_chat_model_start({}, [[_msg("a" * 4000)]])
    await meter.on_llm_end(_response(_gen("some answer text")))
    tokens_in, tokens_out = meter.totals
    assert tokens_in >= 900       # ~4000 chars / 4
    assert tokens_out >= 1
    assert meter.reported is False


async def test_mixed_runs_do_not_add_the_estimate_on_top_of_real_usage():
    """Taking the max, not the sum: the estimate covers calls the provider may
    already have counted, so adding both would bill the same turn twice."""
    meter = TokenMeter()
    await meter.on_chat_model_start({}, [[_msg("a" * 400)]])     # est ~100 in
    await meter.on_llm_end(
        _response(_gen("x", {"input_tokens": 5000, "output_tokens": 100}))
    )
    tokens_in, _ = meter.totals
    assert tokens_in == 5000


async def test_completion_style_prompts_are_counted_too():
    meter = TokenMeter()
    await meter.on_llm_start({}, ["b" * 800])
    await meter.on_llm_end(_response(_gen("done")))
    assert meter.totals[0] >= 190


async def test_multimodal_parts_do_not_crash_the_estimate():
    """An image part has no text; its cost is left to the reported number
    rather than guessed at, but it must not raise."""
    meter = TokenMeter()
    image_msg = SimpleNamespace(content=[
        {"type": "text", "text": "read this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ])
    await meter.on_chat_model_start({}, [[image_msg]])
    await meter.on_llm_end(_response(_gen("transcript")))
    assert meter.totals[0] >= 1


async def test_a_meter_that_never_saw_a_call_reports_zero():
    assert TokenMeter().totals == (0, 0)


# ── scoping ─────────────────────────────────────────────────────────────────
def test_meter_is_wired_as_a_run_callback_not_read_off_final_state():
    """LangGraph checkpoints the thread, so the final message list holds the
    whole conversation. Summing usage across it would re-charge every previous
    answer, growing the bill quadratically over a long chat.
    """
    import inspect

    from graphrag.agent import graph

    src = inspect.getsource(graph.AgentRunner.session)
    assert 'config["callbacks"] = [meter]' in src


async def test_a_fresh_meter_per_run_does_not_inherit_the_previous_one():
    first = TokenMeter()
    await first.on_llm_end(_response(_gen("a", {"input_tokens": 100, "output_tokens": 10})))
    second = TokenMeter()
    assert second.totals == (0, 0) and first.totals == (100, 10)
