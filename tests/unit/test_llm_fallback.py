"""Provider failover.

The engine's whole job is to be correct on the unhappy path, so these are all
failure cases: what fails over, what deliberately doesn't, what happens once
tokens are already on the wire, and whether a dead provider stays out of the
rotation.
"""

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from graphrag.config.settings import EmbeddingCfg, FallbackModel, Secrets
from graphrag.core.errors import ProviderError
from graphrag.llm import build_chat_chain
from graphrag.llm.fallback import FallbackChatModel, Health, classify


class _Stub:
    """A chat-model stand-in: answers, or raises whatever it was given."""

    def __init__(self, reply="ok", error=None, chunks=None):
        self.reply = reply
        self.error = error
        self.chunks = chunks
        self.calls = 0

    def invoke(self, messages, **kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return AIMessage(content=self.reply)

    async def ainvoke(self, messages, **kwargs):
        return self.invoke(messages, **kwargs)

    def stream(self, messages, **kwargs):
        self.calls += 1
        for chunk in self.chunks if self.chunks is not None else [self.reply]:
            if isinstance(chunk, Exception):
                raise chunk
            yield AIMessageChunk(content=chunk)
        if self.error is not None:
            raise self.error

    async def astream(self, messages, **kwargs):
        for chunk in self.stream(messages, **kwargs):
            yield chunk

    def bind_tools(self, tools, **kwargs):
        return self


def _chain(*members, max_failures=2, cooldown=300.0):
    labels = [f"stub{i}" for i in range(len(members))]
    return FallbackChatModel(
        members=list(members),
        labels=labels,
        health=Health(len(members), max_failures, cooldown),
    )


_ASK = [HumanMessage(content="hi")]


# -- classification -----------------------------------------------------------

@pytest.mark.parametrize(
    "message",
    [
        "403 PERMISSION_DENIED: Lightning dunning decision is deny for project",
        "429 Too Many Requests",
        "503 Service Unavailable",
        "Connection refused",
        "Request timed out",
    ],
)
def test_availability_errors_fail_over(message):
    assert classify(RuntimeError(message))[0] is True


@pytest.mark.parametrize(
    "message",
    [
        "This model's maximum context length is 8192 tokens",
        "Response was blocked due to content filter",
    ],
)
def test_request_shaped_errors_do_not_fail_over(message):
    """The next provider would reject the same prompt the same way, so a
    failover only spends a second call to arrive at the same error."""
    assert classify(RuntimeError(message))[0] is False


def test_status_code_beats_message_text():
    exc = RuntimeError("something went wrong")
    exc.status_code = 429
    assert classify(exc) == (True, "rate_limit")


# -- invoke -------------------------------------------------------------------

def test_falls_through_to_the_next_provider():
    dead, alive = _Stub(error=RuntimeError("503 unavailable")), _Stub(reply="second")
    assert _chain(dead, alive).invoke(_ASK).content == "second"
    assert dead.calls == 1 and alive.calls == 1


def test_primary_is_preferred_while_it_works():
    first, second = _Stub(reply="first"), _Stub(reply="second")
    assert _chain(first, second).invoke(_ASK).content == "first"
    assert second.calls == 0


def test_permanent_error_stops_the_chain():
    """A context-length error must surface, not be retried into a second bill."""
    over, spare = _Stub(error=RuntimeError("maximum context length exceeded")), _Stub()
    with pytest.raises(RuntimeError, match="maximum context"):
        _chain(over, spare).invoke(_ASK)
    assert spare.calls == 0


def test_every_provider_down_raises_provider_error():
    a, b = _Stub(error=RuntimeError("503 a")), _Stub(error=RuntimeError("503 b"))
    with pytest.raises(ProviderError, match="Every chat provider"):
        _chain(a, b).invoke(_ASK)


async def test_async_path_fails_over_too():
    dead, alive = _Stub(error=RuntimeError("connection reset")), _Stub(reply="second")
    result = await _chain(dead, alive).ainvoke(_ASK)
    assert result.content == "second"


# -- streaming ----------------------------------------------------------------

def test_stream_fails_over_before_the_first_token():
    dead = _Stub(chunks=[RuntimeError("503 down")])
    alive = _Stub(chunks=["he", "llo"])
    out = "".join(c.content for c in _chain(dead, alive).stream(_ASK))
    assert out == "hello"


def test_stream_that_dies_midway_does_not_replay():
    """The user has already seen "he". Restarting on another provider would
    print the whole answer a second time, so the error has to surface."""
    broken = _Stub(chunks=["he", RuntimeError("503 mid-stream")])
    spare = _Stub(chunks=["completely different"])
    chain = _chain(broken, spare)

    seen = []
    with pytest.raises(RuntimeError, match="mid-stream"):
        for chunk in chain.stream(_ASK):
            seen.append(chunk.content)

    assert seen == ["he"]
    assert spare.calls == 0


async def test_astream_that_dies_midway_does_not_replay():
    broken = _Stub(chunks=["he", RuntimeError("503 mid-stream")])
    spare = _Stub(chunks=["different"])
    seen = []
    with pytest.raises(RuntimeError, match="mid-stream"):
        async for chunk in _chain(broken, spare).astream(_ASK):
            seen.append(chunk.content)
    assert seen == ["he"] and spare.calls == 0


def test_streaming_is_enabled_on_the_wrapper():
    """langchain-core decides a model can stream by checking whether these are
    overridden. If they ever stop being, token streaming silently degrades to
    one big chunk at the end."""
    from langchain_core.language_models.chat_models import BaseChatModel

    assert FallbackChatModel._stream is not BaseChatModel._stream
    assert FallbackChatModel._astream is not BaseChatModel._astream


# -- circuit breaker ----------------------------------------------------------

def test_breaker_takes_a_dead_provider_out_of_rotation():
    dead = _Stub(error=RuntimeError("403 permission_denied"))
    alive = _Stub(reply="ok")
    chain = _chain(dead, alive, max_failures=2)

    for _ in range(5):
        chain.invoke(_ASK)

    # Two probes to trip it, none after: a denied key must not cost a wasted
    # round-trip on every single request.
    assert dead.calls == 2
    assert alive.calls == 5


def test_breaker_reopens_after_the_cooldown():
    dead = _Stub(error=RuntimeError("503"))
    chain = _chain(dead, _Stub(), max_failures=1, cooldown=0.0)
    chain.invoke(_ASK)
    chain.invoke(_ASK)
    assert dead.calls == 2  # cooldown of 0 means it is retried immediately


def test_recovery_clears_the_failure_count():
    flaky = _Stub(error=RuntimeError("503"))
    chain = _chain(flaky, _Stub(reply="backup"), max_failures=2)
    chain.invoke(_ASK)
    flaky.error = None
    chain.invoke(_ASK)
    flaky.error = RuntimeError("503")
    chain.invoke(_ASK)
    assert flaky.calls == 3  # never tripped, because the success reset the count


def test_bind_tools_keeps_the_shared_breaker():
    """The agent rebinds tools on every request. A fresh breaker per bind would
    reset the count before it could ever reach the threshold."""
    dead = _Stub(error=RuntimeError("403 denied"))
    chain = _chain(dead, _Stub(), max_failures=2)
    for _ in range(4):
        chain.bind_tools([]).invoke(_ASK)
    assert dead.calls == 2


# -- chain construction -------------------------------------------------------

def test_chain_without_fallbacks_is_the_bare_model():
    """Nothing pays for the wrapper unless failover is configured."""
    secrets = Secrets(DEEPSEEK_API_KEY="ds")
    assert not isinstance(
        build_chat_chain("deepseek", "deepseek-v4-flash", secrets), FallbackChatModel
    )


def test_fallback_without_a_key_is_dropped():
    """A link with no credentials fails every call; leaving it in the chain
    just buys a guaranteed round-trip before the one that works.

    The empty key is passed explicitly because `Secrets` reads the repo's real
    `.env`, where a developer's DashScope key would otherwise leak in and make
    this pass for the wrong reason.
    """
    secrets = Secrets(DEEPSEEK_API_KEY="ds", DASHSCOPE_API_KEY="")
    chain = build_chat_chain(
        "deepseek", "deepseek-v4-flash", secrets,
        fallbacks=[FallbackModel(provider="qwen", model="qwen3.6-flash")],
    )
    assert not isinstance(chain, FallbackChatModel)


def test_whitespace_key_counts_as_missing():
    secrets = Secrets(DEEPSEEK_API_KEY="ds", DASHSCOPE_API_KEY="   ")
    chain = build_chat_chain(
        "deepseek", "deepseek-v4-flash", secrets,
        fallbacks=[FallbackModel(provider="qwen", model="qwen3.6-flash")],
    )
    assert not isinstance(chain, FallbackChatModel)


def test_configured_chain_is_built_in_order():
    secrets = Secrets(DEEPSEEK_API_KEY="ds", DASHSCOPE_API_KEY="qw", GOOGLE_API_KEY="g")
    chain = build_chat_chain(
        "deepseek", "deepseek-v4-flash", secrets,
        fallbacks=[
            FallbackModel(provider="qwen", model="qwen3.6-flash"),
            FallbackModel(provider="gemini", model="gemini-3.5-flash"),
        ],
    )
    assert isinstance(chain, FallbackChatModel)
    assert chain.labels == [
        "deepseek:deepseek-v4-flash", "qwen:qwen3.6-flash", "gemini:gemini-3.5-flash"
    ]


def test_a_fallback_repeating_the_primary_is_dropped():
    secrets = Secrets(DEEPSEEK_API_KEY="ds")
    chain = build_chat_chain(
        "deepseek", "deepseek-v4-flash", secrets,
        fallbacks=[FallbackModel(provider="deepseek", model="deepseek-v4-flash")],
    )
    assert not isinstance(chain, FallbackChatModel)


# -- embeddings guard ---------------------------------------------------------

def test_embedding_fallback_to_another_model_is_refused():
    """Two embedding models put the same text in different places. A silent
    switch writes vectors that can't be compared with the stored ones, and
    nothing raises — retrieval just quietly stops working."""
    with pytest.raises(ValueError, match="different space|same model"):
        EmbeddingCfg(
            provider="cohere",
            model="embed-v4.0",
            fallbacks=[FallbackModel(provider="voyage", model="voyage-3")],
        )


def test_embedding_fallback_to_the_same_model_is_allowed():
    cfg = EmbeddingCfg(
        provider="cohere",
        model="embed-v4.0",
        fallbacks=[FallbackModel(provider="openai", model="embed-v4.0")],
    )
    assert len(cfg.fallbacks) == 1
