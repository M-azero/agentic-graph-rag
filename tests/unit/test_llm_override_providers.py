"""`GRAPHRAG_LLM` accepts exactly what the factory can build.

The loader kept its own hand-written set of provider names and it drifted:
`deepinfra` and `cohere` were missing, which between them are three of the four
links in the production profile's chain. So the documented one-line model
toggle — `GRAPHRAG_LLM=deepinfra:openai/gpt-oss-120b` — raised a ConfigError
during `load_settings()`, i.e. the container refused to boot, for a provider
`build_chat_model` supports fine.

Deriving one from the other is the fix; this test is what stops a second copy
appearing.
"""

from __future__ import annotations

import pytest

from graphrag.config.loader import _apply_llm_override, _llm_providers
from graphrag.core.errors import ConfigError
from graphrag.llm.factory import chat_providers


def test_the_loader_accepts_exactly_what_the_factory_builds():
    assert _llm_providers() == chat_providers()


@pytest.mark.parametrize(
    "override",
    [
        "deepinfra:openai/gpt-oss-120b",   # production primary fallback
        "cohere:command-a-03-2025",        # production third vendor
        "deepseek:deepseek-v4-flash",
        "ollama:gemma3:4b",                # a tag containing a colon
        "anthropic:claude-opus-5",
        "openai:gpt-5",
        "gemini:gemini-3.5-flash",
        "qwen:qwen3.6-plus",
    ],
)
def test_every_supported_provider_is_accepted(override):
    merged: dict = {}
    _apply_llm_override(merged, override)
    provider, _, model = override.partition(":")
    assert merged["llm"] == {"provider": provider, "model": model, "extra": {}}


def test_an_ollama_tag_keeps_its_colons():
    """Split on the FIRST colon only, or `gemma3:4b` becomes `gemma3`."""
    merged: dict = {}
    _apply_llm_override(merged, "ollama:gemma4:e4b-it-q4_K_M")
    assert merged["llm"]["model"] == "gemma4:e4b-it-q4_K_M"


@pytest.mark.parametrize(
    "override", ["not-a-provider:x", "openai", "openai:", ":gpt-5", ""]
)
def test_malformed_overrides_are_still_refused(override):
    with pytest.raises(ConfigError, match="GRAPHRAG_LLM"):
        _apply_llm_override({}, override)


def test_the_error_names_the_providers_that_would_work():
    with pytest.raises(ConfigError) as caught:
        _apply_llm_override({}, "bedrock:claude")
    message = str(caught.value)
    assert "deepinfra" in message and "cohere" in message


def test_provider_specific_extra_is_dropped_on_a_real_change():
    """`extra` carries model-specific kwargs (Anthropic thinking, Ollama
    num_ctx) that another provider's client rejects."""
    merged = {"llm": {"provider": "anthropic", "model": "claude", "extra": {"thinking": 1}}}
    _apply_llm_override(merged, "deepinfra:openai/gpt-oss-120b")
    assert merged["llm"]["extra"] == {}


def test_extra_survives_a_no_op_override():
    merged = {"llm": {"provider": "anthropic", "model": "claude", "extra": {"thinking": 1}}}
    _apply_llm_override(merged, "anthropic:claude")
    assert merged["llm"]["extra"] == {"thinking": 1}
