"""Provider wiring in the chat-model factory."""

import pytest

from graphrag.config.settings import Secrets
from graphrag.core.errors import ProviderError
from graphrag.llm import build_chat_model


@pytest.fixture
def secrets() -> Secrets:
    return Secrets(
        DEEPSEEK_API_KEY="ds-test", DASHSCOPE_API_KEY="qw-test", OPENAI_API_KEY="oa-test"
    )


def test_deepseek_uses_its_own_endpoint_and_key(secrets):
    """DeepSeek is OpenAI-compatible, so the risk is silently talking to
    OpenAI with a DeepSeek key (or vice versa)."""
    m = build_chat_model("deepseek", "deepseek-v4-flash", secrets)
    assert "api.deepseek.com" in str(m.openai_api_base)
    assert m.openai_api_key.get_secret_value() == "ds-test"


def test_qwen_uses_dashscope_compatible_mode(secrets):
    m = build_chat_model("qwen", "qwen3.6-plus", secrets)
    assert "dashscope" in str(m.openai_api_base)
    assert "compatible-mode" in str(m.openai_api_base)
    assert m.openai_api_key.get_secret_value() == "qw-test"


def test_openai_keeps_its_default_endpoint(secrets):
    m = build_chat_model("openai", "gpt-4o-mini", secrets)
    assert "deepseek" not in str(m.openai_api_base or "")
    assert "dashscope" not in str(m.openai_api_base or "")


def test_unknown_provider_is_rejected(secrets):
    with pytest.raises(ProviderError, match="Unknown LLM provider"):
        build_chat_model("nope", "x", secrets)


# --- DeepInfra ---------------------------------------------------------------
# Another OpenAI-compatible endpoint, which is exactly why it needs pinning:
# the failure mode is silently talking to OpenAI with a DeepInfra token.

def test_deepinfra_uses_its_own_endpoint_and_token():
    s = Secrets(DEEPINFRA_TOKEN="di-test")
    m = build_chat_model("deepinfra", "deepseek-ai/DeepSeek-V3", s)
    assert "api.deepinfra.com" in str(m.openai_api_base)
    assert str(m.openai_api_base).endswith("/v1/openai")
    assert m.openai_api_key.get_secret_value() == "di-test"


def test_deepinfra_accepts_the_api_key_spelling_too():
    """DeepInfra's docs say DEEPINFRA_TOKEN; DEEPINFRA_API_KEY is what people
    type first, and a key that is silently ignored looks identical to a key
    that is wrong."""
    s = Secrets(DEEPINFRA_API_KEY="di-alt")
    assert s.deepinfra_api_key == "di-alt"


def test_deepinfra_model_id_is_passed_through_verbatim():
    """Ids carry an org prefix and are case-sensitive — normalising or
    lowercasing them would 404 on a model that exists."""
    s = Secrets(DEEPINFRA_TOKEN="di-test")
    m = build_chat_model("deepinfra", "Qwen/Qwen3-235B-A22B", s)
    assert m.model_name == "Qwen/Qwen3-235B-A22B"


def test_deepinfra_counts_as_credentialled_only_with_a_token():
    from graphrag.llm import has_credentials

    assert has_credentials("deepinfra", Secrets(DEEPINFRA_TOKEN="di")) is True
    assert has_credentials("deepinfra", Secrets(DEEPINFRA_TOKEN="")) is False


def test_deepinfra_embedder_targets_the_same_endpoint():
    from graphrag.config.settings import EmbeddingCfg
    from graphrag.embeddings.api_providers import build_api_embedder

    cfg = EmbeddingCfg(provider="deepinfra", model="BAAI/bge-m3")
    embedder = build_api_embedder(cfg, Secrets(DEEPINFRA_TOKEN="di-test"))
    assert "api.deepinfra.com" in str(embedder._backend.openai_api_base)
    # bge-m3 is 1024-dim; the vector store is created with this number, so a
    # wrong default costs a full re-ingest to discover.
    assert embedder.dim == 1024


def test_deepinfra_can_be_a_generative_reranker():
    from graphrag.retrieval.reranker import _LLM_PROVIDERS

    assert "deepinfra" in _LLM_PROVIDERS


# --- Cohere chat --------------------------------------------------------------
# Cohere is the embeddings and rerank vendor; this is its *chat* surface, which
# lives on a different path and exists so chains can reach a third vendor.

def test_cohere_chat_uses_the_compatibility_path_not_the_native_one():
    """`/v1` is Cohere's native protocol and rejects the chat-completions shape.

    Only `/compatibility/v1` speaks OpenAI, and getting this wrong produces a
    404 on a model that exists.
    """
    s = Secrets(COHERE_API_KEY="co-test")
    m = build_chat_model("cohere", "command-a-03-2025", s)
    assert str(m.openai_api_base) == "https://api.cohere.ai/compatibility/v1"
    assert m.openai_api_key.get_secret_value() == "co-test"


def test_cohere_counts_as_credentialled_only_with_a_key():
    from graphrag.llm import has_credentials

    assert has_credentials("cohere", Secrets(COHERE_API_KEY="co")) is True
    assert has_credentials("cohere", Secrets(COHERE_API_KEY="")) is False


def test_cohere_is_not_a_generative_reranker():
    """In the rerank config `cohere` means the native rerank endpoint.

    Listing it as an LLM provider would route rerank fallbacks to a chat model
    scoring 0-10 instead of the calibrated reranker, on a different score scale
    from the one `min_relevance` is tuned against.
    """
    from graphrag.retrieval.reranker import _LLM_PROVIDERS

    assert "cohere" not in _LLM_PROVIDERS
