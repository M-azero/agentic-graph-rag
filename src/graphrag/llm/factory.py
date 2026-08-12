"""Build a LangChain chat model for any provider from one config shape.

Every returned model exposes the same interface (`.invoke`, `.astream`,
`.bind_tools`), which is what lets the agent be provider-agnostic and lets you
swap local <-> API with a single config line.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from graphrag.config.settings import FallbackModel, Secrets
from graphrag.core.errors import ProviderError
from graphrag.core.logging import get_logger

log = get_logger(__name__)

# One OpenAI-compatible endpoint serves DeepInfra's chat AND embedding models,
# so both the chat factory and the embedder point here.
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"

# Cohere speaks OpenAI on a SEPARATE path from its native API. The
# /compatibility/v1 prefix is required — plain /v1 is the native protocol and
# rejects the chat-completions shape. The embedder and reranker keep using the
# native SDK; this is only for chat/vision, where it buys a third vendor on a
# key the deployment already holds.
COHERE_BASE_URL = "https://api.cohere.ai/compatibility/v1"

# Which secret each provider authenticates with. Ollama is absent because it
# needs none — it authenticates by being reachable.
_CREDENTIAL = {
    "anthropic": "anthropic_api_key",
    "openai": "openai_api_key",
    "gemini": "google_api_key",
    "deepseek": "deepseek_api_key",
    "qwen": "dashscope_api_key",
    "deepinfra": "deepinfra_api_key",
    "cohere": "cohere_api_key",
}


def chat_providers() -> set[str]:
    """Every provider `build_chat_model` knows how to construct.

    Exported so the config loader can validate `GRAPHRAG_LLM` against what the
    factory actually supports instead of keeping its own copy of the list —
    which drifted, and rejected the two providers the production profile runs on.

    Ollama is added explicitly: it is the one provider with no credential, so it
    has no entry in `_CREDENTIAL`.
    """
    return set(_CREDENTIAL) | {"ollama"}


def has_credentials(provider: str, secrets: Secrets) -> bool:
    """Is this provider configured well enough to be worth calling?

    Used to drop unusable links from a fallback chain at build time. A member
    with no key fails on every request, and a chain that always burns a call to
    discover that is worse than one that never had the link.
    """
    field = _CREDENTIAL.get(provider)
    if field is None:
        return True  # ollama and anything else keyless
    return bool((getattr(secrets, field, None) or "").strip())


def build_chat_model(
    provider: str,
    model: str,
    secrets: Secrets,
    *,
    temperature: float = 0.1,
    max_tokens: int = 2048,
    extra: dict[str, Any] | None = None,
) -> BaseChatModel:
    extra = extra or {}
    try:
        if provider == "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=model,
                base_url=secrets.ollama_base_url,
                temperature=temperature,
                num_predict=max_tokens,
                **extra,
            )
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            # The Anthropic API rejects temperature modifications when extended
            # thinking is on — a configured temperature would 400 every request.
            if "thinking" in extra:
                temperature = 1
            return ChatAnthropic(
                model=model,
                api_key=secrets.anthropic_api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                **extra,
            )
        if provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=model,
                api_key=secrets.openai_api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                **extra,
            )
        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=secrets.google_api_key,
                temperature=temperature,
                max_output_tokens=max_tokens,
                **extra,
            )
        if provider == "deepseek":
            # OpenAI-compatible endpoint. Use the v4 names (deepseek-v4-flash /
            # deepseek-v4-pro) — the old deepseek-chat/-reasoner aliases were
            # retired July 2026.
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=model,
                api_key=secrets.deepseek_api_key,
                base_url="https://api.deepseek.com/v1",
                temperature=temperature,
                max_tokens=max_tokens,
                **extra,
            )
        if provider == "qwen":
            # Alibaba DashScope, OpenAI-compatible mode (qwen3.6-plus / -flash).
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=model,
                api_key=secrets.dashscope_api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                temperature=temperature,
                max_tokens=max_tokens,
                **extra,
            )
        if provider == "deepinfra":
            # Serverless open-weight models behind an OpenAI-compatible API.
            # Model ids carry their upstream org prefix and are case-sensitive
            # — "deepseek-ai/DeepSeek-V3", not "deepseek-v3". Browse
            # https://deepinfra.com/models for what is currently served.
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=model,
                api_key=secrets.deepinfra_api_key,
                base_url=DEEPINFRA_BASE_URL,
                temperature=temperature,
                max_tokens=max_tokens,
                **extra,
            )
        if provider == "cohere":
            # Cohere's OpenAI-compatibility endpoint (command-a-*, command-r*).
            # Verified to emit real tool calls and to accept image_url content,
            # so it can serve as a chat, OCR or rerank link — not just a name
            # in the chain.
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=model,
                api_key=secrets.cohere_api_key,
                base_url=COHERE_BASE_URL,
                temperature=temperature,
                max_tokens=max_tokens,
                **extra,
            )
    except ImportError as exc:  # pragma: no cover
        raise ProviderError(f"LLM provider '{provider}' package is not installed") from exc

    raise ProviderError(f"Unknown LLM provider: {provider}")


def build_chat_chain(
    provider: str,
    model: str,
    secrets: Secrets,
    *,
    fallbacks: Sequence[FallbackModel] = (),
    temperature: float = 0.1,
    max_tokens: int = 2048,
    extra: dict[str, Any] | None = None,
    max_failures: int = 2,
    cooldown_seconds: float = 300.0,
) -> BaseChatModel:
    """The primary model, plus any configured fallbacks, as one chat model.

    With no usable fallbacks this returns the primary model itself, so nothing
    pays for the wrapper unless failover is actually configured.

    `extra` applies only to the primary: provider kwargs are model-specific
    (Anthropic's `thinking`, Ollama's `reasoning`), and forwarding them to a
    different provider is at best ignored and at worst a 400.
    """
    primary = build_chat_model(
        provider, model, secrets,
        temperature=temperature, max_tokens=max_tokens, extra=extra,
    )
    if not fallbacks:
        return primary

    members: list[Any] = [primary]
    labels = [f"{provider}:{model}"]
    for fb in fallbacks:
        label = f"{fb.provider}:{fb.model}"
        if (fb.provider, fb.model) == (provider, model):
            continue  # a chain that retries the same model is just a retry loop
        if not has_credentials(fb.provider, secrets):
            log.warning("fallback_skipped", model=label, reason="no api key configured")
            continue
        try:
            members.append(
                build_chat_model(
                    fb.provider, fb.model, secrets,
                    temperature=temperature, max_tokens=max_tokens,
                )
            )
        except Exception as exc:
            # A broken fallback must never stop the app from booting on a
            # working primary.
            log.warning("fallback_skipped", model=label, reason=str(exc))
            continue
        labels.append(label)

    if len(members) == 1:
        return primary

    from graphrag.llm.fallback import FallbackChatModel, Health

    log.info("llm_chain_ready", chain=" -> ".join(labels))
    return FallbackChatModel(
        members=members,
        labels=labels,
        health=Health(len(members), max_failures, cooldown_seconds),
    )
