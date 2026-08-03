"""Multi-provider judge backends.

One tiny async surface — :class:`LLMProvider.complete` — returns raw model text that the
judge parses. Three implementations:

* :class:`AnthropicProvider`      — official ``AsyncAnthropic`` SDK.
* :class:`OpenAICompatProvider`   — ``AsyncOpenAI`` + configurable ``base_url`` (covers
  OpenAI, DeepSeek, Qwen, Gemini's OpenAI-compat endpoint, Groq, Together, vLLM, Ollama).
* :class:`MockProvider`           — deterministic, offline; drives every test.

``PRESETS`` maps a provider name to its base URL, native key env var, default model, and
whether it supports JSON mode. ``build_provider(settings)`` wires the right one together;
``build_provider_chain(settings)`` adds the ``GUARD_LLM_FALLBACKS`` links behind it so a
dead judge degrades to the next vendor instead of taking screening down with it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from ..config import Settings
from .errors import JudgeAPIError, JudgeRefusal, JudgeTimeout

ProviderType = Literal["anthropic", "openai", "mock"]


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal async completion surface the judge depends on."""

    name: str
    model: str | None

    async def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        timeout: float,
        json_schema: dict | None = None,
    ) -> str: ...


# ── Preset table ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Preset:
    provider_type: ProviderType
    base_url: str | None
    key_env: str | None
    default_model: str | None
    supports_json_mode: bool


PRESETS: dict[str, Preset] = {
    "anthropic": Preset("anthropic", None, "ANTHROPIC_API_KEY", "claude-opus-4-8", True),
    "openai": Preset("openai", "https://api.openai.com/v1", "OPENAI_API_KEY", None, True),
    "gemini": Preset(
        "openai",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY",
        "gemini-2.5-flash",
        True,
    ),
    "deepseek": Preset("openai", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", "deepseek-chat", True),
    # Cohere speaks OpenAI on a *separate* path from its native API — the
    # /compatibility/v1 prefix is required, /v1 is the native protocol and 404s
    # the chat-completions shape.
    "cohere": Preset(
        "openai",
        "https://api.cohere.ai/compatibility/v1",
        "COHERE_API_KEY",
        "command-a-03-2025",
        True,
    ),
    "qwen": Preset(
        "openai",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "DASHSCOPE_API_KEY",
        "qwen-plus",
        True,
    ),
    "groq": Preset("openai", "https://api.groq.com/openai/v1", "GROQ_API_KEY", None, True),
    "together": Preset("openai", "https://api.together.xyz/v1", "TOGETHER_API_KEY", None, True),
    # Local, offline judge. Default targets a stock `ollama pull llama3.1`; override the
    # model with GUARD_LLM_MODEL to match whatever you have pulled.
    "ollama": Preset("openai", "http://localhost:11434/v1", None, "llama3.1", False),
    "vllm": Preset("openai", "http://localhost:8000/v1", None, None, False),
    "custom": Preset("openai", None, None, None, False),
    "mock": Preset("mock", None, None, "mock-1", False),
}


# ── Anthropic ───────────────────────────────────────────────────────────────
class AnthropicProvider:
    """Judge backend using the official Anthropic async SDK."""

    def __init__(self, *, model: str, api_key: str | None, timeout: float) -> None:
        from anthropic import AsyncAnthropic

        self.name = "anthropic"
        self.model = model
        # No client-side retries (the pipeline owns fail behaviour).
        self._client = AsyncAnthropic(api_key=api_key, timeout=timeout, max_retries=0)
        self._use_output_config = True  # disabled permanently on first 400

    async def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        timeout: float,
        json_schema: dict | None = None,
    ) -> str:
        import anthropic

        # No temperature/top_p/top_k, no assistant prefill, no thinking (fastest classifier).
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        client = self._client.with_options(timeout=timeout)
        try:
            if json_schema is not None and self._use_output_config:
                try:
                    resp = await client.messages.create(
                        **kwargs,
                        output_config={"format": {"type": "json_schema", "schema": json_schema}},
                    )
                except (TypeError, anthropic.APIStatusError) as exc:
                    if _is_output_config_error(exc):
                        self._use_output_config = False  # permanent for the process
                        resp = await client.messages.create(**kwargs)
                    else:
                        raise
            else:
                resp = await client.messages.create(**kwargs)
        except anthropic.APITimeoutError as exc:
            raise JudgeTimeout(str(exc)) from exc
        except (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.APIStatusError) as exc:
            raise JudgeAPIError(str(exc)) from exc

        if getattr(resp, "stop_reason", None) == "refusal":
            raise JudgeRefusal("model refused to classify")
        return _anthropic_text(resp)


def _is_output_config_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    if isinstance(exc, TypeError):
        return "output_config" in msg or "unexpected keyword" in msg
    status = getattr(exc, "status_code", None)
    return status == 400 and "output_config" in msg


def _anthropic_text(resp: object) -> str:
    parts: list[str] = []
    for block in getattr(resp, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "".join(parts)


# ── OpenAI-compatible ───────────────────────────────────────────────────────
class OpenAICompatProvider:
    """Judge backend for any OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        *,
        name: str,
        model: str,
        api_key: str | None,
        base_url: str | None,
        timeout: float,
        supports_json_mode: bool,
    ) -> None:
        from openai import AsyncOpenAI

        self.name = name
        self.model = model
        self._client = AsyncOpenAI(
            api_key=api_key or "local", base_url=base_url, timeout=timeout, max_retries=0
        )
        self._use_json_mode = supports_json_mode

    async def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        timeout: float,
        json_schema: dict | None = None,
    ) -> str:
        import openai

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        async def _call(json_mode: bool):
            extra: dict = {"response_format": {"type": "json_object"}} if json_mode else {}
            return await self._client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=messages,  # type: ignore[arg-type]
                timeout=timeout,
                **extra,
            )

        want_json = json_schema is not None and self._use_json_mode
        try:
            resp = await _call(want_json)
        except openai.APITimeoutError as exc:
            raise JudgeTimeout(str(exc)) from exc
        except openai.BadRequestError as exc:
            # Endpoint rejected response_format -> drop JSON mode for the process, retry.
            if want_json:
                self._use_json_mode = False
                try:
                    resp = await _call(False)
                except openai.OpenAIError as exc2:
                    raise JudgeAPIError(str(exc2)) from exc2
            else:
                raise JudgeAPIError(str(exc)) from exc
        except openai.OpenAIError as exc:
            raise JudgeAPIError(str(exc)) from exc

        content = resp.choices[0].message.content if resp.choices else None
        if not content:
            raise JudgeAPIError("empty completion from provider")
        return content


# ── Mock ────────────────────────────────────────────────────────────────────
_REPAIR_MARKER = "RETURN ONLY VALID JSON"


class MockProvider:
    """Deterministic offline provider. Verdict is driven by MOCK_* markers in the text.

    Markers: MOCK_INJECTION, MOCK_JAILBREAK, MOCK_OFFTOPIC, MOCK_HARM, MOCK_UNGROUNDED,
    MOCK_TIMEOUT, MOCK_APIERROR, MOCK_REFUSAL, MOCK_BADJSON. Absent any marker -> allow.
    """

    def __init__(self, model: str = "mock-1") -> None:
        self.name = "mock"
        self.model = model
        self.calls = 0

    async def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        timeout: float,
        json_schema: dict | None = None,
    ) -> str:
        self.calls += 1
        blob = f"{system}\n{user}"

        if "MOCK_TIMEOUT" in blob:
            raise JudgeTimeout("mock timeout")
        if "MOCK_APIERROR" in blob:
            raise JudgeAPIError("mock api error")
        if "MOCK_REFUSAL" in blob:
            raise JudgeRefusal("mock refusal")

        is_repair = _REPAIR_MARKER in system.upper()
        if "MOCK_BADJSON" in blob and not is_repair:
            return "```json\n{ prompt_injection: 0.0, , invalid }\n```"

        is_output = bool(json_schema) and "ungrounded" in json.dumps(json_schema)
        verdict = _mock_output_verdict(blob) if is_output else _mock_input_verdict(blob)
        return json.dumps(verdict)


def _mock_input_verdict(blob: str) -> dict:
    flagged: list[str] = []
    v = {"prompt_injection": 0.0, "jailbreak": 0.0, "off_topic": 0.0, "harmful_content": 0.0}
    if "MOCK_INJECTION" in blob:
        v["prompt_injection"] = 0.95
        flagged.append("MOCK_INJECTION")
    if "MOCK_JAILBREAK" in blob:
        v["jailbreak"] = 0.95
        flagged.append("MOCK_JAILBREAK")
    if "MOCK_OFFTOPIC" in blob:
        v["off_topic"] = 0.8
    if "MOCK_HARM" in blob:
        v["harmful_content"] = 0.9
    return {**v, "reason": "mock verdict", "flagged_phrases": flagged}


def _mock_output_verdict(blob: str) -> dict:
    claims: list[str] = []
    v = {"ungrounded": 0.0, "harmful_content": 0.0, "off_topic": 0.0}
    if "MOCK_UNGROUNDED" in blob:
        v["ungrounded"] = 0.85
        claims.append("an unsupported claim")
    if "MOCK_HARM" in blob:
        v["harmful_content"] = 0.9
    if "MOCK_OFFTOPIC" in blob:
        v["off_topic"] = 0.8
    return {**v, "unsupported_claims": claims, "reason": "mock verdict"}


# ── Factory ─────────────────────────────────────────────────────────────────
def _construct(
    *, name: str, preset: Preset, model: str, base_url: str | None,
    api_key: str | None, timeout: float,
) -> LLMProvider:
    if preset.provider_type == "mock":
        return MockProvider(model=model)
    if preset.provider_type == "anthropic":
        return AnthropicProvider(model=model, api_key=api_key, timeout=timeout)
    return OpenAICompatProvider(
        name=name,
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        supports_json_mode=preset.supports_json_mode,
    )


def build_provider(settings: Settings) -> LLMProvider:
    """Construct the configured provider, resolving model / key / base_url from presets."""
    preset = PRESETS.get(settings.llm_provider)
    if preset is None:
        raise ValueError(
            f"Unknown GUARD_LLM_PROVIDER={settings.llm_provider!r}; "
            f"choose one of: {', '.join(sorted(PRESETS))}"
        )

    if preset.provider_type == "mock":
        return MockProvider(model=settings.llm_model or preset.default_model or "mock-1")

    model = settings.llm_model or preset.default_model
    if not model:
        raise ValueError(
            f"GUARD_LLM_MODEL is required for provider {settings.llm_provider!r} "
            "(this preset has no default model)."
        )

    base_url = settings.llm_base_url or preset.base_url
    if settings.llm_provider == "custom" and not base_url:
        raise ValueError("GUARD_LLM_BASE_URL is required for the 'custom' provider.")

    api_key = settings.llm_api_key or (os.environ.get(preset.key_env) if preset.key_env else None)

    return _construct(
        name=settings.llm_provider, preset=preset, model=model, base_url=base_url,
        api_key=api_key, timeout=settings.llm_timeout_s,
    )


# ── Failover chain ──────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ChainLink:
    """One parsed entry of ``GUARD_LLM_FALLBACKS``."""

    provider: str
    model: str | None = None
    base_url: str | None = None

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}" if self.model else self.provider


def parse_chain_spec(spec: str) -> list[ChainLink]:
    """Parse ``"deepseek:deepseek-v4-flash, cohere:command-a-03-2025"``.

    Grammar per entry: ``provider[:model][@base_url]``. Provider is split on the
    FIRST colon so model ids keep theirs; the ``@base_url`` suffix is only
    recognised when the tail is an http(s) URL, so an ``@`` inside a model id
    (or a userinfo-bearing URL) does not silently truncate the model name.
    """
    links: list[ChainLink] = []
    for raw in spec.split(","):
        entry = raw.strip()
        if not entry:
            continue
        provider, _, rest = entry.partition(":")
        provider = provider.strip()
        if not provider:
            raise ValueError(f"malformed GUARD_LLM_FALLBACKS entry: {entry!r}")

        base_url: str | None = None
        head, sep, tail = rest.rpartition("@")
        if sep and tail.startswith(("http://", "https://")):
            rest, base_url = head, tail

        links.append(ChainLink(provider, rest.strip() or None, base_url))
    return links


def _resolve_key(link: ChainLink, preset: Preset, settings: Settings) -> str | None:
    """Find the key for a fallback link.

    The preset's own env var wins. ``GUARD_LLM_API_KEY`` holds the *primary's*
    credential, and handing a DeepInfra token to DeepSeek does not fail loudly —
    it 401s on every request, so the link looks configured, reports healthy at
    startup, and contributes nothing. It is only reused when the link names the
    same provider as the primary, or when the preset has no key of its own
    (``custom``/``ollama``/``vllm``).
    """
    env_key = os.environ.get(preset.key_env) if preset.key_env else None
    if env_key:
        return env_key
    if link.provider == settings.llm_provider or preset.key_env is None:
        return settings.llm_api_key
    return None


def build_provider_chain(settings: Settings) -> tuple[list[LLMProvider], list[str]]:
    """The primary judge plus every usable fallback, in order.

    Returns ``(providers, skipped)`` — ``skipped`` names the links that were
    dropped and why, so startup can log them. A link is dropped rather than
    built when it has no credential or cannot be constructed: the chain exists
    to be traversed under load, and a link that is certain to fail only spends
    the failure budget that a working link further down needs.
    """
    providers: list[LLMProvider] = [build_provider(settings)]
    seen = {(settings.llm_provider, providers[0].model)}
    skipped: list[str] = []

    for link in parse_chain_spec(settings.llm_fallbacks):
        preset = PRESETS.get(link.provider)
        if preset is None:
            skipped.append(f"{link.label} (unknown provider)")
            continue

        model = link.model or preset.default_model
        if not model:
            skipped.append(f"{link.label} (no model, and this preset has no default)")
            continue
        if (link.provider, model) in seen:
            skipped.append(f"{link.provider}:{model} (duplicate)")
            continue

        base_url = link.base_url or preset.base_url
        if preset.provider_type == "openai" and not base_url:
            skipped.append(f"{link.provider}:{model} (no base_url; use provider:model@https://...)")
            continue

        api_key = _resolve_key(link, preset, settings)
        if preset.key_env and not api_key:
            skipped.append(f"{link.provider}:{model} (no {preset.key_env})")
            continue

        try:
            providers.append(_construct(
                name=link.provider, preset=preset, model=model, base_url=base_url,
                api_key=api_key, timeout=settings.llm_timeout_s,
            ))
        except Exception as exc:  # noqa: BLE001 - a bad link must not stop startup
            skipped.append(f"{link.provider}:{model} ({type(exc).__name__}: {exc})")
            continue
        seen.add((link.provider, model))

    return providers, skipped
