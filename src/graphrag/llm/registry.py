"""The chat-model registry: which (provider, model) pairs a request may use.

User-facing model selection must never pass raw strings into a provider
client — the request names a model id, the registry decides whether it is
allowed and returns the validated pair. Unknown or disabled ids fall back to
the default instead of erroring, so a stale UI selection degrades gracefully.
"""

from __future__ import annotations

from graphrag.config.settings import AllowedModel, Settings


def allowed_models(
    settings: Settings, enabled: list[str] | None = None
) -> list[AllowedModel]:
    """The selectable models. An empty `llm.allowed` means the configured
    default model is the only choice.

    `enabled` is the admin's narrowing of that list, from the console. It is a
    filter, never a source: an id an admin enabled but the profile does not
    allow stays unreachable, so the console cannot conjure a model the
    deployment has no credentials for. An `enabled` list that matches nothing is
    ignored rather than honoured — leaving users with no model to talk to is a
    worse outcome than a stale narrowing.
    """
    configured = (
        list(settings.llm.allowed)
        if settings.llm.allowed
        else [
            AllowedModel(
                provider=settings.llm.provider,
                model=settings.llm.model,
                label=settings.llm.model,
                default=True,
            )
        ]
    )
    if not enabled:
        return configured
    return [m for m in configured if m.model in enabled] or configured


def resolve_model(
    requested: str | None,
    settings: Settings,
    enabled: list[str] | None = None,
) -> AllowedModel:
    """Map a request-supplied model id to an allowed (provider, model) pair.

    `enabled` optionally narrows the YAML list further (admin-controlled). An
    empty admin list must not brick chat, so it is ignored rather than honored.

    A model the admin disabled falls back to the default rather than erroring,
    which is what makes disabling one safe while a user has it selected in a
    stale tab.
    """
    models = allowed_models(settings, enabled)
    if requested:
        for m in models:
            if m.model == requested:
                return m
    return next((m for m in models if m.default), models[0])
