"""The admin's model narrowing is a control, not a display preference.

`PUT /admin/models` validated the list, wrote it to `app_settings`, audited the
change — and nothing ever read it back for enforcement. `resolve_model` accepted
an `enabled` argument for exactly this and every caller omitted it, so disabling
a model removed it from the picker and left it fully callable by anyone who sent
`{"model": "..."}` to /query.

The tests pin both halves: the registry filters, and the request path passes the
list in. Either alone is the bug.
"""

from __future__ import annotations

from graphrag.config.settings import AllowedModel, LLMCfg, Settings
from graphrag.llm.registry import allowed_models, resolve_model

TWO_MODELS = Settings(
    llm=LLMCfg(
        provider="deepseek",
        model="fast",
        allowed=[
            AllowedModel(provider="deepseek", model="fast", label="Fast", default=True),
            AllowedModel(provider="deepinfra", model="expensive", label="Expensive"),
        ],
    )
)


# -- the registry -------------------------------------------------------------

def test_no_narrowing_offers_everything_configured():
    assert [m.model for m in allowed_models(TWO_MODELS)] == ["fast", "expensive"]
    assert [m.model for m in allowed_models(TWO_MODELS, None)] == ["fast", "expensive"]


def test_narrowing_hides_the_disabled_model():
    assert [m.model for m in allowed_models(TWO_MODELS, ["fast"])] == ["fast"]


def test_a_disabled_model_cannot_be_requested():
    """The whole point: asking for it by name falls back to the default."""
    assert resolve_model("expensive", TWO_MODELS).model == "expensive"
    assert resolve_model("expensive", TWO_MODELS, ["fast"]).model == "fast"


def test_narrowing_cannot_add_a_model_the_profile_forbids():
    """`enabled` is a filter, never a source — otherwise the console could name
    a provider this deployment has no credentials for."""
    picked = allowed_models(TWO_MODELS, ["expensive", "gpt-9-ultra"])
    assert [m.model for m in picked] == ["expensive"]
    assert resolve_model("gpt-9-ultra", TWO_MODELS, ["expensive"]).model == "expensive"


def test_a_narrowing_that_matches_nothing_is_ignored():
    """Leaving users with no model to talk to is worse than a stale list."""
    assert [m.model for m in allowed_models(TWO_MODELS, ["retired"])] == [
        "fast", "expensive"
    ]


def test_the_default_follows_the_narrowing():
    """Disabling the profile default must still yield a usable default."""
    assert resolve_model(None, TWO_MODELS, ["expensive"]).model == "expensive"


# -- the request path ---------------------------------------------------------

def test_query_resolves_against_the_admin_list():
    """`_pick_model` is what /query and /compare call; it must read app.state."""
    from types import SimpleNamespace

    from graphrag.api.routers.query import _pick_model

    container = SimpleNamespace(settings=TWO_MODELS)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        enabled_models=["fast"]
    )))

    assert _pick_model(request, container, "expensive").model == "fast"
    assert _pick_model(request, container, None) is None

    request.app.state.enabled_models = None
    assert _pick_model(request, container, "expensive").model == "expensive"


def test_state_missing_the_attribute_does_not_break_resolution():
    """Scripts and tests build the app without the lifespan."""
    from types import SimpleNamespace

    from graphrag.api.routers.query import _pick_model

    container = SimpleNamespace(settings=TWO_MODELS)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    assert _pick_model(request, container, "expensive").model == "expensive"
