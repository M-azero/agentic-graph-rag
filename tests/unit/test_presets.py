"""Job presets: the ten working methods, and the rules that keep them safe.

Three things are being protected here.

The prompt: a preset is interpolated into a hardened, closed-domain system
message, so it must arrive from a server-side table via an enum and must not
carry anything that widens what the model may do.

Compatibility: `general` has to keep rendering the style instruction, or every
API client and the CLI silently change behaviour on upgrade.

The files themselves: the bodies live in `prompts/*.md`, which means they are
edited by people who are not editing this module — so the loader's failure modes
(missing file, mismatched id, stray brace) are tested as behaviour rather than
left to be discovered at runtime.
"""

from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from graphrag.agent import presets as presets_mod
from graphrag.agent.graph import AgentRunner
from graphrag.agent.presets import (
    answer_instruction,
    canonical_preset,
    preset_options,
    prompt_dir,
)
from graphrag.agent.prompts import SYSTEM_PROMPT
from graphrag.agent.styles import style_instruction
from graphrag.core.errors import ConfigError
from graphrag.core.types import AnswerPreset, AnswerStyle

# The ten jobs the picker offers, over and above the neutral default.
_JOBS = [p for p in AnswerPreset if p is not AnswerPreset.GENERAL]


@pytest.fixture(autouse=True)
def _fresh_cache():
    """The loader caches for the process; these tests change what it reads."""
    presets_mod._presets.cache_clear()
    yield
    presets_mod._presets.cache_clear()


def test_ten_jobs_plus_a_neutral_default():
    assert len(_JOBS) == 10
    assert preset_options()[0].id is AnswerPreset.GENERAL  # default sorts first
    assert len(preset_options()) == len(AnswerPreset)


def test_every_preset_has_what_the_picker_needs():
    for preset in preset_options():
        assert preset.label and preset.emoji and preset.description


def test_every_enum_member_has_a_prompt_file():
    """Adding an AnswerPreset without its file would otherwise fail at the first
    request that selected it, not at startup."""
    for preset in AnswerPreset:
        assert (prompt_dir() / f"{preset.value}.md").is_file()


def test_no_orphan_prompt_files():
    """A `.md` with no enum member is unreachable — most likely a rename that
    only happened on one side."""
    ids = {p.value for p in AnswerPreset}
    for path in prompt_dir().glob("*.md"):
        if path.stem == "README":
            continue
        assert path.stem in ids, f"{path.name} matches no AnswerPreset"


@pytest.mark.parametrize("junk", ["banana", "", "STUDY-ish", "../etc", None])
def test_unknown_presets_clamp_to_general(junk):
    """A preset arrives as a raw request string. If it keyed anything raw, junk
    would reach the prompt — and mint an unbounded number of agent cache keys."""
    assert canonical_preset(junk) is AnswerPreset.GENERAL


def test_general_still_carries_the_style_instruction():
    """The compatibility guarantee. A caller that sends no preset — the CLI, an
    API key holder, an older UI build — must still get style-controlled
    phrasing, and each style must still render differently."""
    rendered = {}
    for style in AnswerStyle:
        for named in (answer_instruction(None, style), answer_instruction("general", style)):
            assert style_instruction(style) in named
        rendered[style] = answer_instruction(None, style)
    assert len(set(rendered.values())) == len(AnswerStyle)


def test_general_puts_the_style_last():
    """It lands under general.md's trailing `## Length and register` heading, so
    the two compose into one section rather than reading as an afterthought."""
    body = answer_instruction("general", "concise")
    assert body.index("## Length and register") < body.index(
        style_instruction(AnswerStyle.CONCISE)
    )


def test_a_job_preset_replaces_the_style_rather_than_stacking():
    """The two control the same axis. Concatenating "thorough, explain the
    reasoning" with "tight bullets, front-loaded" hands the model a
    contradiction and it picks one at random."""
    for style in AnswerStyle:
        rendered = answer_instruction(AnswerPreset.SUMMARY, style)
        assert style_instruction(style) not in rendered


def test_every_preset_renders_a_complete_prompt():
    """SYSTEM_PROMPT has exactly one placeholder, and a preset must not
    introduce another — an unfilled `{}` would reach the model verbatim."""
    for preset in AnswerPreset:
        rendered = SYSTEM_PROMPT.format(style=answer_instruction(preset, "detailed"))
        assert "{" not in rendered and "}" not in rendered
        assert rendered.count("# Answer style") == 1


def test_no_preset_weakens_the_grounding_rules():
    """A preset narrows what counts as a good answer; it never grants the model
    permission the prompt above it withholds. This catches the tempting edit —
    telling the finance preset to "calculate the ratio", say — that would quietly
    undo the closed-domain contract."""
    # Phrasings that would *grant* something, as opposed to the many phrasings
    # that tighten. "If the documents do not cover it, say so" reinforces the
    # closed-domain rule and must not be caught here.
    forbidden = (
        "ignore the above", "ignore the instructions",
        "your own knowledge", "general knowledge", "from memory",
        "you may assume", "you can assume", "make up", "fill in the gap",
        "if you are unsure, ", "use your judgement", "use your judgment",
        "even if the documents", "without a citation", "no need to cite",
    )
    for preset in preset_options():
        body = preset.instruction.lower()
        for phrase in forbidden:
            assert phrase not in body, f"{preset.id} relaxes grounding: {phrase!r}"


# Each professional preset must restate its own hard limits rather than rely on
# the general rules being remembered that far down the prompt. Listed as
# alternatives so the wording can be revised without the test becoming a
# spell-checker — what must survive is the guarantee, not the sentence.
_HARD_LIMITS = {
    AnswerPreset.FINANCE: [
        ("do not compute", "never compute", "not yours to recompute"),
        ("not investment advice", "not financial advice"),
        ("never convert", "do not convert", "conversions"),
    ],
    AnswerPreset.LEGAL: [
        ("not legal advice",),
        ("verbatim",),
        ("never resolve an ambiguity", "do not resolve"),
    ],
    AnswerPreset.MEDICAL: [
        ("not medical advice",),
        ("never convert", "do not convert"),
        ("never round", "do not round"),
    ],
    AnswerPreset.CODE: [("never invent", "do not invent")],
    AnswerPreset.WRITING: [("gap",)],
}


@pytest.mark.parametrize("preset", list(_HARD_LIMITS))
def test_each_preset_restates_its_own_hard_limits(preset):
    body = {p.id: p.instruction.lower() for p in preset_options()}[preset]
    for alternatives in _HARD_LIMITS[preset]:
        assert any(phrase in body for phrase in alternatives), (
            f"{preset.value} no longer states: {alternatives[0]!r}"
        )


# --------------------------------------------------------------------------
# Loading the files
# --------------------------------------------------------------------------

def _prompt_set(tmp_path: Path, **overrides: str) -> Path:
    """A complete prompt directory, with named files replaced."""
    for preset in AnswerPreset:
        name = f"{preset.value}.md"
        text = overrides.get(
            preset.value,
            f"---\nid: {preset.value}\nlabel: X\nemoji: x\ndescription: d\n---\n\nBody.\n",
        )
        (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path


def _load_from(monkeypatch, directory: Path):
    monkeypatch.setenv("GRAPHRAG_PROMPT_DIR", str(directory))
    presets_mod._presets.cache_clear()
    return preset_options()


def test_a_custom_prompt_directory_is_honoured(monkeypatch, tmp_path):
    """Deployments override the directory rather than patching the image; the
    Docker build sets GRAPHRAG_PROMPT_DIR for exactly this reason."""
    loaded = _load_from(monkeypatch, _prompt_set(tmp_path))
    assert len(loaded) == len(AnswerPreset)
    assert all(p.instruction.strip() == "Body." for p in loaded)


def test_a_missing_directory_says_so(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHRAG_PROMPT_DIR", str(tmp_path / "nope"))
    presets_mod._presets.cache_clear()
    with pytest.raises(ConfigError, match="Prompt directory not found"):
        preset_options()


def test_a_missing_file_names_the_file(monkeypatch, tmp_path):
    _prompt_set(tmp_path)
    (tmp_path / "finance.md").unlink()
    monkeypatch.setenv("GRAPHRAG_PROMPT_DIR", str(tmp_path))
    presets_mod._presets.cache_clear()
    with pytest.raises(ConfigError, match="finance.md"):
        preset_options()


def test_an_id_that_disagrees_with_its_filename_is_rejected(monkeypatch, tmp_path):
    """The drift that copying a file to start a new preset produces — and which
    would otherwise ship one preset's text under another's name, silently."""
    wrong = "---\nid: legal\nlabel: X\nemoji: x\ndescription: d\n---\n\nBody.\n"
    directory = _prompt_set(tmp_path, finance=wrong)
    monkeypatch.setenv("GRAPHRAG_PROMPT_DIR", str(directory))
    presets_mod._presets.cache_clear()
    with pytest.raises(ConfigError, match="declares id"):
        preset_options()


@pytest.mark.parametrize(
    ("bad", "expected"),
    [
        ("Body with no frontmatter.\n", "frontmatter"),
        ("---\nid: finance\n\nBody.\n", "unterminated"),
        ("---\nid: finance\nlabel: X\nemoji: x\ndescription: d\n---\n\n", "empty body"),
        (
            "---\nid: finance\nlabel: X\nemoji: x\ndescription: d\n---\n\nUse {style}.\n",
            "curly brace",
        ),
    ],
)
def test_malformed_files_are_rejected_by_name(monkeypatch, tmp_path, bad, expected):
    """Each of these would otherwise surface far from its cause: a stray brace
    as a KeyError from inside str.format, an empty body as an agent that quietly
    has no answer instruction at all."""
    directory = _prompt_set(tmp_path, finance=bad)
    monkeypatch.setenv("GRAPHRAG_PROMPT_DIR", str(directory))
    presets_mod._presets.cache_clear()
    with pytest.raises(ConfigError, match=expected):
        preset_options()


# --------------------------------------------------------------------------
# Agent caching
# --------------------------------------------------------------------------

class _Model(FakeListChatModel):
    """Stand-in chat model. `create_react_agent` type-checks for a real
    `BaseChatModel` and binds tools to it; nothing here invokes it."""

    def __init__(self) -> None:
        super().__init__(responses=["unused"])

    def bind_tools(self, tools, **kwargs):
        return self


def _runner() -> AgentRunner:
    return AgentRunner(
        _Model(), vector=None, hybrid=None, graph=None, embedder=None
    )


def test_each_preset_compiles_its_own_agent():
    runner = _runner()
    agents = {runner.session("q", preset=str(p))._agent for p in AnswerPreset}
    assert len(agents) == len(AnswerPreset)


def test_junk_presets_share_the_default_agent():
    """The cache key goes through `canonical_preset`, so request input cannot
    mint entries."""
    runner = _runner()
    baseline = runner.session("q", preset="general")._agent
    for junk in ("banana", "", "../etc"):
        assert runner.session("q", preset=junk)._agent is baseline
    assert len(runner._agents) == 1
