"""Job presets: eleven working methods, authored as Markdown in `prompts/`.

A shelf of tax rulings and a shelf of lecture notes need the same retrieval and
completely different *handling* — one wants figures quoted to the digit with
their reporting basis, the other wants the concept built up from its definition.
The preset is that handling. It is chosen in the UI, stored as a per-shelf
default, and rendered into the one `{style}` slot of `SYSTEM_PROMPT`.

The text lives in `prompts/*.md` rather than in this module. Prompts are content:
they get read, argued over and revised far more often than the code around them,
they are the thing a non-Python reader most needs access to, and a diff of one
should not be a diff of a source file. `prompts/README.md` states the contract
they have to satisfy.

Three properties make this safe to interpolate into a hardened prompt:

- **Server-side only.** A request names a preset *id*; `canonical_preset` clamps
  it to the `AnswerPreset` enum before anything is looked up, so request text
  never reaches the prompt. Same guarantee `canonical_style` gives, for the same
  reason — and it doubles as the cache key bound in `AgentRunner._agent_for`.
- **They narrow, never widen.** No file grants outside knowledge, relaxes the
  citation requirement, or offers an escape from the closed-domain refusal.
  `tests/unit/test_presets.py` enforces that against every file, because it is
  the tempting edit: "calculate the ratio" in the finance preset reads as
  helpful and dismantles the guarantee the rest of the prompt exists to make.
- **GENERAL falls through to `AnswerStyle`.** Its body is followed by the
  requested style instruction, so a client that never heard of presets — the
  CLI, an API key holder, an old build of the UI — still gets style-controlled
  phrasing exactly as it did before presets existed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from graphrag.agent.styles import style_instruction
from graphrag.core.errors import ConfigError
from graphrag.core.types import AnswerPreset, AnswerStyle

# Picker order. Explicit rather than alphabetical or directory order: the
# neutral default has to come first, and the rest are grouped so the list reads
# as study → analysis → professional → craft rather than as a shuffled bag.
_ORDER: tuple[AnswerPreset, ...] = (
    AnswerPreset.GENERAL,
    AnswerPreset.STUDY,
    AnswerPreset.RESEARCH,
    AnswerPreset.FINANCE,
    AnswerPreset.LEGAL,
    AnswerPreset.MEDICAL,
    AnswerPreset.CODE,
    AnswerPreset.BUSINESS,
    AnswerPreset.WRITING,
    AnswerPreset.TEACHING,
    AnswerPreset.SUMMARY,
)

_FRONTMATTER = "---"


@dataclass(frozen=True, slots=True)
class Preset:
    """One preset, and everything the UI needs to offer it."""

    id: AnswerPreset
    label: str
    emoji: str
    description: str
    instruction: str


def _default_prompt_dir() -> Path:
    """Locate `prompts/` without assuming where the package lives.

    Mirrors `config.loader._default_config_dir` exactly, including its
    limitation: in a source checkout this file is
    `<repo>/src/graphrag/agent/presets.py`, so the directory sits three levels
    up; installed into site-packages that walk lands outside the project, so
    fall back to the working directory. The Docker image pins
    GRAPHRAG_PROMPT_DIR rather than relying on either guess.
    """
    checkout = Path(__file__).resolve().parents[3] / "prompts"
    return checkout if checkout.is_dir() else Path.cwd() / "prompts"


def prompt_dir() -> Path:
    """Where preset bodies are read from. `GRAPHRAG_PROMPT_DIR` overrides.

    Read from the environment directly rather than through `Secrets` to keep
    this module free of a config import — presets are loaded while the agent is
    being built, and `Settings` already depends on this direction.
    """
    override = os.environ.get("GRAPHRAG_PROMPT_DIR", "").strip()
    return Path(override) if override else _default_prompt_dir()


def _parse(path: Path) -> tuple[dict, str]:
    """Split `---` YAML frontmatter from the Markdown body.

    Hand-rolled rather than pulled from a library: the format is three lines of
    structure, and the failure modes worth catching are all about *this* file
    being wrong, which a generic parser reports far less usefully.
    """
    text = path.read_text(encoding="utf-8").lstrip("﻿")
    if not text.startswith(_FRONTMATTER):
        raise ConfigError(
            f"{path.name} has no '---' frontmatter block. See prompts/README.md."
        )
    _, _, rest = text.partition(_FRONTMATTER)
    raw_meta, sep, body = rest.partition("\n" + _FRONTMATTER)
    if not sep:
        raise ConfigError(f"{path.name} has an unterminated frontmatter block.")

    meta = yaml.safe_load(raw_meta) or {}
    if not isinstance(meta, dict):
        raise ConfigError(f"{path.name} frontmatter is not a mapping.")
    return meta, body.lstrip("\n").rstrip() + "\n"


def _load(preset: AnswerPreset, directory: Path) -> Preset:
    path = directory / f"{preset.value}.md"
    if not path.is_file():
        raise ConfigError(
            f"Missing prompt file: {path}. Every AnswerPreset needs one — set "
            "GRAPHRAG_PROMPT_DIR if the prompts live elsewhere."
        )
    meta, body = _parse(path)

    # The id is duplicated between the filename and the frontmatter, so check
    # they agree. They drift when a file is copied to start a new preset and the
    # frontmatter is not updated — which would otherwise ship one preset's text
    # under another's name, with nothing failing.
    declared = str(meta.get("id", "")).strip()
    if declared != preset.value:
        raise ConfigError(
            f"{path.name} declares id {declared!r} but its filename says "
            f"{preset.value!r}. They must match."
        )
    if not body.strip():
        raise ConfigError(f"{path.name} has an empty body.")
    # `SYSTEM_PROMPT` is rendered with str.format, so a stray brace in a prompt
    # would either raise or reach the model verbatim. Caught here, naming the
    # file, rather than as a KeyError from deep inside the agent.
    if "{" in body or "}" in body:
        raise ConfigError(
            f"{path.name} contains a curly brace. The prompt is rendered with "
            "str.format, so braces must not appear in preset bodies."
        )

    return Preset(
        id=preset,
        label=str(meta.get("label") or preset.value.title()),
        emoji=str(meta.get("emoji") or ""),
        description=str(meta.get("description") or ""),
        instruction=body,
    )


@lru_cache(maxsize=1)
def _presets() -> dict[AnswerPreset, Preset]:
    """Every preset, read once per process.

    Cached because this is on the path of every agent compile, and because a
    prompt changing under a running process would mean two questions in one
    conversation answered under different instructions. Restart to pick up an
    edit; `_presets.cache_clear()` is the test seam.
    """
    directory = prompt_dir()
    if not directory.is_dir():
        raise ConfigError(
            f"Prompt directory not found: {directory}. It ships in the repo as "
            "`prompts/`; set GRAPHRAG_PROMPT_DIR to point at it."
        )
    return {preset: _load(preset, directory) for preset in _ORDER}


def canonical_preset(preset: str | AnswerPreset | None) -> AnswerPreset:
    """Clamp anything — including a raw request string — to a known preset.

    Unknown, empty and None all resolve to GENERAL rather than raising: a
    request naming a preset this build does not have should get the neutral
    behaviour, not a 500. Callers keying a cache on the preset must go through
    here, or request input could mint unbounded distinct keys.
    """
    if preset is None:
        return AnswerPreset.GENERAL
    try:
        return AnswerPreset(preset)
    except ValueError:
        return AnswerPreset.GENERAL


def preset_options() -> list[Preset]:
    """Every preset, in picker order. The UI renders this rather than its own
    copy of the list, so adding one is a file plus an enum member."""
    return list(_presets().values())


def answer_instruction(
    preset: str | AnswerPreset | None, style: str | AnswerStyle | None = None
) -> str:
    """The body of the prompt's `# Answer style` section.

    A job preset *replaces* the style instruction rather than stacking on top of
    it — the two would otherwise contradict each other on exactly the axis they
    both control. "Thorough, well-structured, explain the reasoning" and "tight
    bullets, front-loaded" cannot both be obeyed, and a model handed both picks
    one at random.

    GENERAL is the exception and the reason this composes cleanly: its body ends
    on a `## Length and register` heading with nothing under it, and the style
    instruction is what goes there. So style keeps meaning what it always meant
    for callers that never send a preset.
    """
    chosen = _presets()[canonical_preset(preset)]
    if chosen.id is not AnswerPreset.GENERAL:
        return chosen.instruction
    return chosen.instruction + style_instruction(style or AnswerStyle.DETAILED) + "\n"


__all__ = [
    "AnswerPreset",
    "Preset",
    "answer_instruction",
    "canonical_preset",
    "preset_options",
    "prompt_dir",
]
