# Job presets

One file per preset. Each is the working method the assistant follows for a
whole class of documents — what a finance question needs surfaced is not what a
teaching question needs, even when the retrieval is identical.

The picker in the chat composer is built from this directory, and every shelf
stores the preset it opens with. Add a file here and it is offered; there is no
second list to update.

## The contract

A preset is interpolated into the `# Answer style` section of `SYSTEM_PROMPT`
(`src/graphrag/agent/prompts.py`), which is a **closed-domain, injection-hardened**
system message. Everything above it already establishes that the assistant
answers only from retrieved documents, cites every claim, and refuses when the
knowledge base does not cover the question.

So a preset has exactly one job: **narrow what counts as a good answer within
those rules.** It must never widen them.

Concretely, a preset may not:

- grant the model its own knowledge, arithmetic, or judgement,
- relax the citation requirement,
- offer any route around the closed-domain refusal,
- introduce a `{`/`}` placeholder — the prompt is rendered with `str.format`,
  and an unfilled brace reaches the model verbatim.

`tests/unit/test_presets.py` enforces all four against every file in this
directory. The first three are the tempting edit: telling the finance preset to
"calculate the ratio", or the writing preset to "fill in reasonable detail",
reads as helpful and quietly dismantles the guarantee the rest of the prompt
exists to make.

## File format

YAML frontmatter, then the instruction body as Markdown.

```markdown
---
id: finance          # must equal the filename stem and an AnswerPreset member
label: Finance       # shown in the picker
emoji: 💰            # shown in the picker
description: ...     # one line, shown under the picker
---

You are reading financial documents.

## Always carry
- ...
```

`id` is the wire value. It is stored on every shelf row, so **renaming a file is
a breaking change**: shelves pointing at the old id fall back to `general`
(`canonical_preset` clamps unknown values rather than raising, so nothing
breaks loudly — it just quietly stops being the preset the user chose).

The body is used verbatim. Its `##` headings nest under the prompt's own `#`
sections, which is why they are second-level.

## `general` is special

`general.md` is the neutral default and the compatibility path. Its body is
followed by the requested `AnswerStyle` instruction (concise / detailed /
technical / eli5), so a caller that names no preset — the CLI, an API key
holder, an older build of the UI — still gets style-controlled phrasing exactly
as it did before presets existed. Every other preset **replaces** the style,
because both control the same axis and "thorough, explain the reasoning"
alongside "tight bullets, front-loaded" is a contradiction the model resolves at
random.

## Cost

The body is part of the system prompt, and the agent's tool loop resends the
system prompt on every turn. A preset that is 200 tokens longer costs that on
each of ~3-6 turns per question. Say what changes the answer; delete what
merely sounds thorough.

## Where this is loaded from

`src/graphrag/agent/presets.py`, resolved the same way `configs/` is: a source
checkout finds this directory on its own, the Docker image pins
`GRAPHRAG_PROMPT_DIR=/app/prompts`, and the files are read once and cached.
