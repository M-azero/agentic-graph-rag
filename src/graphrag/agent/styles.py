"""Answer styles. The requested style is injected into the prompt so the same
retrieval can be phrased for different audiences."""

from __future__ import annotations

from graphrag.core.types import AnswerStyle

_STYLE_INSTRUCTIONS: dict[AnswerStyle, str] = {
    AnswerStyle.CONCISE: "Answer in 2-4 sentences. Lead with the direct answer. No preamble.",
    AnswerStyle.DETAILED: (
        "Give a thorough, well-structured answer with short paragraphs or bullet points. "
        "Explain the reasoning and connect related facts."
    ),
    AnswerStyle.TECHNICAL: (
        "Answer for an expert. Use precise terminology, include specifics (names, numbers, "
        "relationships), and don't over-explain basics."
    ),
    AnswerStyle.ELI5: (
        "Explain simply, as if to a curious beginner. Use plain words and a short analogy "
        "where it helps."
    ),
}


def canonical_style(style: str | AnswerStyle) -> AnswerStyle:
    """Clamp anything — including a raw request string — to a known style.

    Callers that key a cache on the style must use this rather than the raw
    value: "detailed", "banana" and "" all render the same prompt, so keying on
    the raw string would let request input mint unbounded distinct keys.
    """
    try:
        return AnswerStyle(style)
    except ValueError:
        return AnswerStyle.DETAILED


def style_instruction(style: str | AnswerStyle) -> str:
    return _STYLE_INSTRUCTIONS[canonical_style(style)]
