"""Citation validation — the zero-token half of answer review.

The load-bearing property is the one these pin hardest: a name the retriever
never returned must read as fabricated, and a name it *did* return must never
read as fabricated no matter what characters it contains. Both sides normalize
through `tools._src`, so the tests exercise the awkward names on purpose.
"""

import pytest

from graphrag.agent.prompts import CLOSED_DOMAIN_REFUSAL
from graphrag.agent.review.citations import (
    available_sources,
    extract_citations,
    is_refusal,
    strip_fabricated,
    verify_citations,
)
from graphrag.agent.tools import (
    COMMUNITY_LABEL,
    GRAPH_LABEL,
    SourceSink,
    _src,
    build_tools,
    collect_sources,
)
from graphrag.api.routers.query import _response, _review_citations
from graphrag.core.types import QueryResult, RetrievedChunk


def _chunks(*sources: str) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=f"c{i}", text="body", source=s, score=1.0, retriever="test"
        )
        for i, s in enumerate(sources)
    ]


def _sink(*sources: str, labels: tuple[str, ...] = ()) -> SourceSink:
    sink = SourceSink()
    sink.add(_chunks(*sources))
    for label in labels:
        sink.note_label(label)
    return sink


def _avail(*sources: str, labels: tuple[str, ...] = ()) -> set[str]:
    """What a draft is allowed to cite, in the shape `verify_citations` takes."""
    return available_sources(_chunks(*sources), labels)


# ── parsing ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("draft", "expected"),
    [
        ("[source: a.pdf]", {"a.pdf"}),
        ("[source:a.pdf]", {"a.pdf"}),
        ("[source:   a.pdf  ]", {"a.pdf"}),
        ("[source: a.pdf][source: b.pdf]", {"a.pdf", "b.pdf"}),
        ("text [source: a.pdf] more [source: a.pdf]", {"a.pdf"}),
        ("no tags here", set()),
        ("", set()),
        ("[source: naïve—report (2024).pdf]", {"naïve—report (2024).pdf"}),
    ],
)
def test_tag_parsing(draft, expected):
    assert extract_citations(draft) == expected


def test_a_tag_cannot_swallow_a_paragraph():
    """The pattern is newline-bounded, so an unclosed bracket stays local."""
    assert extract_citations("[source: a.pdf\nrest of the answer") == set()


def test_quoted_source_names_survive_the_round_trip():
    """`_src` rewrites double quotes. Normalizing only one side would make a
    correctly-cited document read as fabricated."""
    name = 'the "final" report.pdf'
    written = _src(name)
    report = verify_citations(f"Answer. [source: {written}]", _avail(name))
    assert not report.fabricated
    assert report.cited == frozenset({written})


def test_overlong_source_names_survive_the_round_trip():
    name = "x" * 400 + ".pdf"
    report = verify_citations(f"Answer. [source: {_src(name)}]", _avail(name))
    assert not report.fabricated


def test_src_is_idempotent():
    """Relied on when re-normalizing a tag that `_src` already wrote."""
    for name in ('a "b" c.pdf', "x" * 400, "line\nbreak.pdf", "plain.pdf"):
        assert _src(_src(name)) == _src(name)


# ── verdicts ───────────────────────────────────────────────────────────────


def test_citing_a_document_that_was_never_retrieved_is_fabricated():
    report = verify_citations("Per the memo. [source: secret.pdf]", _avail("report.pdf"))
    assert report.fabricated == frozenset({"secret.pdf"})
    assert "fabricated" in report.findings
    assert not report.clean


def test_citing_a_retrieved_document_is_clean():
    report = verify_citations("Per the memo. [source: report.pdf]", _avail("report.pdf"))
    assert report.clean
    assert not report.fabricated
    assert not report.findings


def test_retrieved_but_uncited_is_not_an_error():
    """Retrieval over-fetches by design — surfacing more than the answer used
    is correct behaviour, so it must not gate anything."""
    report = verify_citations("Short answer. [source: a.pdf]", _avail("a.pdf", "b.pdf"))
    assert report.unused == frozenset({"b.pdf"})
    assert report.clean


def test_a_long_answer_with_no_citation_at_all_is_flagged():
    report = verify_citations("word " * 200, _avail("a.pdf"))
    assert report.uncited
    assert not report.clean


def test_a_short_answer_with_no_citation_is_not_flagged():
    """Acknowledgements and one-line clarifications legitimately cite nothing."""
    report = verify_citations("Yes, that is correct.", _avail("a.pdf"))
    assert not report.uncited
    assert report.clean


# ── graph labels (the prerequisite for everything above) ───────────────────


def test_graph_labels_are_citable():
    report = verify_citations(
        f"They are connected. [source: {GRAPH_LABEL}]",
        _avail("a.pdf", labels=(GRAPH_LABEL,)),
    )
    assert report.clean
    assert not report.fabricated


def test_graph_label_is_fabricated_when_no_graph_tool_ran():
    report = verify_citations(f"Claim. [source: {GRAPH_LABEL}]", _avail("a.pdf"))
    assert report.fabricated == frozenset({GRAPH_LABEL})


@pytest.mark.parametrize(
    ("tool", "arg", "label"),
    [
        ("graph_neighbors", "Acme", GRAPH_LABEL),
        ("get_entity", "Acme", GRAPH_LABEL),
        ("expand_subgraph", "Acme", GRAPH_LABEL),
        ("global_search", "themes?", COMMUNITY_LABEL),
    ],
)
def test_graph_tools_record_their_label(tool, arg, label, monkeypatch):
    """Without this the validator flags every graph-grounded answer, which is
    why it had to land before the validator could ship."""
    monkeypatch.setattr(
        "graphrag.ingestion.enrich.global_search", lambda *a, **k: "summary text"
    )

    class _Graph:
        def neighbors(self, *a, **k):
            return "Acme -> Robotics"

        def get_entity(self, *a, **k):
            return {"name": "Acme", "type": "Org", "description": "d", "connected": []}

        def chunks_for_entities(self, *a, **k):
            return []

    class _Ctx:
        vector = hybrid = embedder = None
        graph = _Graph()
        top_k = 8
        graph_hops = 2

    fn = {t.name: t for t in build_tools(_Ctx())}[tool].func
    with collect_sources() as sink:
        out = fn(arg)
        assert f"[source: {label}]" in out
        assert label in sink.labels
        assert label in available_sources(sink.chunks, sink.labels)


# ── refusal ────────────────────────────────────────────────────────────────


def test_exact_refusal_is_detected():
    assert is_refusal(CLOSED_DOMAIN_REFUSAL)


def test_reflowed_refusal_is_detected():
    """Models rewrap the refusal across lines. Only whitespace changes, so the
    collapsed comparison still recognizes it."""
    assert is_refusal(CLOSED_DOMAIN_REFUSAL.replace(" ", "\n  "))


def test_refusal_with_trailing_text_is_detected():
    """Containment rather than equality: some models append a stray line."""
    assert is_refusal(CLOSED_DOMAIN_REFUSAL + "\n\nLet me know if that helps.")


def test_a_reworded_refusal_is_not_detected():
    """The prompt asks for the refusal verbatim. Anything genuinely reworded is
    a normal answer as far as this check is concerned — it is a string match,
    not a classifier, and pretending otherwise would hide false negatives."""
    assert not is_refusal("I couldn't find that in the knowledge base, sorry.")


def test_refusal_short_circuits_the_report():
    report = verify_citations(CLOSED_DOMAIN_REFUSAL, _avail("a.pdf"))
    assert report.refusal
    assert report.findings == ("refusal",)
    assert not report.uncited  # a refusal citing nothing is correct


def test_a_normal_answer_is_not_a_refusal():
    assert not is_refusal("The report covers Q3 revenue. [source: a.pdf]")
    assert not is_refusal("")


# ── repair of last resort ──────────────────────────────────────────────────


def test_strip_fabricated_removes_only_the_bad_tags():
    text = "A [source: real.pdf] and B [source: fake.pdf] end."
    out = strip_fabricated(text, frozenset({"fake.pdf"}))
    assert "[source: real.pdf]" in out
    assert "fake.pdf" not in out


def test_strip_fabricated_is_a_no_op_when_nothing_is_fabricated():
    text = "A [source: real.pdf] end."
    assert strip_fabricated(text, frozenset()) == text


# ── what the API does with the report ──────────────────────────────────────


def _result(answer: str, *sources: str, labels: tuple[str, ...] = ()):
    return QueryResult(
        answer=answer, sources=_chunks(*sources), source_labels=list(labels)
    )


def test_a_model_refusal_ships_no_sources():
    """The live bug this closes: the model refuses precisely when nothing
    retrieved covers the question, but the rejected chunks were still returned —
    HTTP 200 with a populated `sources` array behind an "I couldn't find it"."""
    sources, report = _review_citations(
        CLOSED_DOMAIN_REFUSAL, _result(CLOSED_DOMAIN_REFUSAL, "a.pdf", "b.pdf")
    )
    assert sources == []
    assert report.refusal


def test_cited_sources_sort_ahead_of_merely_retrieved_ones():
    answer = "Per the memo. [source: b.pdf]"
    sources, report = _review_citations(answer, _result(answer, "a.pdf", "b.pdf", "c.pdf"))
    assert [c.source for c in sources] == ["b.pdf", "a.pdf", "c.pdf"]
    assert report.cited == frozenset({"b.pdf"})


def test_ordering_is_stable_when_nothing_was_cited():
    answer = "word " * 200
    sources, _ = _review_citations(answer, _result(answer, "a.pdf", "b.pdf"))
    assert [c.source for c in sources] == ["a.pdf", "b.pdf"]


def test_graph_labels_reach_the_api_check():
    """`source_labels` has to travel on QueryResult, or an answer grounded in
    the graph reads as fabricated the moment it leaves the agent."""
    answer = f"They are linked. [source: {GRAPH_LABEL}]"
    _, report = _review_citations(answer, _result(answer, "a.pdf", labels=(GRAPH_LABEL,)))
    assert not report.fabricated


def test_response_marks_the_cited_flag():
    answer = "Per the memo. [source: b.pdf]"
    sources, report = _review_citations(answer, _result(answer, "a.pdf", "b.pdf"))
    resp = _response(answer, sources, [], cited=report.cited)
    assert {s.source: s.cited for s in resp.sources} == {"b.pdf": True, "a.pdf": False}
