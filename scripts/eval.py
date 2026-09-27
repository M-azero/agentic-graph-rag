#!/usr/bin/env python
"""Score retrieval, the closed-domain gate and answers against data/eval/qa.yaml.

    python scripts/eval.py               # all three bands
    python scripts/eval.py --no-answers  # gate + retrieval only (no LLM calls, fast)
    python scripts/eval.py --judge       # + LLM-judged faithfulness (in_corpus)
    python scripts/eval.py --guard       # + the guardrails output check on answers
    python scripts/eval.py --band near_miss     # one band only
    python scripts/eval.py --strict      # non-zero exit if any check fails (CI)

Needs the stack (Neo4j, and the model backends the profile selects) running.
Ingests data/sample.md first, so it is self-contained on a fresh database.
`--guard` additionally needs the guardrails service up.

Three bands, because two are not enough. `in_corpus` and `off_topic` sit far
apart on the reranker's scale and stop being informative as soon as
`retrieval.min_relevance` is roughly right. `near_miss` — corpus vocabulary,
absent answer — is the band that decides whether the system invents things, and
it is the one a two-band calibration never sees.

What each band asserts:
  in_corpus  the gate must PASS. A refusal here is a false refusal: a question
             the corpus answers, turned away as off-topic.
  off_topic  the gate should REFUSE, and every question that gets past it is
             counted as a gate bypass. That alone is not scored as a failure:
             the agent is closed-domain as well, and a suspended gate (an
             uncalibrated reranker) hands it these routinely. It fails when
             BOTH lines miss — the gate let it through and the agent answered
             it instead of refusing.
  near_miss  either outcome is legitimate. If the gate refuses, good. If it
             passes, the answer must ADMIT the fact is missing rather than
             fabricate it — that admission is the only thing standing between a
             plausible question and a confident lie.

The point of this file: every retrieval change is a guess until it moves these
numbers. Run it before and after.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphrag.agent.review.citations import is_refusal  # noqa: E402

# The production gate predicate itself, not a copy of it. If the eval decided
# "calibrated" by its own rule, a change to the real one would silently stop
# being measured here — which is the exact failure this band structure exists
# to prevent.
from graphrag.api.routers.query import _gate_applies  # noqa: E402
from graphrag.container import Container  # noqa: E402
from graphrag.pipelines import IngestPipeline, QueryService  # noqa: E402
from graphrag.retrieval.reranker import RERANKED_BY  # noqa: E402

BANDS = ("in_corpus", "near_miss", "off_topic")

_JUDGE_PROMPT = (
    "Question: {question}\n\nRetrieved evidence:\n{evidence}\n\n"
    "Proposed answer:\n{answer}\n\n"
    "Does the answer follow from the evidence, without inventing facts? "
    "Reply with exactly YES or NO."
)

# An answer that admits the corpus does not cover something. The agent emits
# CLOSED_DOMAIN_REFUSAL verbatim when it refuses outright (that is what
# `is_refusal` catches), but a near_miss question usually gets a *partial*
# answer instead — "the corpus says X, but never states Y" — which is the
# correct behaviour and matches no fixed string. Hence phrase matching.
_ABSENCE_MARKERS = (
    "not in the knowledge base", "not in the corpus", "no document",
    "do not name", "does not name", "don't name", "doesn't name", "never name",
    "do not say", "does not say", "doesn't say", "never say",
    "not specify", "does not specify", "doesn't specify",
    "not stated", "not state", "not mention", "no mention", "never mention",
    "could not find", "couldn't find", "not present", "not available",
    "not contain", "does not cover", "doesn't cover", "no information",
)


def admits_absence(text: str) -> bool:
    """Whether the answer concedes the corpus does not hold the fact asked for."""
    if not text:
        return False
    low = " ".join(text.split()).lower()
    return is_refusal(text) or any(m in low for m in _ABSENCE_MARKERS)


def gate(chunks, min_relevance: float) -> tuple[float | None, bool, bool]:
    """Replay the closed-domain gate on a probe result.

    Returns (top_score, applies, passed) using the same comparison the query
    router makes, so a number printed here is the number that decides a real
    request.
    """
    if not chunks:
        return None, False, False
    if min_relevance <= 0:                      # gate disabled by config
        return chunks[0].score, False, True
    applies = _gate_applies(chunks)
    passed = not (applies and chunks[0].score < min_relevance)
    return chunks[0].score, applies, passed


def guard_output(base_url: str, api_key: str | None, question, answer, sources):
    """The guardrails output verdict plus the raw groundedness score.

    The app's own client keeps only the action, so the eval talks to the service
    directly — the number is the thing worth tracking release over release.
    """
    import httpx

    headers = {}
    if api_key:
        headers = {"Authorization": f"Bearer {api_key}", "X-API-Key": api_key}
    body = httpx.post(
        f"{base_url.rstrip('/')}/v1/guard/output",
        json={
            "input": question,
            "output": answer,
            "policy_id": "default",
            "context_docs": [
                {"id": c.chunk_id, "text": c.text[:2000], "source": c.source}
                for c in sources[:8]
            ],
        },
        headers=headers,
        timeout=60.0,
    )
    body.raise_for_status()
    data = body.json()
    return data.get("action"), (data.get("groundedness") or {}).get("score")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-answers", action="store_true", help="skip agent answers")
    parser.add_argument("--judge", action="store_true", help="LLM-judge faithfulness")
    parser.add_argument("--guard", action="store_true", help="run the guardrails output check")
    parser.add_argument("--band", choices=BANDS, help="score one band only")
    parser.add_argument("--strict", action="store_true", help="exit 1 on any failure")
    # NOT "default": under the DuckDB vector provider each user is one file held
    # under an exclusive lock, and a running API process owns `default`. Sharing
    # it makes `make eval` fail against exactly the live stack it needs.
    parser.add_argument("--user", default="eval")
    args = parser.parse_args()

    cases = yaml.safe_load((ROOT / "data" / "eval" / "qa.yaml").read_text(encoding="utf-8"))
    if args.band:
        cases = [c for c in cases if c.get("band", "in_corpus") == args.band]

    container = Container()
    service = QueryService(container)
    min_rel = container.settings.retrieval.min_relevance
    guard_url = container.secrets.guardrails_url or container.settings.safety.base_url
    guard_key = container.secrets.guardrails_api_key

    print("Ingesting data/sample.md (idempotent) ...")
    IngestPipeline(container).run(ROOT / "data" / "sample.md", user_id=args.user)
    print(f"retrieval.min_relevance = {min_rel}\n")

    retrieval_hits = answer_hits = faithful_hits = 0
    answers_run = judged = 0
    failures: list[str] = []
    stats = {b: {"n": 0, "passed": 0, "scores": [], "uncalibrated": 0,
                 "answered": 0, "honest": 0, "bypassed": 0}
             for b in BANDS}
    # Which reranker scored each question. A trial Cohere key rate-limits inside
    # a single run, the chain fails over to a generative 0-10 grader, and the
    # score ranges below then silently mix two scales — the 2026-09-02
    # calibration table carried a generative 0.0000 as a Cohere near_miss
    # minimum for exactly this reason. Recording it makes a mixed run visible.
    scored_by: dict[str, int] = {}

    for case in cases:
        question = case["question"]
        band = case.get("band", "in_corpus")
        keywords = [str(k) for k in case.get("expect_keywords", [])]
        forbidden = [str(k) for k in case.get("forbid_keywords", [])]
        source = str(case.get("expect_source", ""))
        st = stats[band]
        st["n"] += 1

        chunks = service.search(question, k=8, user_id=args.user)
        top, applies, passed = gate(chunks, min_rel)
        st["passed"] += passed
        by = (chunks[0].metadata.get(RERANKED_BY) or "not reranked") if chunks else "-"
        if chunks:
            scored_by[by] = scored_by.get(by, 0) + 1
        # Only scores on the gate's own scale go into the calibration ranges; a
        # generative grade or a raw retrieval value is counted, not averaged in.
        if top is not None and applies:
            st["scores"].append(top)
        elif top is not None:
            st["uncalibrated"] += 1

        shown = f"{top:.4f}" if top is not None else "none"
        line = f"  gate {shown} -> {'PASS' if passed else 'refuse'} · by {by}"
        if not applies and chunks:
            line += " (uncalibrated: gate suspended)"

        # Band expectations on the gate itself.
        if band == "in_corpus" and not passed:
            failures.append(f"FALSE REFUSAL - in-topic question refused: {question}")
        # An off-topic question reaching the agent is a gate bypass, and it is
        # counted as one below — but on its own it is not yet a wrong answer.
        # The agent is closed-domain too, and a suspended gate (an uncalibrated
        # reranker, which is normal on a rate-limited key) routinely hands it
        # questions the gate would otherwise have stopped; measured, it refuses
        # them. So the FAILURE is raised only where the second line also fails,
        # in the answer block. With --no-answers there is no second line to
        # check, so the gate verdict has to stand on its own.
        if band == "off_topic" and passed:
            st["bypassed"] += 1
            if args.no_answers:
                failures.append(
                    f"BYPASS - off-topic question passed the gate: {question}"
                )

        # Retrieval check is only meaningful where an expected source exists.
        if band == "in_corpus":
            got_source = any(source in c.source for c in chunks) if source else bool(chunks)
            retrieval_hits += got_source
            if not got_source:
                failures.append(f"retrieval missed '{source}' for: {question}")
            line += f" · [{'ok' if got_source else 'MISS'}] retrieval"

        # Only generate where production would: the gate refusing means no agent
        # run at all, and billing an eval for answers the app would never
        # produce would misreport both cost and behaviour.
        if not args.no_answers and passed:
            answers_run += 1
            st["answered"] += 1
            result = service.answer(question, user_id=args.user)
            answer = result.answer
            text = answer.lower()

            if band == "in_corpus":
                hit_kw = [k for k in keywords if k.lower() in text]
                ok = len(hit_kw) == len(keywords)
                answer_hits += ok
                if not ok:
                    missing = sorted(set(keywords) - set(hit_kw))
                    failures.append(f"answer missing {missing} for: {question}")
                line += f" · [{'ok' if ok else 'MISS'}] answer ({len(hit_kw)}/{len(keywords)} kw)"
            else:
                # near_miss and any off_topic bypass: the answer must not assert
                # a fact the corpus does not hold.
                honest = admits_absence(answer)
                st["honest"] += honest
                if not honest:
                    # Both lines are down: the gate let it through and the agent
                    # answered it anyway. For off_topic that is the real bypass.
                    label = "BYPASS" if band == "off_topic" else "FABRICATION RISK"
                    failures.append(
                        f"{label} - {band} answered without admitting the gap: {question}"
                    )
                line += f" · [{'ok' if honest else 'MISS'}] admits gap"

            said = [k for k in forbidden if k.lower() in text]
            if said:
                failures.append(f"FABRICATION - answer contains {said} for: {question}")

            if args.guard and answer.strip():
                action, ungrounded = guard_output(
                    guard_url, guard_key, question, answer, result.sources
                )
                line += f" · guard {action} (ungrounded {ungrounded})"
                # A blocked answer that correctly admits the gap is the guard
                # turning honesty into a refusal. Worth failing on: it is the
                # in-topic-becomes-off-topic failure, one layer down.
                if action == "block" and band != "in_corpus" and admits_absence(answer):
                    failures.append(
                        f"GUARD BLOCKED AN HONEST REFUSAL (ungrounded {ungrounded}): {question}"
                    )
                if action == "block" and band == "in_corpus":
                    failures.append(
                        f"GUARD BLOCKED A GOOD ANSWER (ungrounded {ungrounded}): {question}"
                    )

            if args.judge and band == "in_corpus" and answer.strip():
                judged += 1
                evidence = "\n---\n".join(c.text[:600] for c in result.sources[:6])
                verdict = container.llm.invoke(
                    _JUDGE_PROMPT.format(
                        question=question, evidence=evidence, answer=answer
                    )
                )
                faithful = str(verdict.content).strip().upper().startswith("YES")
                faithful_hits += faithful
                if not faithful:
                    failures.append(f"judge says unfaithful: {question}")
                line += f" · [{'ok' if faithful else 'MISS'}] faithful"

        print(f"\n[{band}] {question}\n{line}")

    print(f"\n{'=' * 72}")
    print(f"{'band':<11}{'n':>4}{'gate pass':>12}{'score range':>22}"
          f"{'uncal':>7}{'answered':>10}{'honest':>8}")
    for b in BANDS:
        st = stats[b]
        if not st["n"]:
            continue
        rng = f"{min(st['scores']):.4f} - {max(st['scores']):.4f}" if st["scores"] else "-"
        gate_pass = f"{st['passed']}/{st['n']}"
        honest = (f"{st['honest']}/{st['answered']}"
                  if b != "in_corpus" and st["answered"] else "-")
        print(f"{b:<11}{st['n']:>4}{gate_pass:>12}{rng:>22}{st['uncalibrated']:>7}"
              f"{st['answered']:>10}{honest:>8}")

    print("\nscored by: " + (", ".join(f"{k} x{v}" for k, v in sorted(scored_by.items()))
                             or "-"))
    if len(scored_by) > 1:
        print("  WARNING: more than one reranker scored this run. Score ranges use only "
              "the calibrated scores (column 'uncal' counts the rest), but gate verdicts "
              "for the uncalibrated questions are NOT what the primary would decide. "
              "Re-run when the primary is not rate-limited before calibrating on it.")
        failures.append(f"MIXED RERANKERS - {sorted(scored_by)}")

    if stats["in_corpus"]["n"]:
        miss = stats["in_corpus"]["n"] - stats["in_corpus"]["passed"]
        print(f"\nfalse refusals (in-topic refused) : {miss}")
    if stats["off_topic"]["n"]:
        by = stats["off_topic"]["bypassed"]
        caught = stats["off_topic"]["honest"]
        print(f"gate bypasses (off-topic reached the agent) : {by}"
              + (f", of which the agent refused {caught}" if by else ""))
    if stats["in_corpus"]["n"]:
        print(f"retrieval                         : {retrieval_hits}/{stats['in_corpus']['n']}")
    if stats["in_corpus"]["answered"]:
        print(f"in_corpus answers                 : {answer_hits}/{stats['in_corpus']['answered']}")
    if judged:
        print(f"faithful                          : {faithful_hits}/{judged}")

    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  - {f}")
    return 1 if (args.strict and failures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
