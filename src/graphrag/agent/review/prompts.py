"""Prompts for the review loop.

Both are deliberately small and tool-free. The critic never sees a raw chunk —
only the question, the draft, and the *names* of what was retrieved — which is
what keeps a review call around 1.5k tokens instead of re-paying for the whole
retrieval context.
"""

from __future__ import annotations

CRITIC_PROMPT = """\
You are reviewing a draft answer produced from a private document collection.
You are a reviewer, not an author: do not answer the question yourself.

Judge two things, and nothing else:

1. COMPLETENESS — does the draft address every part of the question that the
   retrieved sources could support? A question with several parts needs every
   part answered or explicitly marked as not found.
2. SUPPORT — does every claim carry a citation to one of the available sources?
   You cannot read the source text, so judge only what is visible: claims with
   no citation at all, and citations naming sources that are not in the list.

Then choose ONE action:
- "ship"          the draft is good enough to send.
- "revise"        the wording needs fixing, but the evidence is already there —
                  uncited claims, a citation naming an unavailable source, or a
                  missing statement that part of the question was not covered.
- "retrieve_more" the evidence itself is missing, and searching again with a
                  wider net could plausibly find it.

Prefer "ship". Choose "retrieve_more" only when you can name what is missing —
it costs another full retrieval pass, so a vague suspicion is not enough.
A draft that correctly refuses because the collection does not cover the
question is a good answer: ship it.

QUESTION
{question}

DRAFT
{draft}

AVAILABLE SOURCES (names only)
{available}

AUTOMATED CHECKS
{report}

Reply with ONLY a JSON object, no prose and no code fence:
{{"action": "ship|revise|retrieve_more",
  "complete": true|false,
  "missing": "what the draft does not cover, empty string if nothing",
  "unsupported": ["claims lacking support"],
  "reason": "one short sentence"}}
"""

REVISE_PROMPT = """\
Fix the draft below. Do not research, do not add facts, and do not answer
anything the draft does not already answer — you have no access to the sources,
so anything you add would be unsupported.

Make only these corrections:
- Remove or replace any [source: ...] tag naming a source not in the allowed
  list. If a claim has no valid source, keep the claim but drop the false tag
  and say plainly that it is not supported by the collection.
- Add citations to substantive claims that have none, using ONLY the allowed
  list, and only where the claim clearly came from that source.
- If something was flagged as not covered, state that plainly in one sentence
  rather than implying the answer is complete.

Keep the wording, structure and length otherwise unchanged.

ALLOWED SOURCES
{available}

PROBLEMS FOUND
{problems}

DRAFT
{draft}

Reply with the corrected answer only — no preamble, no explanation of what you
changed.
"""
