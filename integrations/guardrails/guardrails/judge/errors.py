"""Typed judge errors.

Each carries a ``kind`` that maps 1:1 onto ``JudgeInfo.error`` and drives the pipeline's
``fail_mode`` handling. Nothing here ever reaches the client as a 5xx — the pipeline
converts these into a verdict.
"""

from __future__ import annotations

from typing import Literal

JudgeErrorKind = Literal["timeout", "api_error", "parse_error", "refusal"]


class JudgeError(Exception):
    """Base class for all judge failures.

    ``provider``/``model`` are stamped on by the failover chain when it gives
    up, naming the link that produced the final failure. Without them a chain
    reports every outage against its primary, which is the one link you can be
    sure was not serving.
    """

    kind: JudgeErrorKind = "api_error"
    provider: str | None = None
    model: str | None = None


class JudgeTimeout(JudgeError):
    kind = "timeout"


class JudgeAPIError(JudgeError):
    kind = "api_error"


class JudgeParseError(JudgeError):
    kind = "parse_error"


class JudgeRefusal(JudgeError):
    kind = "refusal"
