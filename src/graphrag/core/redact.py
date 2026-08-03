"""Scrub credentials out of text that is about to leave the process.

Provider SDKs put the request URL in their error messages, and several
providers carry the API key in that URL as a query parameter — so an error
string echoed back to a client, or written into a job record, can hand out a
working credential. Errors are also the text most likely to be pasted into a
chat or a bug report, which is where a key travels furthest.

This is a last line of defence, not a licence to pass secrets around: the right
fix is never to put them in the message. It exists because the messages are
written by libraries we don't control.
"""

from __future__ import annotations

import re

PLACEHOLDER = "[redacted]"

_PATTERNS = (
    # ?key=... / &api_key=... / &token=... in a URL (Google's client does this)
    re.compile(r"([?&](?:key|api[-_]?key|access[-_]?token|token)=)[^&\s\"']+", re.I),
    # Authorization headers echoed into a message
    re.compile(r"((?:authorization|x-api-key)\s*[:=]\s*)(?:bearer\s+)?\S+", re.I),
    # Vendor-prefixed keys, which are recognisable on sight and worth catching
    # even when they appear bare: OpenAI/DeepSeek sk-, Anthropic sk-ant-,
    # Google AIza, Resend re_, this project's own grk_.
    re.compile(r"\b(?:sk-ant-|sk-|AIza|re_|grk_)[A-Za-z0-9_\-]{12,}"),
    # Long opaque bearer-ish blobs sitting after a "key"/"token" word
    re.compile(r"\b((?:api[-_]?key|token|secret|password)\W{1,3})[A-Za-z0-9_\-]{16,}", re.I),
)


def redact_secrets(text: str) -> str:
    """Replace anything key-shaped with a placeholder."""
    if not text:
        return ""
    out = text
    for pattern in _PATTERNS:
        out = pattern.sub(
            lambda m: (m.group(1) + PLACEHOLDER) if m.groups() else PLACEHOLDER, out
        )
    return out


def safe_detail(exc: BaseException, limit: int = 300) -> str:
    """A client-safe one-liner for an exception: scrubbed, flattened, truncated.

    Truncation is deliberate as well as cosmetic — a long provider payload can
    carry request ids, internal hostnames and prompt fragments that the person
    who triggered the job has no reason to receive.
    """
    text = redact_secrets(" ".join(str(exc).split()))
    if len(text) > limit:
        text = text[:limit].rstrip() + "…"
    return text or type(exc).__name__
