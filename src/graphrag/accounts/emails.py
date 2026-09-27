"""Transactional email — verification codes.

One small interface with three backends, chosen by config exactly like every
other provider in this codebase. Sending is best-effort by design: a signup
must not 500 because an email API had a bad minute, so failures are logged and
the caller is told to use "resend" rather than being shown a stack trace.
"""

from __future__ import annotations

import abc

import httpx
from prometheus_client import Counter

from graphrag.config.settings import Secrets, Settings
from graphrag.core.logging import get_logger

log = get_logger(__name__)

_TIMEOUT = 10.0

# A failed send is invisible from outside: signup answers "code sent" either way
# (by design — see the module docstring), so the user just never gets a code and
# the provider's dashboard shows nothing, because it never accepted the message.
# This counter is the only aggregate signal that sign-up has quietly stopped.
_SENDS = Counter(
    "graphrag_email_send_total",
    "Transactional email send attempts",
    ["provider", "outcome"],  # outcome: sent | rejected | error | console
)


class EmailSender(abc.ABC):
    @abc.abstractmethod
    async def send(self, to: str, subject: str, text: str) -> bool:
        """Deliver a message. Returns False instead of raising on failure."""


class ConsoleSender(EmailSender):
    """Development: log the message instead of sending it. The code is printed
    at WARNING so it stands out in a busy log — this is how you sign up locally
    without an email provider."""

    async def send(self, to: str, subject: str, text: str) -> bool:
        log.warning("email_not_sent", to=to, subject=subject, body=text,
                    hint="console email backend — configure RESEND_API_KEY to deliver")
        _SENDS.labels(provider="console", outcome="console").inc()
        return True


class ResendSender(EmailSender):
    def __init__(self, api_key: str, from_addr: str) -> None:
        self._key = api_key
        self._from = from_addr

    async def send(self, to: str, subject: str, text: str) -> bool:
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                r = await client.post(
                    "https://api.resend.com/emails",
                    headers={"Authorization": f"Bearer {self._key}"},
                    json={"from": self._from, "to": [to], "subject": subject, "text": text},
                )
            if r.status_code >= 400:
                log.warning("email_send_failed", provider="resend",
                            status=r.status_code, body=r.text[:200])
                _SENDS.labels(provider="resend", outcome="rejected").inc()
                return False
            _SENDS.labels(provider="resend", outcome="sent").inc()
            return True
        except Exception as exc:
            log.warning("email_send_error", provider="resend", error=str(exc))
            _SENDS.labels(provider="resend", outcome="error").inc()
            return False


class BrevoSender(EmailSender):
    def __init__(self, api_key: str, from_addr: str) -> None:
        self._key = api_key
        self._from = from_addr

    async def send(self, to: str, subject: str, text: str) -> bool:
        name, addr = _split_address(self._from)
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                r = await client.post(
                    "https://api.brevo.com/v3/smtp/email",
                    headers={"api-key": self._key, "content-type": "application/json"},
                    json={
                        "sender": {"name": name, "email": addr},
                        "to": [{"email": to}],
                        "subject": subject,
                        "textContent": text,
                    },
                )
            if r.status_code >= 400:
                log.warning("email_send_failed", provider="brevo",
                            status=r.status_code, body=r.text[:200])
                _SENDS.labels(provider="brevo", outcome="rejected").inc()
                return False
            _SENDS.labels(provider="brevo", outcome="sent").inc()
            return True
        except Exception as exc:
            log.warning("email_send_error", provider="brevo", error=str(exc))
            _SENDS.labels(provider="brevo", outcome="error").inc()
            return False


def _split_address(value: str) -> tuple[str, str]:
    """Parse `Name <addr@host>` into its parts; Brevo wants them separately."""
    if "<" in value and ">" in value:
        name, _, rest = value.partition("<")
        return name.strip().strip('"') or "GraphRAG", rest.partition(">")[0].strip()
    return "GraphRAG", value.strip()


def build_email_sender(settings: Settings, secrets: Secrets) -> EmailSender:
    provider = settings.auth.email.provider
    sender = settings.auth.email.from_addr or secrets.email_from
    if provider == "resend" and secrets.resend_api_key:
        return ResendSender(secrets.resend_api_key, sender)
    if provider == "brevo" and secrets.brevo_api_key:
        return BrevoSender(secrets.brevo_api_key, sender)
    if provider not in ("console", "none"):
        log.warning("email_provider_unconfigured", provider=provider,
                    fallback="console", hint="set the provider's API key")
    return ConsoleSender()
