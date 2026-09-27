"""Every send attempt is counted by outcome.

Signup answers "code sent" whether or not the provider accepted the message, so
a revoked key or a lost domain verification looks, from outside, exactly like
nobody signing up. `graphrag_email_send_total` is the signal that tells the two
apart.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from prometheus_client import REGISTRY

from graphrag.accounts import emails


def _count(provider: str, outcome: str) -> float:
    return REGISTRY.get_sample_value(
        "graphrag_email_send_total", {"provider": provider, "outcome": outcome}
    ) or 0.0


class _Client:
    """Stands in for httpx.AsyncClient; answers every POST with one response."""

    def __init__(self, response):
        self._response = response

    def __call__(self, *args, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, *args, **kwargs):
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


@pytest.mark.parametrize(
    ("sender", "provider"),
    [
        (emails.ResendSender("k", "App <a@example.com>"), "resend"),
        (emails.BrevoSender("k", "App <a@example.com>"), "brevo"),
    ],
)
@pytest.mark.parametrize(
    ("response", "outcome", "ok"),
    [
        (httpx.Response(200, json={"id": "1"}), "sent", True),
        (httpx.Response(401, text="invalid api key"), "rejected", False),
        (httpx.ConnectError("unreachable"), "error", False),
    ],
)
def test_provider_sends_are_counted(monkeypatch, sender, provider, response, outcome, ok):
    monkeypatch.setattr(emails.httpx, "AsyncClient", _Client(response))
    before = _count(provider, outcome)
    assert asyncio.run(sender.send("to@example.com", "s", "t")) is ok
    assert _count(provider, outcome) == before + 1


def test_console_backend_is_counted_as_not_delivered():
    before = _count("console", "console")
    assert asyncio.run(emails.ConsoleSender().send("to@example.com", "s", "t")) is True
    assert _count("console", "console") == before + 1
