"""The usage kinds the admin console charts actually get written.

`MESSAGE`, `UPLOAD` and `INGEST_CHUNKS` were defined in `usage/recorder.py`,
exported from `usage/__init__.py`, and recorded by nothing. Only the token kinds
ever reached `usage_events`. So `/admin/usage` plotted a flat zero for messages
and uploads, and every row of `/admin/users` reported `messages_30d: 0` however
much the account had chatted — a dashboard that is confidently wrong rather than
visibly broken.

These assert the call is made. The admin queries that read the rows back are
covered by the integration suite.
"""

from __future__ import annotations

import pytest

from graphrag.usage.recorder import (
    INGEST_CHUNKS,
    MESSAGE,
    TOKENS_IN,
    TOKENS_OUT,
    UPLOAD,
    UsageRecorder,
)


class _SpyRecorder(UsageRecorder):
    """Records the calls without needing a database."""

    def __init__(self) -> None:
        super().__init__(None, None)
        self.calls: list[tuple[str, str, int]] = []

    async def record(self, user_id, kind, amount=1, meta=None):
        self.calls.append((user_id, kind, amount))


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _kinds(spy: _SpyRecorder) -> list[str]:
    return [kind for _user, kind, _amount in spy.calls]


@pytest.mark.anyio
async def test_a_chat_request_records_a_message_event():
    """`enforce_message_limits` is the one place every chat route passes
    through, which is why the counter belongs there rather than per-endpoint."""
    from types import SimpleNamespace

    from graphrag.api.deps import AuthUser
    from graphrag.limits.deps import enforce_message_limits
    from graphrag.limits.service import LimitService

    spy = _SpyRecorder()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(usage=spy)))
    user = AuthUser("user-1", "tenant-1", "user", "u@example.com")

    returned = await enforce_message_limits(request, user, LimitService(None, None))

    assert returned is user
    assert _kinds(spy) == [MESSAGE]
    assert spy.calls[0][0] == "user-1"


@pytest.mark.anyio
async def test_a_quota_breach_records_nothing():
    """A refused request never ran, so it must not show up as a message."""
    from types import SimpleNamespace

    from fastapi import HTTPException

    from graphrag.api.deps import AuthUser
    from graphrag.limits.deps import enforce_message_limits
    from graphrag.limits.service import LimitBreach, LimitService

    class _Full(LimitService):
        def check_messages(self, user_id, limits):
            return LimitBreach("messages_per_day", 100, 100, 60)

    spy = _SpyRecorder()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(usage=spy)))
    user = AuthUser("user-1", "tenant-1")

    with pytest.raises(HTTPException) as caught:
        await enforce_message_limits(request, user, _Full(None, None))

    assert caught.value.status_code == 429
    assert spy.calls == []


@pytest.mark.anyio
async def test_an_ingest_records_the_chunks_it_produced():
    from graphrag.api.routers.ingest import _record_chunks
    from graphrag.jobs import JobStatus

    spy = _SpyRecorder()
    await _record_chunks(spy, "user-1", JobStatus("j1", status="done", chunks=42))
    assert spy.calls == [("user-1", INGEST_CHUNKS, 42)]


@pytest.mark.anyio
async def test_a_failed_ingest_records_no_chunks():
    from graphrag.api.routers.ingest import _record_chunks
    from graphrag.jobs import JobStatus

    spy = _SpyRecorder()
    await _record_chunks(spy, "user-1", JobStatus("j1", status="error", chunks=0))
    await _record_chunks(spy, "user-1", None)
    await _record_chunks(None, "user-1", JobStatus("j1", status="done", chunks=5))
    assert spy.calls == []


def test_every_declared_kind_has_a_writer():
    """The guard against this whole class of bug: a constant that nothing
    records is a chart that silently reads zero."""
    import pathlib

    src = pathlib.Path("src/graphrag")
    body = "\n".join(
        p.read_text(encoding="utf-8")
        for p in src.rglob("*.py")
        if p.name != "recorder.py"
    )
    for kind in (MESSAGE, UPLOAD, INGEST_CHUNKS, TOKENS_IN, TOKENS_OUT):
        constant = {
            MESSAGE: "MESSAGE", UPLOAD: "UPLOAD", INGEST_CHUNKS: "INGEST_CHUNKS",
            TOKENS_IN: "TOKENS_IN", TOKENS_OUT: "TOKENS_OUT",
        }[kind]
        assert f"{constant}," in body or f"{constant})" in body, (
            f"usage kind {constant} is declared but never recorded"
        )
