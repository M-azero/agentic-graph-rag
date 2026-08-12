"""Conversations, stored server-side.

Chat history used to live in the browser's localStorage, which meant it was
lost on a cache clear and invisible from another device — while the agent's own
memory sat on the server under a matching id. These endpoints make the
transcript the server's too, so the two halves of a conversation stop drifting
apart.

A thread id is a UUID that belongs to a user. Every read and write checks
ownership, so an id guessed or copied from elsewhere reaches nothing.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select

from graphrag.api.deps import AuthUser, get_current_user, get_db
from graphrag.api.schemas import (
    Acknowledged,
    MessageInfo,
    ThreadCreate,
    ThreadInfo,
    ThreadList,
    ThreadMessages,
    ThreadUpdate,
)
from graphrag.db.engine import session_scope
from graphrag.db.models import Message, Thread
from graphrag.limits import LimitService, get_limits, reject_with
from graphrag.limits.service import LimitBreach
from graphrag.shelves import shelf_for_request

router = APIRouter(prefix="/threads", tags=["threads"])


def _require_db(db):
    if db is None:
        raise HTTPException(
            status_code=503, detail="Chat history needs a database. Set GRAPHRAG_DATABASE_URL."
        )
    return db


def _user_uuid(user: AuthUser) -> uuid.UUID:
    try:
        return uuid.UUID(str(user.user_id))
    except (ValueError, AttributeError, TypeError):
        # Dev-mode identities aren't account rows, so there is nothing to own.
        raise HTTPException(
            status_code=400, detail="Server-side threads require a real account."
        ) from None


def _shape(thread: Thread) -> ThreadInfo:
    return ThreadInfo(
        id=str(thread.id),
        title=thread.title,
        created_at=thread.created_at.isoformat() if thread.created_at else "",
        updated_at=thread.updated_at.isoformat() if thread.updated_at else "",
        shelf_id=str(thread.shelf_id) if thread.shelf_id else None,
    )


async def _owned(s, thread_id: str, owner: uuid.UUID) -> Thread:
    try:
        tid = uuid.UUID(thread_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="No such conversation.") from None
    thread = (
        await s.execute(
            select(Thread).where(
                Thread.id == tid, Thread.user_id == owner, Thread.deleted_at.is_(None)
            )
        )
    ).scalar_one_or_none()
    if thread is None:
        # 404 rather than 403: a thread the caller doesn't own should be
        # indistinguishable from one that was never there.
        raise HTTPException(status_code=404, detail="No such conversation.")
    return thread


@router.get("", response_model=ThreadList)
async def list_threads(
    user: AuthUser = Depends(get_current_user),
    db=Depends(get_db),
) -> ThreadList:
    async with session_scope(_require_db(db)) as s:
        rows = (
            await s.execute(
                select(Thread)
                .where(Thread.user_id == _user_uuid(user), Thread.deleted_at.is_(None))
                .order_by(Thread.updated_at.desc())
                .limit(200)
            )
        ).scalars().all()
        return ThreadList(threads=[_shape(t) for t in rows])


@router.post("", response_model=ThreadInfo)
async def create_thread(
    payload: ThreadCreate,
    user: AuthUser = Depends(get_current_user),
    db=Depends(get_db),
    limits: LimitService = Depends(get_limits),
) -> ThreadInfo:
    owner = _user_uuid(user)
    effective = await limits.effective(user.user_id)
    # Resolved (and ownership-checked) before the transaction: `resolve` opens
    # its own session, and nesting one inside this one would deadlock on the
    # same connection.
    shelf = await shelf_for_request(db, user, payload.shelf_id)
    async with session_scope(_require_db(db)) as s:
        live = (
            await s.execute(
                select(func.count())
                .select_from(Thread)
                .where(Thread.user_id == owner, Thread.deleted_at.is_(None))
            )
        ).scalar_one()
        if live >= effective.max_threads:
            raise reject_with(
                LimitBreach("max_threads", int(live), effective.max_threads)
            )

        thread = Thread(
            user_id=owner,
            title=(payload.title or "New chat")[:120],
            # Pinned here and never changed. The agent's memory for this thread
            # is keyed on the shelf's corpus, so a conversation that moved
            # shelves would lose its history and start citing passages its own
            # transcript never mentioned.
            shelf_id=shelf.id,
        )
        s.add(thread)
        await s.flush()
        return _shape(thread)


@router.patch("/{thread_id}", response_model=ThreadInfo)
async def rename_thread(
    thread_id: str,
    payload: ThreadUpdate,
    user: AuthUser = Depends(get_current_user),
    db=Depends(get_db),
) -> ThreadInfo:
    async with session_scope(_require_db(db)) as s:
        thread = await _owned(s, thread_id, _user_uuid(user))
        if payload.title is not None:
            thread.title = payload.title[:120] or "New chat"
        return _shape(thread)


@router.delete("/{thread_id}", response_model=Acknowledged)
async def delete_thread(
    thread_id: str,
    user: AuthUser = Depends(get_current_user),
    db=Depends(get_db),
) -> Acknowledged:
    """Soft-delete the transcript and drop the agent's memory of it."""
    async with session_scope(_require_db(db)) as s:
        thread = await _owned(s, thread_id, _user_uuid(user))
        thread.deleted_at = func.now()
    return Acknowledged(message="Conversation deleted.")


@router.get("/{thread_id}/messages", response_model=ThreadMessages)
async def thread_messages(
    thread_id: str,
    user: AuthUser = Depends(get_current_user),
    db=Depends(get_db),
) -> ThreadMessages:
    async with session_scope(_require_db(db)) as s:
        thread = await _owned(s, thread_id, _user_uuid(user))
        rows = (
            await s.execute(
                select(Message).where(Message.thread_id == thread.id).order_by(Message.id)
            )
        ).scalars().all()
        return ThreadMessages(
            thread=_shape(thread),
            messages=[
                MessageInfo(
                    id=m.id,
                    role=m.role,
                    content=m.content,
                    sources=m.sources or [],
                    model=m.model or "",
                    created_at=m.created_at.isoformat() if m.created_at else "",
                )
                for m in rows
            ],
        )
