"""The admin surface: users, limits, usage, per-tenant graph inspection.

Everything here is gated by `require_admin_user` — an account with the admin
role, or the `X-Admin-Key` break-glass header — and every mutation writes an
audit row, because "who suspended this account" is the first question asked
when something looks wrong.

Reads are aggregates rather than dumps. The user list is paginated, usage is
bucketed server-side, and the graph sample is capped, so a growing corpus can't
turn an admin page load into a full table scan.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import delete, func, select

from graphrag.accounts import AccountError
from graphrag.api.deps import AuthUser, get_container, get_db, require_admin_user
from graphrag.api.schemas import (
    Acknowledged,
    AdminUser,
    AdminUserCreate,
    AdminUserDetail,
    AdminUserList,
    BulkLimits,
    GraphSample,
    LimitsPatch,
    ModelOption,
    ModelSettings,
    ModelSettingsUpdate,
    PurgeResult,
    StoredFile,
    SystemStatus,
    UsagePoint,
    UsageSeries,
    UserPatch,
)
from graphrag.container import Container
from graphrag.core.logging import get_logger
from graphrag.db.engine import session_scope
from graphrag.db.models import (
    LIMIT_COLUMNS,
    AppSetting,
    AuditLog,
    File,
    GlobalLimit,
    Shelf,
    Thread,
    UsageEvent,
    User,
    UserLimit,
)
from graphrag.llm.registry import allowed_models, resolve_model

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin_user)])
log = get_logger(__name__)

_MODELS_KEY = "enabled_models"


_NEEDS_DB = "The admin panel needs a database."


def _require_db(db):
    if db is None:
        raise HTTPException(status_code=503, detail=_NEEDS_DB)
    return db


def _require_accounts(request: Request):
    """The account service, or 503.

    `available` as well as `is not None`: the service object exists even when
    the deployment has no database, and calling into it then raises out of
    `session_scope` — a 500 where the honest answer is "this needs Postgres".
    """
    accounts = request.app.state.accounts
    if accounts is None or not accounts.available:
        raise HTTPException(status_code=503, detail=_NEEDS_DB)
    return accounts


def _require_key_store(request: Request):
    key_store = request.app.state.key_store
    if key_store is None or not key_store.available:
        raise HTTPException(status_code=503, detail=_NEEDS_DB)
    return key_store


def _uid(value: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=404, detail="No such user.") from None


async def _audit(
    s, actor: AuthUser | None, action: str, target: uuid.UUID | None = None, **detail
) -> None:
    actor_id = None
    if actor is not None:
        try:
            actor_id = uuid.UUID(str(actor.user_id))
        except (ValueError, AttributeError, TypeError):
            actor_id = None  # the break-glass admin key has no account row
    s.add(
        AuditLog(
            actor_user_id=actor_id, action=action, target_user_id=target, detail=detail
        )
    )


def _shape_user(user: User, **counts) -> AdminUser:
    # `locked_until` is reported as-is rather than as a boolean: an admin
    # deciding whether to unlock someone needs to know it clears in 60 seconds
    # versus an hour.
    return AdminUser(
        id=str(user.id),
        email=user.email,
        role=user.role,
        status=user.status,
        tenant_id=user.tenant_id,
        created_at=user.created_at.isoformat() if user.created_at else "",
        last_login_at=user.last_login_at.isoformat() if user.last_login_at else None,
        email_verified=user.email_verified_at is not None,
        locked_until=user.locked_until.isoformat() if user.locked_until else None,
        failed_logins=user.failed_logins or 0,
        password_changed_at=(
            user.password_changed_at.isoformat() if user.password_changed_at else None
        ),
        **counts,
    )


# -- users --------------------------------------------------------------------

@router.get("/users", response_model=AdminUserList)
async def list_users(
    query: str = Query("", description="match on email"),
    status: str = Query("", description="pending | active | suspended"),
    page: int = Query(1, ge=1),
    size: int = Query(25, ge=1, le=100),
    db=Depends(get_db),
) -> AdminUserList:
    since = datetime.now(UTC) - timedelta(days=30)
    async with session_scope(_require_db(db)) as s:
        where = []
        if query:
            where.append(User.email.ilike(f"%{query}%"))
        if status:
            where.append(User.status == status)

        total = (
            await s.execute(select(func.count()).select_from(User).where(*where))
        ).scalar_one()
        rows = (
            await s.execute(
                select(User)
                .where(*where)
                .order_by(User.created_at.desc())
                .offset((page - 1) * size)
                .limit(size)
            )
        ).scalars().all()

        ids = [u.id for u in rows]
        files = await _count_by_user(s, File, File.user_id, ids)
        threads = await _count_by_user(
            s, Thread, Thread.user_id, ids, Thread.deleted_at.is_(None)
        )
        usage = await _usage_by_user(s, ids, since)

    return AdminUserList(
        users=[
            _shape_user(
                u,
                files=files.get(u.id, 0),
                threads=threads.get(u.id, 0),
                messages_30d=usage.get((u.id, "message"), 0),
                tokens_30d=usage.get((u.id, "tokens_out"), 0),
            )
            for u in rows
        ],
        total=int(total),
        page=page,
        size=size,
    )


async def _count_by_user(s, model, column, ids, *extra) -> dict:
    """One grouped query instead of a count per row on the page."""
    if not ids:
        return {}
    rows = await s.execute(
        select(column, func.count())
        .select_from(model)
        .where(column.in_(ids), *extra)
        .group_by(column)
    )
    return dict(rows.all())


async def _usage_by_user(s, ids, since) -> dict:
    if not ids:
        return {}
    rows = await s.execute(
        select(UsageEvent.user_id, UsageEvent.kind, func.sum(UsageEvent.amount))
        .where(UsageEvent.user_id.in_(ids), UsageEvent.created_at >= since)
        .group_by(UsageEvent.user_id, UsageEvent.kind)
    )
    return {(uid, kind): int(total or 0) for uid, kind, total in rows.all()}


@router.get("/users/{user_id}", response_model=AdminUserDetail)
async def user_detail(
    user_id: str,
    request: Request,
    db=Depends(get_db),
    container: Container = Depends(get_container),
) -> AdminUserDetail:
    owner = _uid(user_id)
    since = datetime.now(UTC) - timedelta(days=30)

    async with session_scope(_require_db(db)) as s:
        user = (await s.execute(select(User).where(User.id == owner))).scalar_one_or_none()
        if user is None:
            raise HTTPException(status_code=404, detail="No such user.")

        files_count, stored = (
            await s.execute(
                select(func.count(), func.coalesce(func.sum(File.size_bytes), 0))
                .where(File.user_id == owner)
            )
        ).one()
        threads = (
            await s.execute(
                select(func.count()).select_from(Thread)
                .where(Thread.user_id == owner, Thread.deleted_at.is_(None))
            )
        ).scalar_one()
        usage_rows = (
            await s.execute(
                select(UsageEvent.kind, func.sum(UsageEvent.amount))
                .where(UsageEvent.user_id == owner, UsageEvent.created_at >= since)
                .group_by(UsageEvent.kind)
            )
        ).all()
        override = (
            await s.execute(select(UserLimit).where(UserLimit.user_id == owner))
        ).scalar_one_or_none()
        file_rows = (
            await s.execute(
                select(File).where(File.user_id == owner).order_by(File.created_at.desc())
            )
        ).scalars().all()
        # Every shelf, plus the default (slug "") whose row may not exist on an
        # account created before shelves. Without this the panel would report
        # only the default shelf's graph and quietly understate anyone who keeps
        # their documents on a named one.
        shelf_slugs = set(
            (
                await s.execute(select(Shelf.slug).where(Shelf.user_id == owner))
            ).scalars().all()
        )
        shelf_slugs.add("")
        tenant_id = user.tenant_id
        shaped = _shape_user(
            user,
            files=int(files_count),
            threads=int(threads),
            messages_30d=int(dict(usage_rows).get("message", 0) or 0),
            tokens_30d=int(dict(usage_rows).get("tokens_out", 0) or 0),
        )

    limits = await request.app.state.limits.effective(user_id)
    return AdminUserDetail(
        user=shaped,
        limits=limits.as_dict(),
        overrides={
            c: (getattr(override, c) if override is not None else None)
            for c in LIMIT_COLUMNS
        },
        usage={k: int(v or 0) for k, v in usage_rows},
        storage_used_mb=round(int(stored) / (1024 * 1024), 2),
        graph=_graph_stats(container, tenant_id, shelf_slugs),
        files=[
            StoredFile(
                file_id=f.id, name=f.name, source=f.path,
                shelf_id=str(f.shelf_id) if f.shelf_id else None,
            )
            for f in file_rows
        ],
    )


def _graph_store(container: Container, tenant_id: str, shelf: str = ""):
    """A graph store for one shelf, without building the rest of the tenant.

    `container.tenant()` would also construct the embedder, reranker and agent —
    so inspecting a user's graph would load models, and stall on a provider
    that happens to be unreachable. Reading the graph needs the driver only.
    """
    from graphrag.storage import build_graph_store

    database, corpus = container._resolve_scope(tenant_id, shelf)
    return build_graph_store(container.driver, database, corpus, container.settings)


def _graph_stats(
    container: Container, tenant_id: str, shelves: set[str] | None = None
) -> dict[str, int]:
    """The user's graph totals, summed over their shelves.

    Summed rather than reported per shelf because this feeds an account
    overview: an admin looking at "does this user have a knowledge graph" wants
    one set of numbers. Neo4j may be down, and an admin page should still render
    the rest — so a failure on any shelf contributes nothing rather than raising.
    """
    totals: dict[str, int] = {}
    for slug in sorted(shelves if shelves is not None else {""}):
        try:
            for key, value in _graph_store(container, tenant_id, slug).stats().items():
                totals[key] = totals.get(key, 0) + int(value or 0)
        except Exception as exc:
            log.warning(
                "graph_stats_unavailable", tenant=tenant_id, shelf=slug, error=str(exc)
            )
    return totals


@router.post("/users", response_model=AdminUser, status_code=201)
async def create_user(
    payload: AdminUserCreate,
    request: Request,
    admin: AuthUser | None = Depends(require_admin_user),
    db=Depends(get_db),
) -> AdminUser:
    """Invite an account into existence.

    This is what makes `open_registration: false` a usable configuration
    rather than a dead end — without it, closing registration closes onboarding
    too. No password crosses this boundary: the invitee sets their own via the
    emailed code, through the same reset flow everyone else uses.
    """
    accounts = _require_accounts(request)
    try:
        user = await accounts.admin_create_user(payload.email, payload.role)
    except AccountError as exc:
        status = 409 if exc.code == "already_exists" else 400
        raise HTTPException(
            status_code=status, detail={"code": exc.code, "message": str(exc)}
        ) from None

    async with session_scope(_require_db(db)) as s:
        await _audit(s, admin, "user.create", user.id, email=user.email, role=user.role)
    return _shape_user(user)


@router.post("/users/{user_id}/force-password-reset", response_model=Acknowledged)
async def force_password_reset(
    user_id: str,
    request: Request,
    admin: AuthUser | None = Depends(require_admin_user),
    db=Depends(get_db),
) -> Acknowledged:
    """Cut a user off and email them a code to set a new password.

    The remedy for "this account may be compromised": sessions die first, then
    the code goes out, so there is no window in which a stolen session is still
    usable.
    """
    accounts = _require_accounts(request)
    if not await accounts.admin_force_reset(user_id):
        raise HTTPException(status_code=404, detail="No such user.")
    async with session_scope(_require_db(db)) as s:
        await _audit(s, admin, "user.force_reset", _uid(user_id))
    return Acknowledged(message="Sessions revoked and a reset code sent.")


@router.post("/users/{user_id}/unlock", response_model=Acknowledged)
async def unlock_user(
    user_id: str,
    request: Request,
    admin: AuthUser | None = Depends(require_admin_user),
    db=Depends(get_db),
) -> Acknowledged:
    """Clear a login lockout.

    Required, not a convenience: without it the only remedy for a locked-out
    user is waiting out the backoff, which reaches an hour. `graphrag unlock`
    is the equivalent for when the locked-out account is the only admin.
    """
    accounts = _require_accounts(request)
    if not await accounts.unlock(user_id):
        raise HTTPException(status_code=404, detail="No such user.")
    async with session_scope(_require_db(db)) as s:
        await _audit(s, admin, "user.unlock", _uid(user_id))
    return Acknowledged(message="Account unlocked.")


@router.patch("/users/{user_id}", response_model=AdminUser)
async def patch_user(
    user_id: str,
    payload: UserPatch,
    request: Request,
    admin: AuthUser | None = Depends(require_admin_user),
    db=Depends(get_db),
) -> AdminUser:
    """Suspend, reactivate, or change a role."""
    owner = _uid(user_id)
    if payload.status and payload.status not in ("active", "suspended"):
        raise HTTPException(status_code=400, detail="status must be active or suspended")
    if payload.role and payload.role not in ("user", "admin"):
        raise HTTPException(status_code=400, detail="role must be user or admin")

    async with session_scope(_require_db(db)) as s:
        user = (await s.execute(select(User).where(User.id == owner))).scalar_one_or_none()
        if user is None:
            raise HTTPException(status_code=404, detail="No such user.")
        if payload.status:
            user.status = payload.status
        if payload.role:
            user.role = payload.role
        await _audit(
            s, admin, "user.patch", owner, status=payload.status, role=payload.role
        )
        shaped = _shape_user(user)

    if payload.status == "suspended":
        # Don't wait for caches to expire — a suspension should bite now.
        #
        # Deliberately soft rather than `_require_accounts`: the status change
        # is already committed at this point, so raising here would report a
        # suspension that did in fact happen as a failure. `available` guards
        # the same call raising out of `session_scope` for want of a database.
        accounts = request.app.state.accounts
        if accounts is not None and accounts.available:
            await accounts.revoke_sessions(user_id)
        key_store = request.app.state.key_store
        if key_store is not None and key_store.available:
            await key_store.revoke_user(user_id)
    return shaped


@router.post("/users/{user_id}/revoke-keys", response_model=Acknowledged)
async def revoke_keys(
    user_id: str,
    request: Request,
    admin: AuthUser | None = Depends(require_admin_user),
    db=Depends(get_db),
) -> Acknowledged:
    # Both checks before the write: revoking keys and then failing to audit it
    # is the one ordering that loses the record of what happened.
    _require_db(db)
    revoked = await _require_key_store(request).revoke_user(user_id)
    async with session_scope(db) as s:
        await _audit(s, admin, "user.revoke_keys", _uid(user_id), revoked=revoked)
    return Acknowledged(message=f"Revoked {revoked} key(s).")


@router.post("/users/{user_id}/resend-verification", response_model=Acknowledged)
async def resend_verification(
    user_id: str,
    request: Request,
    db=Depends(get_db),
) -> Acknowledged:
    async with session_scope(_require_db(db)) as s:
        user = (
            await s.execute(select(User).where(User.id == _uid(user_id)))
        ).scalar_one_or_none()
        if user is None:
            raise HTTPException(status_code=404, detail="No such user.")
        email = user.email
    await _require_accounts(request).resend_code(email)
    return Acknowledged(message="Verification code sent.")


@router.delete("/users/{user_id}", response_model=PurgeResult)
async def delete_user(
    user_id: str,
    request: Request,
    keep_account: bool = Query(False, description="wipe content but keep the login"),
    admin: AuthUser | None = Depends(require_admin_user),
    db=Depends(get_db),
    container: Container = Depends(get_container),
) -> PurgeResult:
    """Remove a user's data from every store: graph, vectors, files, rows."""
    from graphrag.accounts.purge import purge_user

    report = await purge_user(
        _require_db(db), container, user_id, keep_account=keep_account
    )
    if report.errors and not report.rows_removed:
        raise HTTPException(status_code=404, detail="; ".join(report.errors))

    async with session_scope(db) as s:
        # The user row may be gone, so this is recorded without a foreign-key
        # target — the id lives in the detail instead.
        await _audit(
            s, admin, "user.purge", None,
            purged_user=user_id, kept_account=keep_account, errors=report.errors,
        )
    return PurgeResult(**report.as_dict())


# -- limits -------------------------------------------------------------------

@router.get("/limits", response_model=dict)
async def get_global_limits(db=Depends(get_db)) -> dict:
    async with session_scope(_require_db(db)) as s:
        row = (await s.execute(select(GlobalLimit))).scalars().first()
        if row is None:
            return {}
        return {c: getattr(row, c) for c in LIMIT_COLUMNS}


@router.put("/limits", response_model=dict)
async def set_global_limits(
    payload: LimitsPatch,
    request: Request,
    admin: AuthUser | None = Depends(require_admin_user),
    db=Depends(get_db),
) -> dict:
    """Change the defaults every user inherits."""
    values = {k: v for k, v in payload.model_dump().items() if v is not None}
    async with session_scope(_require_db(db)) as s:
        row = (await s.execute(select(GlobalLimit))).scalars().first()
        if row is None:
            row = GlobalLimit(id=1)
            s.add(row)
        for key, value in values.items():
            setattr(row, key, value)
        await _audit(s, admin, "limits.global", None, **values)
        result = {c: getattr(row, c) for c in LIMIT_COLUMNS}

    # Every user's effective limits just changed.
    request.app.state.limits.invalidate()
    return result


@router.put("/users/{user_id}/limits", response_model=dict)
async def set_user_limits(
    user_id: str,
    payload: LimitsPatch,
    request: Request,
    admin: AuthUser | None = Depends(require_admin_user),
    db=Depends(get_db),
) -> dict:
    """Override limits for one user. Null clears a field back to the default."""
    owner = _uid(user_id)
    values = payload.model_dump()
    async with session_scope(_require_db(db)) as s:
        row = (
            await s.execute(select(UserLimit).where(UserLimit.user_id == owner))
        ).scalar_one_or_none()
        if row is None:
            row = UserLimit(user_id=owner)
            s.add(row)
        for key, value in values.items():
            setattr(row, key, value)
        await _audit(s, admin, "limits.user", owner, **dict(values))
        result = {c: getattr(row, c) for c in LIMIT_COLUMNS}

    request.app.state.limits.invalidate(user_id)
    return result


@router.delete("/users/{user_id}/limits", response_model=Acknowledged)
async def clear_user_limits(
    user_id: str,
    request: Request,
    admin: AuthUser | None = Depends(require_admin_user),
    db=Depends(get_db),
) -> Acknowledged:
    owner = _uid(user_id)
    async with session_scope(_require_db(db)) as s:
        await s.execute(delete(UserLimit).where(UserLimit.user_id == owner))
        await _audit(s, admin, "limits.user.clear", owner)
    request.app.state.limits.invalidate(user_id)
    return Acknowledged(message="Overrides cleared; global defaults apply.")


@router.post("/limits/bulk", response_model=Acknowledged)
async def bulk_limits(
    payload: BulkLimits,
    request: Request,
    admin: AuthUser | None = Depends(require_admin_user),
    db=Depends(get_db),
) -> Acknowledged:
    """Apply one change to every user at once, or drop all overrides."""
    async with session_scope(_require_db(db)) as s:
        if payload.clear:
            await s.execute(delete(UserLimit))
            await _audit(s, admin, "limits.bulk.clear", None)
            message = "All per-user overrides cleared."
        else:
            values = {
                k: v for k, v in (payload.set.model_dump() if payload.set else {}).items()
                if v is not None
            }
            if not values:
                raise HTTPException(status_code=400, detail="Nothing to apply.")
            ids = (await s.execute(select(User.id))).scalars().all()
            existing = set(
                (await s.execute(select(UserLimit.user_id))).scalars().all()
            )
            for uid in ids:
                if uid in existing:
                    row = (
                        await s.execute(select(UserLimit).where(UserLimit.user_id == uid))
                    ).scalar_one()
                else:
                    row = UserLimit(user_id=uid)
                    s.add(row)
                for key, value in values.items():
                    setattr(row, key, value)
            await _audit(s, admin, "limits.bulk.set", None, users=len(ids), **values)
            message = f"Applied to {len(ids)} user(s)."

    request.app.state.limits.invalidate()
    return Acknowledged(message=message)


# -- usage --------------------------------------------------------------------

@router.get("/usage", response_model=UsageSeries)
async def usage_series(
    days: int = Query(30, ge=1, le=365),
    user_id: str = Query("", description="restrict to one user"),
    db=Depends(get_db),
) -> UsageSeries:
    """Daily message/token/upload counts, bucketed in the database."""
    since = datetime.now(UTC) - timedelta(days=days)
    where = [UsageEvent.created_at >= since]
    if user_id:
        where.append(UsageEvent.user_id == _uid(user_id))

    bucket = func.date_trunc("day", UsageEvent.created_at).label("bucket")
    async with session_scope(_require_db(db)) as s:
        rows = (
            await s.execute(
                select(bucket, UsageEvent.kind, func.sum(UsageEvent.amount))
                .where(*where)
                .group_by(bucket, UsageEvent.kind)
                .order_by(bucket)
            )
        ).all()

    points: dict[str, UsagePoint] = {}
    totals = {"messages": 0, "tokens": 0, "uploads": 0}
    field_for = {"message": "messages", "tokens_out": "tokens", "upload": "uploads"}
    for when, kind, total in rows:
        field = field_for.get(kind)
        if field is None:
            continue
        key = when.date().isoformat()
        point = points.setdefault(key, UsagePoint(bucket=key))
        setattr(point, field, getattr(point, field) + int(total or 0))
        totals[field] += int(total or 0)
    return UsageSeries(points=list(points.values()), totals=totals)


# -- graph inspection ---------------------------------------------------------

async def _tenant_and_shelves(s, user_id: str) -> tuple[str, set[str]]:
    """A user's storage namespace and every shelf slug in it, or 404."""
    user = (
        await s.execute(select(User).where(User.id == _uid(user_id)))
    ).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="No such user.")
    slugs = set(
        (
            await s.execute(select(Shelf.slug).where(Shelf.user_id == user.id))
        ).scalars().all()
    )
    slugs.add("")  # the default shelf may have no row on an older account
    return user.tenant_id, slugs


@router.get("/users/{user_id}/graph", response_model=dict)
async def user_graph_stats(
    user_id: str, db=Depends(get_db), container: Container = Depends(get_container)
) -> dict:
    """Graph totals across every shelf the user has."""
    async with session_scope(_require_db(db)) as s:
        tenant_id, slugs = await _tenant_and_shelves(s, user_id)
    return _graph_stats(container, tenant_id, slugs)


@router.get("/users/{user_id}/graph/sample", response_model=GraphSample)
async def user_graph_sample(
    user_id: str,
    limit: int = Query(100, ge=1, le=500),
    shelf: str = Query("", max_length=32),
    db=Depends(get_db),
    container: Container = Depends(get_container),
) -> GraphSample:
    """Highest-degree slice of one shelf's entity graph, for visualization.

    One shelf, not the union: each is a separate graph with no edges between
    them, so drawing them together would render as disconnected islands that
    imply a relationship the data does not have. `shelf` is a slug; empty is the
    default shelf.
    """
    async with session_scope(_require_db(db)) as s:
        tenant_id, slugs = await _tenant_and_shelves(s, user_id)
    if shelf not in slugs:
        raise HTTPException(status_code=404, detail="No such shelf for this user.")
    try:
        return GraphSample(
            **_graph_store(container, tenant_id, shelf).sample_subgraph(limit)
        )
    except Exception as exc:
        log.warning(
            "graph_sample_unavailable", tenant=tenant_id, shelf=shelf, error=str(exc)
        )
        return GraphSample()


# -- system + models ----------------------------------------------------------

@router.get("/system", response_model=SystemStatus)
async def system_status(
    request: Request,
    db=Depends(get_db),
    container: Container = Depends(get_container),
) -> SystemStatus:
    from graphrag import __version__

    status = SystemStatus(
        version=__version__,
        redis=container.redis is not None,
        database=db is not None,
        vector_provider=container.settings.storage.vector.provider,
        memory_backend=container.settings.agent.memory_backend,
        default_model=resolve_model(None, container.settings).model,
    )
    try:
        container.driver.verify_connectivity()
        status.neo4j = True
    except Exception:
        status.neo4j = False

    if db is not None:
        async with session_scope(db) as s:
            status.users = int(
                (await s.execute(select(func.count()).select_from(User))).scalar_one()
            )
            status.active_users = int(
                (
                    await s.execute(
                        select(func.count()).select_from(User).where(User.status == "active")
                    )
                ).scalar_one()
            )
            status.threads = int(
                (
                    await s.execute(
                        select(func.count()).select_from(Thread)
                        .where(Thread.deleted_at.is_(None))
                    )
                ).scalar_one()
            )
            status.files = int(
                (await s.execute(select(func.count()).select_from(File))).scalar_one()
            )
    return status


@router.get("/models", response_model=ModelSettings)
async def get_models(
    db=Depends(get_db), container: Container = Depends(get_container)
) -> ModelSettings:
    available = [
        ModelOption(model=m.model, label=m.label or m.model, provider=m.provider)
        for m in allowed_models(container.settings)
    ]
    enabled = [m.model for m in allowed_models(container.settings)]
    if db is not None:
        async with session_scope(db) as s:
            row = (
                await s.execute(select(AppSetting).where(AppSetting.key == _MODELS_KEY))
            ).scalar_one_or_none()
            if row is not None and isinstance(row.value, dict):
                stored = row.value.get("enabled")
                if isinstance(stored, list) and stored:
                    enabled = stored
    return ModelSettings(available=available, enabled=enabled)


@router.put("/models", response_model=ModelSettings)
async def set_models(
    payload: ModelSettingsUpdate,
    request: Request,
    admin: AuthUser | None = Depends(require_admin_user),
    db=Depends(get_db),
    container: Container = Depends(get_container),
) -> ModelSettings:
    """Narrow what the chat UI offers. Disabling everything is refused — it
    would leave users with no model to talk to.

    The write-back to `app.state` is what makes this setting mean anything. It
    was persisted and then read by nobody: /query resolved model overrides
    against `llm.allowed` alone, so a disabled model vanished from the picker
    and stayed fully callable over the API.
    """
    known = {m.model for m in allowed_models(container.settings)}
    enabled = [m for m in payload.enabled if m in known]
    if not enabled:
        raise HTTPException(status_code=400, detail="Enable at least one model.")

    async with session_scope(_require_db(db)) as s:
        row = (
            await s.execute(select(AppSetting).where(AppSetting.key == _MODELS_KEY))
        ).scalar_one_or_none()
        if row is None:
            s.add(AppSetting(key=_MODELS_KEY, value={"enabled": enabled}))
        else:
            row.value = {"enabled": enabled}
        await _audit(s, admin, "models.set", None, enabled=enabled)

    request.app.state.enabled_models = enabled
    return await get_models(db, container)


# -- audit --------------------------------------------------------------------

@router.get("/audit", response_model=list)
async def audit_log(
    limit: int = Query(100, ge=1, le=500),
    db=Depends(get_db),
) -> list:
    async with session_scope(_require_db(db)) as s:
        rows = (
            await s.execute(
                select(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit)
            )
        ).scalars().all()
    return [
        {
            "id": r.id,
            "action": r.action,
            "actor": str(r.actor_user_id) if r.actor_user_id else None,
            "target": str(r.target_user_id) if r.target_user_id else None,
            "detail": r.detail,
            "created_at": r.created_at.isoformat() if r.created_at else "",
        }
        for r in rows
    ]
