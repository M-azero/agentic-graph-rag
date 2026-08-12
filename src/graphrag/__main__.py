"""Command-line interface.  `graphrag --help`"""

from __future__ import annotations

import typer

app = typer.Typer(add_completion=False, help="Agentic Graph RAG command line.")


@app.command()
def init() -> None:
    """Create Neo4j constraints and indexes."""
    from graphrag.container import Container

    Container().setup_storage()
    typer.echo("Storage initialized.")


@app.command()
def ingest(path: str, user: str = typer.Option("default", help="user namespace")) -> None:
    """Ingest a file or folder into a user's knowledge base."""
    from graphrag.container import Container
    from graphrag.pipelines import IngestPipeline

    stats = IngestPipeline(Container()).run(path, user_id=user)
    typer.echo(
        f"Done. documents={stats.documents} chunks={stats.chunks} "
        f"entities={stats.entities} relations={stats.relations}"
    )


@app.command()
def query(
    question: str,
    style: str = typer.Option("detailed", help="concise | detailed | technical | eli5"),
    user: str = typer.Option("default", help="user namespace"),
) -> None:
    """Ask a question from the terminal."""
    from graphrag.container import Container
    from graphrag.pipelines import QueryService

    result = QueryService(Container()).answer(question, style=style, user_id=user)
    typer.echo("\n" + result.answer + "\n")
    if result.sources:
        typer.echo("Sources:")
        for s in result.sources:
            typer.echo(f"  - {s.source}")


@app.command()
def search(
    query: str, k: int = 8, user: str = typer.Option("default", help="user namespace")
) -> None:
    """Show raw retrieval results (no LLM)."""
    from graphrag.container import Container
    from graphrag.pipelines import QueryService

    for i, chunk in enumerate(QueryService(Container()).search(query, k, user_id=user), 1):
        typer.echo(f"{i}. [{chunk.retriever} {chunk.score:.3f}] {chunk.source}")
        typer.echo(f"   {chunk.text[:160]}...")


def _accounts(container):
    """Build the account services against Postgres, for CLI use."""
    from graphrag.accounts import AccountService, PgKeyStore, build_email_sender
    from graphrag.db import build_engine, build_sessionmaker

    engine = build_engine(container.secrets.database_url)
    factory = build_sessionmaker(engine)
    sender = build_email_sender(container.settings, container.secrets)
    return (
        engine,
        AccountService(factory, container.settings, sender, container.redis),
        PgKeyStore(factory, container.redis),
    )


def _run(coro):
    """Run one async call and dispose the engine afterwards.

    A selector loop, because psycopg's async mode refuses the ProactorEventLoop
    that Windows uses by default.
    """
    import asyncio
    import selectors
    import sys

    if sys.platform == "win32":
        loop = asyncio.SelectorEventLoop(selectors.SelectSelector())
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    return asyncio.run(coro)


@app.command()
def apikey(email: str, label: str = typer.Option("", help="a note to identify this key")) -> None:
    """Mint an API key for an account (prints it once)."""
    from sqlalchemy import func, select

    from graphrag.accounts import normalize_email
    from graphrag.container import Container
    from graphrag.db.models import User

    container = Container()
    engine, accounts, keys = _accounts(container)

    async def _mint() -> str | None:
        from graphrag.db.engine import session_scope

        async with session_scope(keys._factory) as s:
            user = (
                await s.execute(
                    select(User).where(func.lower(User.email) == normalize_email(email))
                )
            ).scalar_one_or_none()
            if user is None:
                return None
            user_id, tenant = str(user.id), user.tenant_id
        _key_id, key = await keys.create_key(user_id, label)
        container.tenant(tenant)  # ensure the namespace exists
        await engine.dispose()
        return key

    key = _run(_mint())
    if key is None:
        typer.echo(f"No account for {email}. Sign up first.")
        raise typer.Exit(1)
    typer.echo(key)


@app.command()
def revoke(email: str) -> None:
    """Revoke every API key an account holds (their data stays)."""
    from sqlalchemy import func, select

    from graphrag.accounts import normalize_email
    from graphrag.container import Container
    from graphrag.db.models import User

    container = Container()
    engine, accounts, keys = _accounts(container)

    async def _revoke() -> int | None:
        from graphrag.db.engine import session_scope

        async with session_scope(keys._factory) as s:
            user = (
                await s.execute(
                    select(User).where(func.lower(User.email) == normalize_email(email))
                )
            ).scalar_one_or_none()
            if user is None:
                return None
            user_id = str(user.id)
        count = await keys.revoke_user(user_id)
        await engine.dispose()
        return count

    count = _run(_revoke())
    if count is None:
        typer.echo(f"No account for {email}.")
        raise typer.Exit(1)
    typer.echo(f"Revoked {count} key(s) for '{email}'.")


@app.command("promote-admin")
def promote_admin(email: str) -> None:
    """Grant an existing account the admin role (and activate it)."""
    from graphrag.container import Container

    container = Container()
    engine, accounts, _keys = _accounts(container)

    async def _promote() -> bool:
        ok = await accounts.promote_admin(email)
        await engine.dispose()
        return ok

    if not _run(_promote()):
        typer.echo(f"No account for {email}. Sign up first, then run this again.")
        raise typer.Exit(1)
    typer.echo(f"{email} is now an admin.")


@app.command()
def unlock(email: str) -> None:
    """Clear a login lockout on an account.

    The break-glass for the case the admin panel cannot fix: the locked-out
    account is the only admin, so there is nobody left who can sign in to
    unlock it. `GRAPHRAG_ADMIN_KEY` bypasses login entirely but only as a
    header, which cannot drive the cookie-based console.
    """
    from graphrag.container import Container

    container = Container()
    engine, accounts, _keys = _accounts(container)

    async def _unlock() -> bool:
        user = await accounts.get_by_email(email)
        ok = await accounts.unlock(str(user.id)) if user is not None else False
        await engine.dispose()
        return ok

    if not _run(_unlock()):
        typer.echo(f"No account for {email}.")
        raise typer.Exit(1)
    typer.echo(f"{email} can sign in again.")


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Start the API server."""
    import uvicorn

    uvicorn.run("graphrag.api.app:create_app", factory=True, host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
