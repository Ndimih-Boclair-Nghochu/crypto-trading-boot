from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from config import BASE_DIR, Settings, settings
from utils.logger import logger

try:
    import asyncpg
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import text
    from sqlalchemy.engine import make_url
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
except Exception:  # pragma: no cover
    asyncpg = None
    command = None
    Config = None
    text = None
    make_url = None
    AsyncEngine = None
    AsyncSession = None
    async_sessionmaker = None
    create_async_engine = None


class Database:
    def __init__(self, cfg: Settings = settings) -> None:
        self.settings = cfg
        self.engine: AsyncEngine | None = None
        self.sessionmaker: async_sessionmaker[AsyncSession] | None = None
        self.pool: Any | None = None

    async def initialize(self) -> None:
        if create_async_engine is None or asyncpg is None or make_url is None:
            logger.warning("Database dependencies unavailable; PostgreSQL disabled.")
            return
        if not self.settings.database_url.startswith("postgresql+asyncpg://"):
            raise RuntimeError("DATABASE_URL must use postgresql+asyncpg; SQLite fallback is intentionally unsupported.")
        self.engine = create_async_engine(
            self.settings.database_url,
            pool_size=5,
            max_overflow=15,
            pool_pre_ping=True,
            future=True,
        )
        self.sessionmaker = async_sessionmaker(self.engine, expire_on_commit=False)
        url = make_url(self.settings.database_url)
        self.pool = await asyncpg.create_pool(
            user=url.username,
            password=url.password,
            database=url.database,
            host=url.host,
            port=url.port or 5432,
            min_size=5,
            max_size=20,
        )

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()
        if self.engine:
            await self.engine.dispose()

    async def run_migrations(self) -> None:
        if command is None or Config is None:
            raise RuntimeError("Alembic is required to run migrations")
        alembic_ini = BASE_DIR / "db" / "migrations" / "alembic.ini"
        cfg = Config(str(alembic_ini))
        cfg.set_main_option("sqlalchemy.url", self.settings.database_url)
        await asyncio.to_thread(command.upgrade, cfg, "head")

    async def execute(self, statement: str, params: dict[str, Any] | None = None) -> None:
        if not self.sessionmaker or text is None:
            logger.warning("Database execute skipped; database not initialized.")
            return
        async with self.sessionmaker() as session:
            async with session.begin():
                await session.execute(text(statement), params or {})

    async def fetch_all(self, statement: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if not self.sessionmaker or text is None:
            return []
        async with self.sessionmaker() as session:
            result = await session.execute(text(statement), params or {})
            return [dict(row._mapping) for row in result.fetchall()]

    async def fetch_one(self, statement: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        rows = await self.fetch_all(statement, params)
        return rows[0] if rows else None

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[Any]:
        if not self.sessionmaker:
            yield None
            return
        async with self.sessionmaker() as session:
            async with session.begin():
                yield session


database = Database()
