"""SQLAlchemy async engine and session management."""

from __future__ import annotations

import contextlib
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings

# Global engine (created lazily)
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        settings = get_settings()
        # SQLite needs special args for async
        connect_args = {}
        kwargs = {}
        if "sqlite" in settings.effective_database_url:
            connect_args = {"check_same_thread": False}
            # Use StaticPool for in-memory SQLite so all connections share
            # the same in-memory database (otherwise each connection gets
            # a separate fresh database).
            if ":memory:" in settings.effective_database_url:
                from sqlalchemy.pool import StaticPool

                kwargs["poolclass"] = StaticPool
        _engine = create_async_engine(
            settings.effective_database_url,
            echo=False,
            connect_args=connect_args,
            **kwargs,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


async def get_session() -> AsyncSession:
    """Dependency for FastAPI to get an async DB session."""
    factory = get_session_factory()
    async with factory() as session:
        yield session


@contextlib.asynccontextmanager
async def async_session() -> AsyncIterator[AsyncSession]:
    """Context manager for standalone async session usage."""
    factory = get_session_factory()
    async with factory() as session:
        yield session


async def init_db() -> None:
    """Create all tables. Used for testing/dev without Alembic."""
    from app.storage.orm_base import Base
    from app.storage import orm_models  # noqa: F401 - register models

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Dispose engine connections."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None


async def reset_engine() -> None:
    """Reset engine — useful for tests that change DATABASE_URL."""
    await close_db()
