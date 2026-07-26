"""Runtime database configuration and async SQLAlchemy sessions."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Mapping

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from utils.config_loader import get_app_config

_ASYNC_ENGINES: dict[str, AsyncEngine] = {}
_ASYNC_SESSION_FACTORIES: dict[str, async_sessionmaker[AsyncSession]] = {}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _config(config: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    return config if config is not None else get_app_config()


def get_storage_backend(config: Mapping[str, Any] | None = None) -> str:
    app_config = _config(config)
    storage = _mapping(app_config.get("storage"))
    vector = _mapping(app_config.get("vector"))
    return (
        _clean(os.getenv("STORAGE_BACKEND"))
        or _clean(os.getenv("VECTOR_STORE_BACKEND"))
        or _clean(storage.get("backend"))
        or _clean(vector.get("backend"))
        or "pgvector"
    ).lower()


def get_db_backend(config: Mapping[str, Any] | None = None) -> str:
    app_config = _config(config)
    database = _mapping(app_config.get("db"))
    return (_clean(os.getenv("DB_BACKEND")) or _clean(database.get("backend")) or "pgvector").lower()


def get_pgvector_database_url(config: Mapping[str, Any] | None = None) -> str:
    app_config = _config(config)
    storage = _mapping(app_config.get("storage"))
    pgvector = _mapping(storage.get("pgvector"))
    vector = _mapping(app_config.get("vector"))
    return (
        _clean(os.getenv("PGVECTOR_DATABASE_URL"))
        or _clean(pgvector.get("database_url"))
        or _clean(vector.get("pgvector_database_url"))
    )


def get_local_dev_database_url(config: Mapping[str, Any] | None = None) -> str:
    app_config = _config(config)
    storage = _mapping(app_config.get("storage"))
    local_dev = _mapping(storage.get("local_dev"))
    return (
        _clean(os.getenv("LOCAL_DEV_DATABASE_URL"))
        or _clean(local_dev.get("database_url"))
        or "sqlite:///database/local_dev.db"
    )


def _async_database_url(backend: str, database_url: str) -> str:
    if backend == "pgvector":
        if database_url.startswith("postgresql+asyncpg://"):
            return database_url
        if database_url.startswith("postgresql+psycopg2://"):
            return database_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
        if database_url.startswith("postgresql://"):
            return database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        raise RuntimeError("PostgreSQL database URL must start with postgresql:// or postgresql+asyncpg://.")
    if backend == "local_dev":
        if database_url.startswith("sqlite+aiosqlite:///"):
            return database_url
        if database_url.startswith("sqlite:///"):
            return database_url.replace("sqlite:///", "sqlite+aiosqlite:///", 1)
        raise RuntimeError("Local development database URL must start with sqlite:///.")
    raise RuntimeError(f"Unsupported database backend: {backend}")


def _database_url_for_backend(
    backend: str,
    database_url: str | None,
    config: Mapping[str, Any] | None,
) -> str:
    explicit_url = _clean(database_url)
    if explicit_url:
        return explicit_url
    if backend == "pgvector":
        return get_pgvector_database_url(config)
    if backend == "local_dev":
        return get_local_dev_database_url(config)
    raise RuntimeError(f"Unsupported database backend: {backend}")


def _session_factory(backend: str, database_url: str) -> async_sessionmaker[AsyncSession]:
    async_url = _async_database_url(backend, database_url)
    cached = _ASYNC_SESSION_FACTORIES.get(async_url)
    if cached is not None:
        return cached

    engine_options: dict[str, Any] = {"pool_pre_ping": True}
    if backend == "local_dev":
        engine_options["connect_args"] = {"check_same_thread": False}
    engine = create_async_engine(async_url, **engine_options)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    _ASYNC_ENGINES[async_url] = engine
    _ASYNC_SESSION_FACTORIES[async_url] = factory
    return factory


@asynccontextmanager
async def get_async_session(
    *,
    backend: str | None = None,
    database_url: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> AsyncIterator[AsyncSession]:
    selected_backend = (_clean(backend) or get_db_backend(config)).lower()
    selected_url = _database_url_for_backend(selected_backend, database_url, config)
    if not selected_url:
        raise RuntimeError(f"Database URL is empty for backend={selected_backend}.")

    factory = _session_factory(selected_backend, selected_url)
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def dispose_async_engines() -> None:
    engines = list(_ASYNC_ENGINES.values())
    _ASYNC_ENGINES.clear()
    _ASYNC_SESSION_FACTORIES.clear()
    for engine in engines:
        await engine.dispose()
