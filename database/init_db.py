"""Initialize PostgreSQL core tables, pgvector tables, and local SQLite tables."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Mapping

from sqlalchemy import text

from database.connection import (
    get_async_session,
    get_db_backend,
    get_local_dev_database_url,
    get_pgvector_database_url,
    get_storage_backend,
)
from utils.config_loader import get_app_config

DATABASE_DIR = Path(__file__).resolve().parent
POSTGRES_CORE_SCHEMA_PATH = DATABASE_DIR / "postgres_core_schema.sql"
PGVECTOR_SCHEMA_PATH = DATABASE_DIR / "pgvector_schema.sql"


def _sqlite_path(database_url: str) -> Path:
    value = str(database_url or "").strip()
    if value.startswith("sqlite+aiosqlite:///"):
        value = value.replace("sqlite+aiosqlite:///", "", 1)
    elif value.startswith("sqlite:///"):
        value = value.replace("sqlite:///", "", 1)
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def init_local_dev_schema(database_url: str | None = None) -> Path:
    sqlite_path = _sqlite_path(database_url or get_local_dev_database_url())
    connection = sqlite3.connect(sqlite_path)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;

            CREATE TABLE IF NOT EXISTS pdf_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL UNIQUE,
                collection_name TEXT NOT NULL,
                doc_source TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                doc_hash TEXT NOT NULL DEFAULT '',
                page_count INTEGER NOT NULL DEFAULT 0,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                indexed_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS pdf_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT NOT NULL UNIQUE,
                doc_id TEXT NOT NULL,
                collection_name TEXT NOT NULL,
                doc_source TEXT NOT NULL,
                page_idx INTEGER,
                page_range TEXT NOT NULL DEFAULT '',
                chunk_type TEXT NOT NULL DEFAULT 'text',
                chunk_index INTEGER NOT NULL DEFAULT 0,
                heading_path TEXT NOT NULL DEFAULT '',
                level1_title TEXT NOT NULL DEFAULT '',
                level2_title TEXT NOT NULL DEFAULT '',
                level3_title TEXT NOT NULL DEFAULT '',
                table_id TEXT NOT NULL DEFAULT '',
                sub_table_id TEXT NOT NULL DEFAULT '',
                table_header_text TEXT NOT NULL DEFAULT '',
                table_context_text TEXT NOT NULL DEFAULT '',
                search_text TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                embedding_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS qa_sessions (
                session_id TEXT PRIMARY KEY,
                collection_name TEXT NOT NULL,
                user_id TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS qa_messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                query_type TEXT NOT NULL DEFAULT '',
                question TEXT NOT NULL DEFAULT '',
                answer TEXT NOT NULL DEFAULT '',
                decision TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0,
                citations_json TEXT NOT NULL DEFAULT '[]',
                evidence_json TEXT NOT NULL DEFAULT '[]',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                retrieval_trace_id TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS retrieval_traces (
                trace_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL DEFAULT '',
                message_id TEXT NOT NULL DEFAULT '',
                collection_name TEXT NOT NULL DEFAULT '',
                question TEXT NOT NULL DEFAULT '',
                expanded_queries_json TEXT NOT NULL DEFAULT '[]',
                retrieval_trace_json TEXT NOT NULL DEFAULT '{}',
                rerank_trace_json TEXT NOT NULL DEFAULT '{}',
                selected_candidates_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS evaluation_records (
                evaluation_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL DEFAULT '',
                message_id TEXT NOT NULL DEFAULT '',
                metrics_json TEXT NOT NULL DEFAULT '{}',
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_pdf_documents_collection
                ON pdf_documents(collection_name, updated_at);
            CREATE INDEX IF NOT EXISTS idx_pdf_chunks_collection
                ON pdf_chunks(collection_name, doc_id, chunk_index);
            CREATE INDEX IF NOT EXISTS idx_qa_sessions_collection
                ON qa_sessions(collection_name, updated_at);
            CREATE INDEX IF NOT EXISTS idx_qa_messages_session
                ON qa_messages(session_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_retrieval_traces_session
                ON retrieval_traces(session_id, created_at);
            """
        )
        connection.commit()
    finally:
        connection.close()
    return sqlite_path


def _schema_statements(schema_path: Path) -> list[str]:
    source = schema_path.read_text(encoding="utf-8")
    return [statement.strip() for statement in source.split(";") if statement.strip()]


async def _execute_postgres_schema(database_url: str, schema_path: Path) -> None:
    async with get_async_session(backend="pgvector", database_url=database_url) as session:
        for statement in _schema_statements(schema_path):
            await session.execute(text(statement))
        await session.commit()


async def init_postgres_core_schema(database_url: str | None = None) -> None:
    selected_url = str(database_url or get_pgvector_database_url()).strip()
    if not selected_url:
        raise RuntimeError("PostgreSQL database URL is empty.")
    await _execute_postgres_schema(selected_url, POSTGRES_CORE_SCHEMA_PATH)


async def init_pgvector_schema(database_url: str | None = None) -> None:
    selected_url = str(database_url or get_pgvector_database_url()).strip()
    if not selected_url:
        raise RuntimeError("PGVECTOR_DATABASE_URL is empty.")
    await _execute_postgres_schema(selected_url, PGVECTOR_SCHEMA_PATH)


async def bootstrap_database(config: Mapping[str, Any] | None = None) -> None:
    app_config = config if config is not None else get_app_config()
    db_backend = get_db_backend(app_config)
    storage_backend = get_storage_backend(app_config)

    if db_backend == "pgvector":
        await init_postgres_core_schema(get_pgvector_database_url(app_config))
    elif db_backend == "local_dev":
        init_local_dev_schema(get_local_dev_database_url(app_config))
    else:
        raise RuntimeError(f"Unsupported db.backend: {db_backend}")

    if storage_backend == "pgvector":
        await init_pgvector_schema(get_pgvector_database_url(app_config))
    elif storage_backend == "local_dev":
        init_local_dev_schema(get_local_dev_database_url(app_config))
    else:
        raise RuntimeError(f"Unsupported storage.backend: {storage_backend}")
