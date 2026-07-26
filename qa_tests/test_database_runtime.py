from __future__ import annotations

import sqlite3

from database.connection import (
    get_db_backend,
    get_local_dev_database_url,
    get_pgvector_database_url,
    get_storage_backend,
)
from database.init_db import init_local_dev_schema


def test_database_config_helpers_resolve_separate_backends(monkeypatch) -> None:
    for key in (
        "DB_BACKEND",
        "STORAGE_BACKEND",
        "VECTOR_STORE_BACKEND",
        "PGVECTOR_DATABASE_URL",
        "LOCAL_DEV_DATABASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)

    config = {
        "db": {"backend": "pgvector"},
        "storage": {
            "backend": "local_dev",
            "pgvector": {"database_url": "postgresql+asyncpg://user:pass@localhost/example"},
            "local_dev": {"database_url": "sqlite:///database/example.db"},
        },
    }

    assert get_db_backend(config) == "pgvector"
    assert get_storage_backend(config) == "local_dev"
    assert get_pgvector_database_url(config).endswith("/example")
    assert get_local_dev_database_url(config) == "sqlite:///database/example.db"


def test_init_local_dev_schema_creates_runtime_tables(tmp_path) -> None:
    sqlite_path = init_local_dev_schema(f"sqlite:///{tmp_path / 'local.db'}")

    connection = sqlite3.connect(sqlite_path)
    try:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()
    finally:
        connection.close()

    table_names = {row[0] for row in rows}
    assert {
        "pdf_documents",
        "pdf_chunks",
        "qa_sessions",
        "qa_messages",
        "retrieval_traces",
        "evaluation_records",
    } <= table_names
