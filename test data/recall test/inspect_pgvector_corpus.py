from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from sqlalchemy import text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.connection import get_async_session, get_pgvector_database_url  # noqa: E402
from utils.config_loader import get_app_config  # noqa: E402


async def main() -> None:
    database_url = get_pgvector_database_url(get_app_config())
    async with get_async_session(backend="pgvector", database_url=database_url) as session:
        documents = (
            await session.execute(
                text(
                    """
                    SELECT
                        d.doc_id,
                        d.collection_name,
                        d.doc_source,
                        d.title,
                        d.doc_hash,
                        d.page_count,
                        d.indexed_at,
                        d.metadata_json,
                        COUNT(c.id) AS chunk_count,
                        MIN(c.page_idx) AS min_page_idx,
                        MAX(c.page_idx) AS max_page_idx,
                        COUNT(*) FILTER (WHERE c.embedding IS NOT NULL) AS embedded_chunk_count,
                        COUNT(*) FILTER (WHERE c.chunk_type = 'table') AS table_chunk_count
                    FROM pdf_documents d
                    LEFT JOIN pdf_chunks c ON c.doc_id = d.doc_id
                    GROUP BY
                        d.doc_id, d.collection_name, d.doc_source, d.title,
                        d.doc_hash, d.page_count, d.indexed_at, d.metadata_json
                    ORDER BY d.indexed_at DESC, d.doc_id
                    """
                )
            )
        ).mappings().all()

        scopes = (
            await session.execute(
                text(
                    """
                    SELECT
                        collection_name,
                        metadata_json->>'company_id' AS company_id,
                        metadata_json->>'company_name' AS company_name,
                        metadata_json->>'year' AS year,
                        COUNT(*) AS chunk_count,
                        MIN(page_idx) AS min_page_idx,
                        MAX(page_idx) AS max_page_idx
                    FROM pdf_chunks
                    GROUP BY
                        collection_name,
                        metadata_json->>'company_id',
                        metadata_json->>'company_name',
                        metadata_json->>'year'
                    ORDER BY collection_name, year, company_id
                    """
                )
            )
        ).mappings().all()

        samples = (
            await session.execute(
                text(
                    """
                    SELECT
                        chunk_id, doc_id, collection_name, doc_source, page_idx,
                        page_range, chunk_type, heading_path,
                        LEFT(search_text, 180) AS search_preview,
                        LEFT(content, 180) AS content_preview,
                        metadata_json
                    FROM pdf_chunks
                    WHERE page_idx IN (
                        SELECT MIN(page_idx) FROM pdf_chunks
                        UNION
                        SELECT MAX(page_idx) FROM pdf_chunks
                    )
                    ORDER BY doc_id, page_idx, chunk_index
                    LIMIT 20
                    """
                )
            )
        ).mappings().all()

    payload = {
        "documents": [dict(row) for row in documents],
        "scopes": [dict(row) for row in scopes],
        "boundary_samples": [dict(row) for row in samples],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
