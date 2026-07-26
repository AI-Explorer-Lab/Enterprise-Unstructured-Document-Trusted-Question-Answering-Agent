CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS pdf_documents (
    id BIGSERIAL PRIMARY KEY,
    doc_id TEXT NOT NULL UNIQUE,
    collection_name TEXT NOT NULL,
    doc_source TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    doc_hash TEXT NOT NULL DEFAULT '',
    page_count INTEGER NOT NULL DEFAULT 0,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS pdf_chunks (
    id BIGSERIAL PRIMARY KEY,
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
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    embedding vector(1024),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE pdf_chunks
    ADD COLUMN IF NOT EXISTS embedding vector(1024);

CREATE INDEX IF NOT EXISTS idx_pdf_documents_collection
    ON pdf_documents(collection_name, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_pdf_chunks_collection
    ON pdf_chunks(collection_name, doc_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_pdf_chunks_embedding_hnsw
    ON pdf_chunks USING hnsw (embedding vector_cosine_ops);
