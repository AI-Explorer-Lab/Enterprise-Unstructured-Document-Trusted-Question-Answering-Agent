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
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS qa_sessions (
    session_id TEXT PRIMARY KEY,
    collection_name TEXT NOT NULL,
    user_id TEXT NOT NULL DEFAULT '',
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS qa_messages (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    query_type TEXT NOT NULL DEFAULT '',
    question TEXT NOT NULL DEFAULT '',
    answer TEXT NOT NULL DEFAULT '',
    decision TEXT NOT NULL DEFAULT '',
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
    citations_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    evidence_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    retrieval_trace_id TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS retrieval_traces (
    trace_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL DEFAULT '',
    message_id TEXT NOT NULL DEFAULT '',
    collection_name TEXT NOT NULL DEFAULT '',
    question TEXT NOT NULL DEFAULT '',
    expanded_queries_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    retrieval_trace_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    rerank_trace_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    selected_candidates_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS evaluation_records (
    evaluation_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL DEFAULT '',
    message_id TEXT NOT NULL DEFAULT '',
    metrics_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    notes TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pdf_documents_collection
    ON pdf_documents(collection_name, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_pdf_chunks_collection
    ON pdf_chunks(collection_name, doc_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_qa_sessions_collection
    ON qa_sessions(collection_name, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_qa_messages_session
    ON qa_messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_retrieval_traces_session
    ON retrieval_traces(session_id, created_at);
