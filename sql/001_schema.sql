-- Enclave schema: lexical and dense retrieval in one store.
--
-- IMPORTANT: the vector dimension below must match ENCLAVE_EMBED_DIM.
-- Changing it requires re-embedding the corpus and rebuilding the HNSW
-- index, so pick one dimension per deployment. Default is 256, the
-- Matryoshka-truncated size used by the `portable` profile.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    doc_id       text PRIMARY KEY,
    source_path  text        NOT NULL,
    title        text,
    doc_type     text        NOT NULL,          -- markdown | html | pdf | text
    ingested_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id            bigserial PRIMARY KEY,
    doc_id        text        NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    ordinal       int         NOT NULL,          -- position within the document
    heading_path  text,                          -- e.g. 'Postgres > Indexes > GIN'
    content       text        NOT NULL,
    content_hash  text        NOT NULL,          -- sha256, drives idempotent ingest
    token_count   int,
    embedding     vector(256),                   -- keep in sync with ENCLAVE_EMBED_DIM
    tsv           tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at    timestamptz NOT NULL DEFAULT now(),
    UNIQUE (doc_id, ordinal)
);

-- Idempotent ingestion: re-running a job must not duplicate chunks.
CREATE UNIQUE INDEX IF NOT EXISTS chunks_content_hash_idx ON chunks (content_hash);

-- Lexical retrieval.
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (tsv);

-- Dense retrieval. HNSW build is memory-hungry; for large corpora raise
-- maintenance_work_mem for the session before creating this index.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Relevance feedback. This table is the raw material for hard-negative
-- mining in Sprint 4, so capture the rank the passage was shown at.
CREATE TABLE IF NOT EXISTS feedback (
    id            bigserial PRIMARY KEY,
    query         text        NOT NULL,
    chunk_id      bigint      NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    shown_rank    int         NOT NULL,
    relevant      boolean     NOT NULL,
    stage         text        NOT NULL,          -- retrieval | rerank
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS feedback_query_idx ON feedback (query);

-- Benchmark runs are versioned in-repo as well (see benchmarks/), but
-- keeping them queryable makes regression checks easy to script.
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id       text PRIMARY KEY,
    profile      text        NOT NULL,           -- portable | accelerated
    dataset      text        NOT NULL,           -- scifact | nfcorpus | fiqa | ...
    config       jsonb       NOT NULL,           -- models, rerank depth, dims, quantization
    metrics      jsonb       NOT NULL,           -- {"ndcg@10": 0.0, "recall@100": 0.0, ...}
    platform     text        NOT NULL,           -- win32-cuda | darwin-mps | linux-cpu
    created_at   timestamptz NOT NULL DEFAULT now()
);
