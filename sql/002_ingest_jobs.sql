CREATE TABLE IF NOT EXISTS ingest_jobs (
    job_id          text PRIMARY KEY,
    filename        text        NOT NULL,
    stored_path     text        NOT NULL,
    status          text        NOT NULL CHECK (
        status IN ('queued', 'parsing', 'embedding', 'complete', 'failed')
    ),
    progress        int         NOT NULL DEFAULT 0 CHECK (
        progress BETWEEN 0 AND 100
    ),
    doc_id          text REFERENCES documents(doc_id) ON DELETE SET NULL,
    chunks_written  int         NOT NULL DEFAULT 0,
    error           text,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ingest_jobs_created_idx
    ON ingest_jobs (created_at DESC);
