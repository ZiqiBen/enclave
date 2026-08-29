CREATE TABLE IF NOT EXISTS users (
    user_id        text PRIMARY KEY,
    username       text        NOT NULL,
    password_hash  text        NOT NULL,
    is_admin       boolean     NOT NULL DEFAULT false,
    disabled       boolean     NOT NULL DEFAULT false,
    created_at     timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT users_username_normalized CHECK (username = lower(username))
);

CREATE UNIQUE INDEX IF NOT EXISTS users_username_idx ON users (username);

CREATE TABLE IF NOT EXISTS auth_sessions (
    token_hash   text PRIMARY KEY,
    user_id      text        NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    expires_at   timestamptz NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS auth_sessions_expiry_idx ON auth_sessions (expires_at);

CREATE TABLE IF NOT EXISTS auth_login_attempts (
    id            bigserial PRIMARY KEY,
    username      text        NOT NULL,
    source         text        NOT NULL,
    succeeded      boolean     NOT NULL,
    attempted_at   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS auth_login_attempts_lookup_idx
    ON auth_login_attempts (username, source, attempted_at DESC);

ALTER TABLE documents ADD COLUMN IF NOT EXISTS owner_id text
    REFERENCES users(user_id) ON DELETE CASCADE;
ALTER TABLE ingest_jobs ADD COLUMN IF NOT EXISTS owner_id text
    REFERENCES users(user_id) ON DELETE CASCADE;
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS owner_id text
    REFERENCES users(user_id) ON DELETE CASCADE;
ALTER TABLE feedback ADD COLUMN IF NOT EXISTS owner_id text
    REFERENCES users(user_id) ON DELETE CASCADE;

CREATE INDEX IF NOT EXISTS documents_owner_idx ON documents (owner_id);
CREATE INDEX IF NOT EXISTS ingest_jobs_owner_created_idx
    ON ingest_jobs (owner_id, created_at DESC);
CREATE INDEX IF NOT EXISTS conversations_owner_updated_idx
    ON conversations (owner_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS feedback_owner_created_idx
    ON feedback (owner_id, created_at DESC);

-- The old global hash constraint prevented two users from importing the same
-- content. Deduplication is document-local in a multi-tenant system.
DROP INDEX IF EXISTS chunks_content_hash_idx;
CREATE UNIQUE INDEX IF NOT EXISTS chunks_document_content_hash_idx
    ON chunks (doc_id, content_hash);
