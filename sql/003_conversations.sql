CREATE TABLE IF NOT EXISTS conversations (
    conversation_id text PRIMARY KEY,
    title           text        NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS conversation_messages (
    id               bigserial PRIMARY KEY,
    conversation_id  text        NOT NULL REFERENCES conversations(conversation_id)
                                  ON DELETE CASCADE,
    role             text        NOT NULL CHECK (role IN ('user', 'assistant')),
    content          text        NOT NULL,
    metadata         jsonb       NOT NULL DEFAULT '{}'::jsonb,
    created_at       timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS conversations_updated_idx
    ON conversations (updated_at DESC);
CREATE INDEX IF NOT EXISTS conversation_messages_order_idx
    ON conversation_messages (conversation_id, id);
