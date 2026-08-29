"""Persistent local conversation history."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from psycopg.types.json import Jsonb


@dataclass(frozen=True, slots=True)
class ConversationSummary:
    conversation_id: str
    title: str
    message_count: int
    preview: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ConversationMessage:
    id: int
    role: str
    content: str
    metadata: dict
    created_at: datetime


def conversation_exists(
    conn, conversation_id: str, owner_id: str | None = None
) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT EXISTS(SELECT 1 FROM conversations WHERE conversation_id=%s "
            "AND (%s::text IS NULL OR owner_id=%s::text))",
            (conversation_id, owner_id, owner_id),
        )
        return bool(cur.fetchone()[0])


def save_exchange(
    conn,
    *,
    conversation_id: str | None,
    query: str,
    answer: str,
    metadata: dict,
    owner_id: str | None = None,
) -> str:
    """Atomically create/continue a conversation and save both messages."""
    identifier = conversation_id or uuid.uuid4().hex
    with conn.transaction(), conn.cursor() as cur:
        if conversation_id is None:
            title = query.strip().replace("\n", " ")[:80]
            cur.execute(
                "INSERT INTO conversations (conversation_id, title, owner_id) "
                "VALUES (%s, %s, %s)",
                (identifier, title, owner_id),
            )
        else:
            cur.execute(
                "SELECT 1 FROM conversations WHERE conversation_id=%s "
                "AND (%s::text IS NULL OR owner_id=%s::text) FOR UPDATE",
                (identifier, owner_id, owner_id),
            )
            if cur.fetchone() is None:
                raise KeyError(identifier)
        cur.execute(
            "INSERT INTO conversation_messages (conversation_id, role, content) "
            "VALUES (%s, 'user', %s)",
            (identifier, query),
        )
        cur.execute(
            "INSERT INTO conversation_messages "
            "(conversation_id, role, content, metadata) "
            "VALUES (%s, 'assistant', %s, %s)",
            (identifier, answer, Jsonb(metadata)),
        )
        cur.execute(
            "UPDATE conversations SET updated_at=now() WHERE conversation_id=%s",
            (identifier,),
        )
    return identifier


def list_conversations(conn, owner_id: str | None = None) -> list[ConversationSummary]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT c.conversation_id, c.title, count(m.id), latest.content, "
            "c.created_at, c.updated_at "
            "FROM conversations c LEFT JOIN conversation_messages m "
            "ON m.conversation_id=c.conversation_id "
            "LEFT JOIN LATERAL (SELECT content FROM conversation_messages "
            "WHERE conversation_id=c.conversation_id AND role='assistant' "
            "ORDER BY id DESC LIMIT 1) latest ON true "
            "WHERE (%s::text IS NULL OR c.owner_id=%s::text) "
            "GROUP BY c.conversation_id, latest.content "
            "ORDER BY c.updated_at DESC",
            (owner_id, owner_id),
        )
        return [ConversationSummary(*row) for row in cur.fetchall()]


def get_conversation(
    conn, conversation_id: str, owner_id: str | None = None
) -> list[ConversationMessage] | None:
    if not conversation_exists(conn, conversation_id, owner_id):
        return None
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, role, content, metadata, created_at "
            "FROM conversation_messages WHERE conversation_id=%s ORDER BY id",
            (conversation_id,),
        )
        return [ConversationMessage(*row) for row in cur.fetchall()]


def user_queries(
    conn, conversation_id: str, limit: int = 8, owner_id: str | None = None
) -> list[str]:
    """Return recent user questions in chronological order for context repair."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT content FROM (SELECT id, content FROM conversation_messages "
            "WHERE conversation_id=%s AND role='user' "
            "AND EXISTS (SELECT 1 FROM conversations c WHERE "
            "c.conversation_id=conversation_messages.conversation_id "
            "AND (%s::text IS NULL OR c.owner_id=%s::text)) "
            "ORDER BY id DESC LIMIT %s) q "
            "ORDER BY id",
            (conversation_id, owner_id, owner_id, limit),
        )
        return [row[0] for row in cur.fetchall()]


def delete_conversation(
    conn, conversation_id: str, owner_id: str | None = None
) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM conversations WHERE conversation_id=%s "
            "AND (%s::text IS NULL OR owner_id=%s::text)",
            (conversation_id, owner_id, owner_id),
        )
        return bool(cur.rowcount)
