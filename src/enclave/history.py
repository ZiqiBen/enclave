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


def conversation_exists(conn, conversation_id: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT EXISTS(SELECT 1 FROM conversations WHERE conversation_id=%s)",
            (conversation_id,),
        )
        return bool(cur.fetchone()[0])


def save_exchange(
    conn,
    *,
    conversation_id: str | None,
    query: str,
    answer: str,
    metadata: dict,
) -> str:
    """Atomically create/continue a conversation and save both messages."""
    identifier = conversation_id or uuid.uuid4().hex
    with conn.transaction(), conn.cursor() as cur:
        if conversation_id is None:
            title = query.strip().replace("\n", " ")[:80]
            cur.execute(
                "INSERT INTO conversations (conversation_id, title) VALUES (%s, %s)",
                (identifier, title),
            )
        else:
            cur.execute(
                "SELECT 1 FROM conversations WHERE conversation_id=%s FOR UPDATE",
                (identifier,),
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


def list_conversations(conn) -> list[ConversationSummary]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT c.conversation_id, c.title, count(m.id), latest.content, "
            "c.created_at, c.updated_at "
            "FROM conversations c LEFT JOIN conversation_messages m "
            "ON m.conversation_id=c.conversation_id "
            "LEFT JOIN LATERAL (SELECT content FROM conversation_messages "
            "WHERE conversation_id=c.conversation_id AND role='assistant' "
            "ORDER BY id DESC LIMIT 1) latest ON true "
            "GROUP BY c.conversation_id, latest.content "
            "ORDER BY c.updated_at DESC"
        )
        return [ConversationSummary(*row) for row in cur.fetchall()]


def get_conversation(conn, conversation_id: str) -> list[ConversationMessage] | None:
    if not conversation_exists(conn, conversation_id):
        return None
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, role, content, metadata, created_at "
            "FROM conversation_messages WHERE conversation_id=%s ORDER BY id",
            (conversation_id,),
        )
        return [ConversationMessage(*row) for row in cur.fetchall()]


def delete_conversation(conn, conversation_id: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM conversations WHERE conversation_id=%s", (conversation_id,)
        )
        return bool(cur.rowcount)
