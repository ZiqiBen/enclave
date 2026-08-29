from __future__ import annotations

import pytest

from enclave.history import (
    conversation_exists,
    delete_conversation,
    get_conversation,
    list_conversations,
    save_exchange,
)


@pytest.mark.db
def test_conversation_lifecycle_persists_metadata_and_cascades(db_conn):
    identifier = save_exchange(
        db_conn,
        conversation_id=None,
        query="What is PostgreSQL?",
        answer="A relational database [E1].",
        metadata={"verified": True, "evidence": [{"evidence_id": "E1"}]},
    )

    assert conversation_exists(db_conn, identifier) is True
    summaries = list_conversations(db_conn)
    assert len(summaries) == 1
    assert summaries[0].title == "What is PostgreSQL?"
    assert summaries[0].message_count == 2
    assert summaries[0].preview == "A relational database [E1]."

    messages = get_conversation(db_conn, identifier)
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[1].metadata["verified"] is True
    assert messages[1].metadata["evidence"][0]["evidence_id"] == "E1"

    save_exchange(
        db_conn,
        conversation_id=identifier,
        query="And what is a cluster?",
        answer="A group of databases [E1].",
        metadata={"verified": True},
    )
    assert len(get_conversation(db_conn, identifier)) == 4
    assert list_conversations(db_conn)[0].preview == "A group of databases [E1]."

    assert delete_conversation(db_conn, identifier) is True
    assert get_conversation(db_conn, identifier) is None
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM conversation_messages WHERE conversation_id=%s",
            (identifier,),
        )
        assert cur.fetchone()[0] == 0


@pytest.mark.db
def test_continuing_missing_conversation_fails(db_conn):
    with pytest.raises(KeyError):
        save_exchange(
            db_conn,
            conversation_id="missing",
            query="Question",
            answer="Answer",
            metadata={},
        )


@pytest.mark.db
def test_delete_missing_conversation_returns_false(db_conn):
    assert delete_conversation(db_conn, "missing") is False
