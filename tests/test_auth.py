"""Authentication and tenant isolation security boundaries."""

from __future__ import annotations

import hashlib
from contextlib import nullcontext

import pytest
from fastapi.testclient import TestClient

from enclave.api.main import Services, create_app
from enclave.auth import create_user, hash_password, verify_password
from enclave.history import list_conversations, save_exchange
from enclave.retrieval.hybrid import lexical_search


def test_password_hash_round_trip_and_rejects_short_passwords():
    encoded = hash_password("correct horse battery staple")
    assert "correct horse" not in encoded
    assert verify_password("correct horse battery staple", encoded)
    assert not verify_password("wrong password", encoded)
    with pytest.raises(ValueError, match="12 characters"):
        hash_password("too-short")


@pytest.mark.db
def test_authentication_cookie_and_protected_endpoint(db_conn):
    create_user(db_conn, "alice", "alice-password-123")
    app = create_app(
        Services(connect=lambda: nullcontext(db_conn)), require_auth=True
    )
    with TestClient(app) as client:
        assert client.get("/v1/auth/me").status_code == 401
        assert client.get("/v1/documents").status_code == 401
        assert client.post(
            "/v1/auth/login",
            json={"username": "alice", "password": "wrong-password"},
        ).status_code == 401
        response = client.post(
            "/v1/auth/login",
            json={"username": "Alice", "password": "alice-password-123"},
        )
        assert response.status_code == 200
        assert response.json() == {"username": "alice", "is_admin": False}
        assert "HttpOnly" in response.headers["set-cookie"]
        assert "SameSite=strict" in response.headers["set-cookie"]
        assert client.get("/v1/documents").json() == []
        assert client.post("/v1/auth/logout").status_code == 204
        assert client.get("/v1/auth/me").status_code == 401


@pytest.mark.db
def test_two_users_cannot_read_each_others_search_or_conversations(db_conn):
    alice = create_user(db_conn, "alice", "alice-password-123")
    bob = create_user(db_conn, "bob", "bob-password-456")
    with db_conn.cursor() as cur:
        for user, key, content in (
            (alice, "alice-doc", "orchid private project notes"),
            (bob, "bob-doc", "orchid confidential budget"),
        ):
            cur.execute(
                "INSERT INTO documents "
                "(doc_id, source_path, title, doc_type, owner_id) "
                "VALUES (%s, %s, %s, 'text', %s)",
                (key, f"{key}.txt", key, user.user_id),
            )
            cur.execute(
                "INSERT INTO chunks "
                "(doc_id, ordinal, content, content_hash, token_count) "
                "VALUES (%s, 0, %s, %s, %s)",
                (key, content, hashlib.sha256(content.encode()).hexdigest(), 4),
            )
    save_exchange(
        db_conn,
        conversation_id=None,
        query="Alice question",
        answer="Alice answer",
        metadata={},
        owner_id=alice.user_id,
    )

    assert [item.doc_id for item in lexical_search(
        db_conn, "orchid", owner_id=alice.user_id
    )] == ["alice-doc"]
    assert [item.doc_id for item in lexical_search(
        db_conn, "orchid", owner_id=bob.user_id
    )] == ["bob-doc"]
    assert len(list_conversations(db_conn, alice.user_id)) == 1
    assert list_conversations(db_conn, bob.user_id) == []
