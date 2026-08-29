"""Local password authentication with opaque, server-side sessions."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from getpass import getpass

import typer

from enclave import db

_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1


@dataclass(frozen=True, slots=True)
class User:
    user_id: str
    username: str
    is_admin: bool


def normalize_username(username: str) -> str:
    value = username.strip().lower()
    if not (3 <= len(value) <= 64):
        raise ValueError("username must be between 3 and 64 characters")
    if not all(character.isalnum() or character in "._-" for character in value):
        raise ValueError("username may only contain letters, numbers, ., _, and -")
    return value


def hash_password(password: str) -> str:
    if len(password) < 12:
        raise ValueError("password must contain at least 12 characters")
    salt = secrets.token_bytes(16)
    derived = hashlib.scrypt(
        password.encode(), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P
    )
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${derived.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, n, r, p, salt, expected = encoded.split("$", 5)
        if algorithm != "scrypt":
            return False
        actual = hashlib.scrypt(
            password.encode(),
            salt=bytes.fromhex(salt),
            n=int(n),
            r=int(r),
            p=int(p),
        )
        return hmac.compare_digest(actual, bytes.fromhex(expected))
    except (ValueError, TypeError):
        return False


def create_user(
    conn,
    username: str,
    password: str,
    *,
    is_admin: bool = False,
    claim_existing: bool = False,
) -> User:
    normalized = normalize_username(username)
    user = User(uuid.uuid4().hex, normalized, is_admin)
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(
            "INSERT INTO users (user_id, username, password_hash, is_admin) "
            "VALUES (%s, %s, %s, %s)",
            (user.user_id, user.username, hash_password(password), is_admin),
        )
        if claim_existing:
            for table in ("documents", "ingest_jobs", "conversations", "feedback"):
                cur.execute(
                    f"UPDATE {table} SET owner_id=%s WHERE owner_id IS NULL",  # noqa: S608
                    (user.user_id,),
                )
    return user


def authenticate(conn, username: str, password: str, source: str) -> User | None:
    try:
        normalized = normalize_username(username)
    except ValueError:
        normalized = username.strip().lower()[:64]
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM auth_login_attempts WHERE username=%s AND source=%s "
            "AND succeeded=false AND attempted_at > now() - interval '10 minutes'",
            (normalized, source),
        )
        if cur.fetchone()[0] >= 5:
            return None
        cur.execute(
            "SELECT user_id, username, password_hash, is_admin FROM users "
            "WHERE username=%s AND disabled=false",
            (normalized,),
        )
        row = cur.fetchone()
        valid = row is not None and verify_password(password, row[2])
        cur.execute(
            "INSERT INTO auth_login_attempts (username, source, succeeded) "
            "VALUES (%s, %s, %s)",
            (normalized, source, valid),
        )
    return User(row[0], row[1], row[3]) if valid else None


def create_session(conn, user_id: str, lifetime_hours: int) -> str:
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(UTC) + timedelta(hours=lifetime_hours)
    with conn.cursor() as cur:
        cur.execute("DELETE FROM auth_sessions WHERE expires_at <= now()")
        cur.execute(
            "INSERT INTO auth_sessions (token_hash, user_id, expires_at) "
            "VALUES (%s, %s, %s)",
            (_token_hash(token), user_id, expires_at),
        )
    return token


def session_user(conn, token: str | None) -> User | None:
    if not token:
        return None
    with conn.cursor() as cur:
        cur.execute(
            "SELECT u.user_id, u.username, u.is_admin FROM auth_sessions s "
            "JOIN users u ON u.user_id=s.user_id "
            "WHERE s.token_hash=%s AND s.expires_at > now() AND u.disabled=false",
            (_token_hash(token),),
        )
        row = cur.fetchone()
    return User(*row) if row else None


def delete_session(conn, token: str | None) -> None:
    if token:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM auth_sessions WHERE token_hash=%s",
                (_token_hash(token),),
            )


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _create_user_command(
    username: str = typer.Argument(...),
    admin: bool = typer.Option(False, "--admin"),
    claim_existing: bool = typer.Option(False, "--claim-existing"),
) -> None:
    """Create a local Enclave account without exposing the password in shell history."""
    password = getpass("Password (12+ characters): ")
    confirmation = getpass("Confirm password: ")
    if password != confirmation:
        raise typer.BadParameter("passwords do not match")
    db.migrate()
    with db.connect() as conn:
        user = create_user(
            conn, username, password, is_admin=admin, claim_existing=claim_existing
        )
    typer.echo(f"created user {user.username}")


def cli_create_user() -> None:
    """Package entry point for the local account-creation command."""
    typer.run(_create_user_command)
