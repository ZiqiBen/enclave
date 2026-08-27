"""Database connection and migrations.

Deliberately tiny: raw psycopg plus .sql files, no ORM. The interesting
part of this project is a hand-written hybrid retrieval query, and an ORM
would only get in its way.
"""

from __future__ import annotations

from pathlib import Path

import psycopg

from enclave.config import settings

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "sql"

_MIGRATION_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename   text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);
"""


def connect(url: str | None = None, *, autocommit: bool = True) -> psycopg.Connection:
    """Open a connection to the configured database."""
    return psycopg.connect(url or settings().database_url, autocommit=autocommit)


def migrate(conn: psycopg.Connection | None = None) -> list[str]:
    """Apply every .sql file in sql/ that has not run yet.

    Idempotent, so it is safe on every deploy. Returns the filenames
    applied this time, so callers can log or assert on it.

    The compose file also mounts sql/ into the Postgres entrypoint, but
    that only fires on an empty data directory -- this is the path that
    works against an existing volume.
    """
    own = conn is None
    conn = conn or connect()
    applied: list[str] = []
    try:
        with conn.cursor() as cur:
            cur.execute(_MIGRATION_TABLE)
            cur.execute("SELECT filename FROM schema_migrations")
            done = {row[0] for row in cur.fetchall()}

            for path in sorted(SCHEMA_DIR.glob("*.sql")):
                if path.name in done:
                    continue
                cur.execute(path.read_text(encoding="utf-8"))
                cur.execute(
                    "INSERT INTO schema_migrations (filename) VALUES (%s)",
                    (path.name,),
                )
                applied.append(path.name)
    finally:
        if own:
            conn.close()
    return applied


def cli_migrate() -> None:
    """Entry point for `uv run enclave-migrate`."""
    applied = migrate()
    print(f"applied: {', '.join(applied)}" if applied else "schema already up to date")


def reset(conn: psycopg.Connection) -> None:
    """Drop everything. Test fixtures only -- never call from app code."""
    with conn.cursor() as cur:
        cur.execute(
            "DROP TABLE IF EXISTS ingest_jobs, feedback, chunks, documents, "
            "eval_runs, schema_migrations CASCADE"
        )
