"""Persistent background ingestion jobs for locally uploaded documents."""

from __future__ import annotations

import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from enclave import db
from enclave.ingest.parsers import SUPPORTED_SUFFIXES
from enclave.ingest.pipeline import document_id, ingest_path

_UNSAFE_FILENAME = re.compile(r"[^\w.() -]", re.UNICODE)


@dataclass(frozen=True, slots=True)
class IngestJob:
    job_id: str
    filename: str
    status: str
    progress: int
    doc_id: str | None
    chunks_written: int
    error: str | None
    created_at: object
    updated_at: object


def safe_filename(raw: str) -> str:
    """Return a flat, readable filename that cannot escape the upload root."""
    name = unicodedata.normalize("NFKC", Path(raw or "").name).strip()
    name = _UNSAFE_FILENAME.sub("_", name).strip(" .")
    if not name:
        raise ValueError("filename is empty")
    if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError("unsupported document type")
    return name[:180]


def create_job(conn, job_id: str, filename: str, stored_path: Path) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO ingest_jobs (job_id, filename, stored_path, status) "
            "VALUES (%s, %s, %s, 'queued')",
            (job_id, filename, str(stored_path)),
        )


def _set_status(
    conn,
    job_id: str,
    status: str,
    progress: int,
    *,
    doc_id: str | None = None,
    chunks_written: int = 0,
    error: str | None = None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE ingest_jobs SET status=%s, progress=%s, doc_id=%s, "
            "chunks_written=%s, error=%s, updated_at=now() WHERE job_id=%s",
            (status, progress, doc_id, chunks_written, error, job_id),
        )


def run_ingest_job(job_id: str, stored_path: Path, database_url: str) -> None:
    """Parse and embed one upload in a worker thread, persisting every state."""
    with db.connect(database_url) as conn:
        try:
            _set_status(conn, job_id, "parsing", 10)
            from enclave.models.encoders import get_embedder

            _set_status(conn, job_id, "embedding", 35)
            stats = ingest_path(stored_path, conn=conn, embedder=get_embedder())
            if stats.failures or not stats.files_ingested:
                raise ValueError("document did not produce ingestible content")
            _set_status(
                conn,
                job_id,
                "complete",
                100,
                doc_id=document_id(stored_path.name),
                chunks_written=stats.chunks_written,
            )
        except Exception as exc:
            _set_status(conn, job_id, "failed", 100, error=str(exc)[:500])


def _rows(conn, *, job_id: str | None = None) -> list[IngestJob]:
    sql = (
        "SELECT job_id, filename, status, progress, doc_id, chunks_written, "
        "error, created_at, updated_at FROM ingest_jobs"
    )
    params = ()
    if job_id is not None:
        sql += " WHERE job_id = %s"
        params = (job_id,)
    sql += " ORDER BY created_at DESC"
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return [IngestJob(*row) for row in cur.fetchall()]


def list_jobs(conn) -> list[IngestJob]:
    return _rows(conn)


def get_job(conn, job_id: str) -> IngestJob | None:
    rows = _rows(conn, job_id=job_id)
    return rows[0] if rows else None


def fail_interrupted_jobs(conn) -> int:
    """Make abandoned in-process jobs visible after an application restart."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE ingest_jobs SET status='failed', progress=100, "
            "error='import interrupted by service restart', updated_at=now() "
            "WHERE status IN ('queued', 'parsing', 'embedding')"
        )
        return cur.rowcount


def delete_job(conn, job_id: str, upload_root: Path) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT stored_path, status, doc_id FROM ingest_jobs WHERE job_id=%s",
            (job_id,),
        )
        row = cur.fetchone()
        if row is None:
            return False
        stored_path, status, doc_id = row
        if status in {"queued", "parsing", "embedding"}:
            raise RuntimeError("cannot delete a document while it is importing")
        if doc_id:
            cur.execute("DELETE FROM documents WHERE doc_id=%s", (doc_id,))
        cur.execute("DELETE FROM ingest_jobs WHERE job_id=%s", (job_id,))

    root = upload_root.resolve()
    path = Path(stored_path).resolve()
    if path.is_relative_to(root) and path.parent != root:
        shutil.rmtree(path.parent, ignore_errors=True)
    return True
