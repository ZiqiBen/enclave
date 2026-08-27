from __future__ import annotations

import hashlib

import pytest

from enclave.ingest.jobs import (
    create_job,
    delete_job,
    fail_interrupted_jobs,
    get_job,
    list_jobs,
    safe_filename,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../private/report.pdf", "report.pdf"),
        ("notes?.md", "notes_.md"),
        ("  handbook.txt  ", "handbook.txt"),
    ],
)
def test_safe_filename_flattens_paths_and_preserves_supported_suffix(raw, expected):
    assert safe_filename(raw) == expected


@pytest.mark.parametrize("raw", ["", "payload.exe", "archive.zip", "..."])
def test_safe_filename_rejects_empty_or_unsupported_names(raw):
    with pytest.raises(ValueError):
        safe_filename(raw)


@pytest.mark.db
def test_job_lifecycle_and_delete_remove_document_chunks(db_conn, tmp_path):
    job_dir = tmp_path / "job-1"
    job_dir.mkdir()
    stored = job_dir / "job-1_notes.md"
    stored.write_text("# Notes\nPrivate content", encoding="utf-8")
    create_job(db_conn, "job-1", "notes.md", stored)

    assert get_job(db_conn, "job-1").status == "queued"
    assert [job.job_id for job in list_jobs(db_conn)] == ["job-1"]
    with pytest.raises(RuntimeError, match="while it is importing"):
        delete_job(db_conn, "job-1", tmp_path)

    content = "Private content"
    with db_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO documents (doc_id, source_path, doc_type) "
            "VALUES ('uploaded-doc', 'job-1_notes.md', 'markdown')"
        )
        cur.execute(
            "INSERT INTO chunks (doc_id, ordinal, content, content_hash) "
            "VALUES ('uploaded-doc', 0, %s, %s)",
            (content, hashlib.sha256(content.encode()).hexdigest()),
        )
        cur.execute(
            "UPDATE ingest_jobs SET status='complete', progress=100, "
            "doc_id='uploaded-doc', chunks_written=1 WHERE job_id='job-1'"
        )

    assert delete_job(db_conn, "job-1", tmp_path) is True
    assert get_job(db_conn, "job-1") is None
    assert not job_dir.exists()
    with db_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks WHERE doc_id='uploaded-doc'")
        assert cur.fetchone()[0] == 0


@pytest.mark.db
def test_delete_missing_job_returns_false(db_conn, tmp_path):
    assert delete_job(db_conn, "missing", tmp_path) is False


@pytest.mark.db
def test_incomplete_jobs_are_failed_after_restart(db_conn, tmp_path):
    stored = tmp_path / "queued.md"
    create_job(db_conn, "interrupted", "queued.md", stored)

    assert fail_interrupted_jobs(db_conn) == 1

    job = get_job(db_conn, "interrupted")
    assert job.status == "failed"
    assert job.progress == 100
    assert "service restart" in job.error
