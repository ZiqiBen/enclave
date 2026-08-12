"""Document parsing, chunking, discovery, and persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from enclave.ingest.chunking import chunk_document
from enclave.ingest.parsers import ParsedDocument, Section, parse_document
from enclave.ingest.pipeline import discover_documents, ingest_path


class TestParsing:
    def test_markdown_preserves_nested_headings(self, tmp_path: Path):
        path = tmp_path / "guide.md"
        path.write_text(
            "# Database Guide\nIntro text.\n## Indexes\nGIN details.\n"
            "### Tuning\nTune it.",
            encoding="utf-8",
        )

        document = parse_document(path)

        assert document.title == "Database Guide"
        assert [section.heading_path for section in document.sections] == [
            "Database Guide",
            "Database Guide > Indexes",
            "Database Guide > Indexes > Tuning",
        ]
        assert document.sections[1].text == "GIN details."

    def test_html_ignores_scripts_and_keeps_headings(self, tmp_path: Path):
        path = tmp_path / "guide.html"
        path.write_text(
            "<html><head><title>Guide</title><script>secret()</script></head>"
            "<body><h1>Indexes</h1><p>Use GIN.</p><h2>Tuning</h2>"
            "<p>Measure first.</p></body></html>",
            encoding="utf-8",
        )

        document = parse_document(path)

        assert document.title == "Guide"
        assert document.sections[0] == Section("Indexes", "Use GIN.")
        assert document.sections[1] == Section("Indexes > Tuning", "Measure first.")
        assert all("secret" not in section.text for section in document.sections)

    def test_plain_text_uses_filename_as_title(self, tmp_path: Path):
        path = tmp_path / "policy.txt"
        path.write_text("Keep data local.\n\nNever upload it.", encoding="utf-8")
        document = parse_document(path)
        assert document.title == "policy"
        assert document.doc_type == "text"
        assert document.sections[0].text == "Keep data local.\n\nNever upload it."


class TestChunking:
    def test_chunks_are_bounded_overlapping_and_deterministic(self):
        document = ParsedDocument(
            "Example", "text", (Section("Part One", "one two three four five six"),)
        )

        first = chunk_document(document, max_words=4, overlap_words=1)
        second = chunk_document(document, max_words=4, overlap_words=1)

        assert [chunk.content for chunk in first] == [
            "one two three four",
            "four five six",
        ]
        assert [chunk.ordinal for chunk in first] == [0, 1]
        assert all(chunk.heading_path == "Part One" for chunk in first)
        assert [chunk.content_hash for chunk in first] == [
            chunk.content_hash for chunk in second
        ]

    @pytest.mark.parametrize(
        ("max_words", "overlap_words"), [(0, 0), (5, -1), (5, 5), (5, 6)]
    )
    def test_rejects_invalid_windows(self, max_words: int, overlap_words: int):
        document = ParsedDocument("Empty", "text", ())
        with pytest.raises(ValueError):
            chunk_document(document, max_words=max_words, overlap_words=overlap_words)


class TestDiscovery:
    def test_discovers_supported_files_recursively_in_stable_order(
        self, tmp_path: Path
    ):
        (tmp_path / "nested").mkdir()
        (tmp_path / "b.txt").write_text("b", encoding="utf-8")
        (tmp_path / "nested" / "a.md").write_text("a", encoding="utf-8")
        (tmp_path / "ignored.csv").write_text("x", encoding="utf-8")

        paths = discover_documents(tmp_path)

        assert [path.relative_to(tmp_path).as_posix() for path in paths] == [
            "b.txt",
            "nested/a.md",
        ]

    def test_rejects_unsupported_single_file(self, tmp_path: Path):
        path = tmp_path / "data.csv"
        path.write_text("a,b", encoding="utf-8")
        with pytest.raises(ValueError, match="unsupported"):
            discover_documents(path)


def test_embeddings_are_batched_across_documents(tmp_path: Path, monkeypatch):
    (tmp_path / "a.txt").write_text("alpha document", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta document", encoding="utf-8")

    class Embedder:
        calls: list[tuple[list[str], int]] = []

        def encode_documents(self, texts, batch_size=16):
            self.calls.append((texts, batch_size))
            return [[0.0] * 256 for _ in texts]

    class Connection:
        pass

    embedder = Embedder()
    from enclave.ingest import pipeline

    monkeypatch.setattr(
        pipeline,
        "_write_document",
        lambda *args, **kwargs: (len(kwargs["chunks"]), 0),
    )
    stats = ingest_path(tmp_path, conn=Connection(), embedder=embedder, batch_size=7)

    assert stats.files_ingested == 2
    assert len(embedder.calls) == 1
    assert embedder.calls[0] == (["alpha document", "beta document"], 7)


@pytest.mark.db
class TestPersistence:
    def test_ingests_and_replaces_a_document_idempotently(
        self, db_conn, fake_embedder, tmp_path: Path
    ):
        path = tmp_path / "guide.md"
        path.write_text("# Guide\n" + "word " * 12, encoding="utf-8")

        first = ingest_path(
            tmp_path,
            conn=db_conn,
            embedder=fake_embedder,
            max_words=5,
            overlap_words=1,
        )
        second = ingest_path(
            tmp_path,
            conn=db_conn,
            embedder=fake_embedder,
            max_words=5,
            overlap_words=1,
        )

        assert first.files_ingested == second.files_ingested == 1
        with db_conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM documents")
            assert cur.fetchone()[0] == 1
            cur.execute("SELECT count(*), count(embedding) FROM chunks")
            count, embedded = cur.fetchone()
        assert count == embedded == first.chunks_written

    def test_no_embed_leaves_chunks_available_for_lexical_search(
        self, db_conn, tmp_path: Path
    ):
        (tmp_path / "policy.txt").write_text(
            "Confidential documents stay local.", encoding="utf-8"
        )

        stats = ingest_path(tmp_path, conn=db_conn, embedder=None)

        assert stats.chunks_written == 1
        with db_conn.cursor() as cur:
            cur.execute("SELECT content, embedding IS NULL FROM chunks")
            assert cur.fetchone() == ("Confidential documents stay local.", True)
