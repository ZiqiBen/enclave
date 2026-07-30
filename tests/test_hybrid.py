"""Hybrid retrieval: the SQL, and the RRF arithmetic inside it.

Marked `db` because it needs a real Postgres with pgvector -- the fusion is
a SQL statement, so testing it against a mock would test nothing. It does
NOT need model weights: `fake_embedder` supplies deterministic vectors,
which is why this can run on a CI runner in seconds.
"""

from __future__ import annotations

import pytest

from enclave.retrieval.hybrid import hybrid_search, to_vector_literal

pytestmark = pytest.mark.db


class TestVectorLiteral:
    def test_uses_pgvector_text_format(self):
        assert to_vector_literal([1.0, 0.0, -0.5]).startswith("[")
        assert to_vector_literal([1.0, 0.0, -0.5]).endswith("]")
        assert to_vector_literal([1.0, 2.0]) == "[1.000000,2.000000]"

    def test_accepts_a_numpy_array(self):
        np = pytest.importorskip("numpy")
        assert to_vector_literal(np.array([0.25, 0.5], dtype="float32")) == (
            "[0.250000,0.500000]"
        )


class TestHybridSearch:
    def test_returns_candidates_from_both_channels(
        self, db_conn, seeded_corpus, fake_embedder
    ):
        """All three synthetic chunks should surface: two via lexical match,
        and the cooking one via the dense channel alone."""
        fake_embedder.query_vectors[seeded_corpus["query"]] = seeded_corpus[
            "query_vector"
        ]
        results = hybrid_search(db_conn, seeded_corpus["query"], limit=10)

        assert {c.doc_id for c in results} == {"pg-both", "pg-lexical", "cooking"}

    def test_channel_membership_is_reported_correctly(
        self, db_conn, seeded_corpus, fake_embedder
    ):
        """lex_rank / dense_rank drive hard-negative mining later, so a wrong
        NULL here silently corrupts Sprint 4's training data."""
        fake_embedder.query_vectors[seeded_corpus["query"]] = seeded_corpus[
            "query_vector"
        ]
        by_doc = {c.doc_id: c for c in hybrid_search(db_conn, seeded_corpus["query"])}

        # Lexical match on 'postgres gin index', and closest vector.
        assert by_doc["pg-both"].lex_rank is not None
        assert by_doc["pg-both"].dense_rank is not None
        assert by_doc["pg-both"].found_by_both is True

        # Lexical match, but its vector is orthogonal to the query.
        assert by_doc["pg-lexical"].lex_rank is not None

        # No shared terms with the query -- dense channel only.
        assert by_doc["cooking"].lex_rank is None
        assert by_doc["cooking"].dense_rank is not None
        assert by_doc["cooking"].found_by_both is False

    def test_agreement_outranks_a_single_channel_hit(
        self, db_conn, seeded_corpus, fake_embedder
    ):
        """The core property of RRF: two channels agreeing beats one channel
        shouting. If this inverts, fusion is broken."""
        fake_embedder.query_vectors[seeded_corpus["query"]] = seeded_corpus[
            "query_vector"
        ]
        results = hybrid_search(db_conn, seeded_corpus["query"])

        assert results[0].doc_id == "pg-both"
        assert results[-1].doc_id == "cooking"

    def test_scores_are_descending_and_match_the_rrf_formula(
        self, db_conn, seeded_corpus, fake_embedder
    ):
        fake_embedder.query_vectors[seeded_corpus["query"]] = seeded_corpus[
            "query_vector"
        ]
        results = hybrid_search(db_conn, seeded_corpus["query"])
        scores = [c.fusion_score for c in results]
        assert scores == sorted(scores, reverse=True)

        rrf_k = 60
        for c in results:
            expected = 0.0
            if c.lex_rank is not None:
                expected += 1.0 / (rrf_k + c.lex_rank)
            if c.dense_rank is not None:
                expected += 1.0 / (rrf_k + c.dense_rank)
            assert c.fusion_score == pytest.approx(expected, rel=1e-6)

    def test_limit_is_respected(self, db_conn, seeded_corpus, fake_embedder):
        fake_embedder.query_vectors[seeded_corpus["query"]] = seeded_corpus[
            "query_vector"
        ]
        assert len(hybrid_search(db_conn, seeded_corpus["query"], limit=2)) == 2

    def test_punctuation_does_not_raise(self, db_conn, seeded_corpus, fake_embedder):
        """websearch_to_tsquery is used precisely so raw user input cannot
        blow up the query the way to_tsquery does."""
        for q in ['"postgres gin"', "postgres OR index", "postgres -gin", "!!!", ""]:
            hybrid_search(db_conn, q, limit=5)

    def test_chunks_without_embeddings_are_skipped_by_the_dense_channel(
        self, db_conn, seeded_corpus, fake_embedder
    ):
        """Ingestion writes content first and embeddings in a later batch, so
        half-ingested rows exist in practice and must not break search."""
        with db_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (doc_id, source_path, doc_type) "
                "VALUES ('pending', 'synthetic/pending.md', 'markdown')"
            )
            cur.execute(
                "INSERT INTO chunks (doc_id, ordinal, content, content_hash) "
                "VALUES ('pending', 0, 'postgres gin index pending', 'hash-pending')"
            )

        fake_embedder.query_vectors[seeded_corpus["query"]] = seeded_corpus[
            "query_vector"
        ]
        by_doc = {c.doc_id: c for c in hybrid_search(db_conn, seeded_corpus["query"])}

        # Still findable lexically, but absent from the dense ranking.
        assert "pending" in by_doc
        assert by_doc["pending"].dense_rank is None


class TestSchema:
    def test_migration_is_idempotent(self, db_conn):
        """`enclave-migrate` runs on every deploy, so a second run must be a
        no-op rather than an error."""
        from enclave import db

        assert db.migrate(db_conn) == []

    def test_content_hash_is_unique(self, db_conn, seeded_corpus):
        """Idempotent ingestion depends on this constraint existing."""
        psycopg = pytest.importorskip("psycopg")
        with pytest.raises(psycopg.errors.UniqueViolation), db_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chunks (doc_id, ordinal, content, content_hash) "
                "SELECT doc_id, 99, content, content_hash FROM chunks LIMIT 1"
            )
