"""Reranking orchestration without loading model weights."""

from __future__ import annotations

import pytest

from enclave.rank import rank_candidates, retrieve_and_rank
from enclave.retrieval.hybrid import Candidate


def candidate(
    chunk_id: int,
    fusion_score: float,
    *,
    lex_rank: int | None = 1,
    dense_rank: int | None = 1,
) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        heading_path=f"Section {chunk_id}",
        content=f"passage {chunk_id}",
        fusion_score=fusion_score,
        lex_rank=lex_rank,
        dense_rank=dense_rank,
    )


class FakeReranker:
    def __init__(self, scores: list[float]):
        self.scores = scores
        self.calls: list[tuple[str, list[str], int]] = []

    def score(self, query: str, documents: list[str], batch_size: int = 8):
        self.calls.append((query, documents, batch_size))
        return self.scores


def test_reranks_in_one_batch_and_returns_final_order():
    candidates = [candidate(1, 0.9), candidate(2, 0.8), candidate(3, 0.7)]
    reranker = FakeReranker([0.1, 0.95, 0.4])

    result = rank_candidates(
        "question", candidates, reranker=reranker, limit=2, batch_size=4, force=True
    )

    assert result.reranked is True
    assert result.retrieved_count == 3
    assert [item.chunk_id for item in result.candidates] == [2, 3]
    assert [item.rerank_score for item in result.candidates] == [0.95, 0.4]
    assert reranker.calls == (
        [("question", ["passage 1", "passage 2", "passage 3"], 4)]
    )


def test_skipped_rerank_preserves_rrf_order_and_does_not_load_model():
    candidates = [candidate(1, 0.9), candidate(2, 0.8)]

    result = rank_candidates("question", candidates, limit=1, force=False)

    assert result.reranked is False
    assert [item.chunk_id for item in result.candidates] == [1]
    assert result.candidates[0].rerank_score is None


def test_equal_scores_preserve_original_rrf_order():
    candidates = [candidate(1, 0.9), candidate(2, 0.8)]
    result = rank_candidates(
        "question", candidates, reranker=FakeReranker([0.5, 0.5]), force=True
    )
    assert [item.chunk_id for item in result.candidates] == [1, 2]


def test_empty_candidates_never_load_reranker():
    result = rank_candidates("question", [], force=True)
    assert result == result.__class__((), False, 0)


def test_rejects_wrong_score_count():
    candidates = [candidate(1, 0.9), candidate(2, 0.8)]
    with pytest.raises(ValueError, match="wrong number"):
        rank_candidates(
            "question", candidates, reranker=FakeReranker([0.5]), force=True
        )


@pytest.mark.parametrize(("limit", "batch_size"), [(0, 8), (10, 0)])
def test_validates_limits(limit: int, batch_size: int):
    with pytest.raises(ValueError):
        rank_candidates("question", [], limit=limit, batch_size=batch_size)


@pytest.mark.db
def test_retrieves_then_reranks_real_database_candidates(
    db_conn, seeded_corpus, fake_embedder
):
    fake_embedder.query_vectors[seeded_corpus["query"]] = seeded_corpus["query_vector"]
    reranker = FakeReranker([0.1, 0.9, 0.5])

    result = retrieve_and_rank(
        db_conn, seeded_corpus["query"], limit=2, reranker=reranker
    )

    assert result.reranked is True
    assert result.retrieved_count == 3
    assert [item.rerank_score for item in result.candidates] == [0.9, 0.5]
