"""Stage-two reranking orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from enclave.retrieval.hybrid import Candidate, hybrid_search, should_rerank


class CandidateReranker(Protocol):
    def score(
        self, query: str, documents: list[str], batch_size: int = 8
    ) -> list[float]: ...


@dataclass(frozen=True, slots=True)
class RankedEvidence:
    candidates: tuple[Candidate, ...]
    reranked: bool
    retrieved_count: int


def rank_candidates(
    query: str,
    candidates: list[Candidate],
    *,
    reranker: CandidateReranker | None = None,
    limit: int = 10,
    batch_size: int = 8,
    force: bool | None = None,
) -> RankedEvidence:
    """Conditionally rerank hybrid candidates and return final evidence.

    ``force`` exists for evaluation and tests: ``True`` always reranks and
    ``False`` never reranks. Normal application code leaves it as ``None``.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    retrieved_count = len(candidates)
    run_reranker = force if force is not None else should_rerank(candidates)
    if not candidates or not run_reranker:
        return RankedEvidence(tuple(candidates[:limit]), False, retrieved_count)

    if reranker is None:
        from enclave.models.encoders import get_reranker

        reranker = get_reranker()

    scores = reranker.score(
        query, [candidate.content for candidate in candidates], batch_size=batch_size
    )
    if len(scores) != len(candidates):
        raise ValueError("reranker returned the wrong number of scores")

    for candidate, score in zip(candidates, scores, strict=True):
        candidate.rerank_score = float(score)

    # Python's sort is stable, so equal reranker scores retain their RRF order.
    ordered = sorted(
        candidates, key=lambda candidate: candidate.rerank_score, reverse=True
    )
    return RankedEvidence(tuple(ordered[:limit]), True, retrieved_count)


def retrieve_and_rank(
    conn,
    query: str,
    *,
    limit: int = 10,
    batch_size: int = 8,
    reranker: CandidateReranker | None = None,
) -> RankedEvidence:
    """Run hybrid retrieval followed by conditional stage-two reranking."""
    candidates = hybrid_search(conn, query)
    return rank_candidates(
        query,
        candidates,
        reranker=reranker,
        limit=limit,
        batch_size=batch_size,
    )
