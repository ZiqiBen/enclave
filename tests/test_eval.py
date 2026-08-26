from __future__ import annotations

import json
import math

import pytest

from enclave.eval.run import EvalCase, load_cases, run_pipeline, score_case, summarize
from enclave.rank import RankedEvidence
from enclave.retrieval.hybrid import Candidate


def candidate(chunk_id: int, text: str, heading: str = "Section") -> Candidate:
    return Candidate(chunk_id, "doc", heading, text, 0.1, 1, 1)


def test_scores_ranked_relevant_evidence():
    case = EvalCase("database question", ("correct phrase",))
    ranked = RankedEvidence(
        (
            candidate(1, "unrelated"),
            candidate(2, "contains the correct phrase"),
        ),
        True,
        2,
    )

    result = score_case(case, ranked, 12.5)

    assert result.hit is True
    assert result.first_relevant_rank == 2
    assert result.reciprocal_rank == 0.5
    assert result.ndcg_at_10 == pytest.approx(1 / math.log2(3))


def test_summary_reports_quality_and_latency():
    first = score_case(
        EvalCase("hit", ("yes",)),
        RankedEvidence((candidate(1, "yes"),), False, 1),
        10,
    )
    second = score_case(
        EvalCase("miss", ("yes",)),
        RankedEvidence((candidate(2, "no"),), True, 1),
        30,
    )

    summary = summarize([first, second])

    assert summary["hit_rate"] == 0.5
    assert summary["mrr"] == 0.5
    assert summary["mean_latency_ms"] == 20
    assert summary["warm_mean_latency_ms"] == 30
    assert summary["p95_latency_ms"] == 30


def test_load_cases_rejects_answerable_case_without_terms(tmp_path):
    dataset = tmp_path / "questions.json"
    dataset.write_text(json.dumps([{"query": "question"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="no relevance terms"):
        load_cases(dataset)


@pytest.mark.parametrize(
    ("mode", "reranked"),
    [
        ("lexical", False),
        ("dense", False),
        ("hybrid", False),
        ("selective-rerank", False),
    ],
)
def test_ablation_modes_do_not_rerank(monkeypatch, mode, reranked):
    candidates = [candidate(1, "evidence")]
    monkeypatch.setattr("enclave.eval.run.lexical_search", lambda *a, **k: candidates)
    monkeypatch.setattr("enclave.eval.run.dense_search", lambda *a, **k: candidates)
    monkeypatch.setattr("enclave.eval.run.hybrid_search", lambda *a, **k: candidates)

    result = run_pipeline(object(), "question", mode, 1)

    assert result.reranked is reranked
