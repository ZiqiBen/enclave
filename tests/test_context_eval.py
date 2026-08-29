from enclave.eval.contextual import (
    ConversationCase,
    ConversationCaseResult,
    evaluate_case,
    load_cases,
    summarize,
)
from enclave.rank import RankedEvidence
from enclave.retrieval.hybrid import Candidate


def result(*, decision=True, anchor=True, rank=1, latency=10):
    return ConversationCaseResult(
        name="case",
        contextualized=True,
        expected_contextualized=True,
        decision_correct=decision,
        anchor_correct=anchor,
        resolved_query="resolved",
        first_relevant_rank=rank,
        reciprocal_rank=0 if rank is None else 1 / rank,
        reranked=False,
        latency_ms=latency,
        answerable=True,
    )


def test_summary_reports_context_and_retrieval_quality():
    summary = summarize(
        [
            result(),
            result(decision=False, anchor=False, rank=None, latency=30),
        ]
    )
    assert summary["decision_accuracy"] == 0.5
    assert summary["anchor_accuracy"] == 0.5
    assert summary["retrieval_hit_rate"] == 0.5
    assert summary["mrr"] == 0.5
    assert summary["mean_latency_ms"] == 20


def test_default_dataset_contains_positive_and_negative_cases():
    from enclave.eval.contextual import DEFAULT_DATASET

    cases = load_cases(DEFAULT_DATASET)
    assert len(cases) >= 15
    assert any(case.should_contextualize for case in cases)
    assert any(not case.should_contextualize for case in cases)


def test_contextual_case_forces_reranking_under_selective_policy(monkeypatch):
    modes = []
    candidate = Candidate(1, "doc", "MVCC", "MVCC evidence", 0.1, 2, 2)

    def fake_pipeline(conn, query, mode, top_k):
        modes.append(mode)
        return RankedEvidence((candidate,), True, 1)

    monkeypatch.setattr("enclave.eval.contextual.run_pipeline", fake_pipeline)
    case = ConversationCase(
        "followup",
        ("What is MVCC?",),
        "How does it work?",
        True,
        "What is MVCC?",
        ("MVCC",),
    )

    evaluate_case(object(), case, "selective-rerank", 5)

    assert modes == ["hybrid-rerank"]
