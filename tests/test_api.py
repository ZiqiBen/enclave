"""FastAPI contracts with deterministic injected services."""

from __future__ import annotations

from fastapi.testclient import TestClient

from enclave.answer import Answer, Citation, ClaimVerification, VerifiedAnswer
from enclave.api.main import Services, create_app
from enclave.rank import RankedEvidence
from enclave.retrieval.hybrid import Candidate


class FakeCursor:
    def __init__(self, *, schema_ready: bool = True, feedback_id: int = 41):
        self.schema_ready = schema_ready
        self.feedback_id = feedback_id
        self.executions: list[tuple[str, object]] = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, query: str, params=None):
        self.executions.append((query, params))

    def fetchone(self):
        query = self.executions[-1][0]
        if "to_regclass" in query:
            return (self.schema_ready,)
        return (self.feedback_id,)


class FakeConnection:
    def __init__(self, *, schema_ready: bool = True):
        self.fake_cursor = FakeCursor(schema_ready=schema_ready)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def cursor(self):
        return self.fake_cursor


def candidate() -> Candidate:
    return Candidate(
        chunk_id=7,
        doc_id="policy.md",
        heading_path="Data handling",
        content="Documents must remain local.",
        fusion_score=0.03,
        lex_rank=1,
        dense_rank=2,
        rerank_score=0.94,
    )


def ranked() -> RankedEvidence:
    return RankedEvidence((candidate(),), True, 25)


def verified() -> VerifiedAnswer:
    result = Answer(
        text="Documents must remain local [E1].",
        citations=(Citation("E1", 7, "policy.md", "Data handling"),),
        insufficient_evidence=False,
        model="qwen3:4b",
        total_duration_ns=1_250_000,
    )
    claim = ClaimVerification(
        claim=result.text,
        citation_ids=("E1",),
        support_score=0.97,
        supported=True,
    )
    return VerifiedAnswer(result, (claim,), True)


def client(*, schema_ready: bool = True):
    conn = FakeConnection(schema_ready=schema_ready)
    services = Services(
        connect=lambda: conn,
        search=lambda connection, query, limit: ranked(),
        query=lambda connection, query, limit: (ranked(), verified()),
    )
    return TestClient(create_app(services)), conn


def test_liveness_does_not_touch_dependencies():
    api, _ = client()
    response = api.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_checks_migrated_schema():
    api, _ = client()
    assert api.get("/health/ready").json() == {
        "status": "ready",
        "database": "ok",
    }


def test_readiness_fails_when_schema_is_missing():
    api, _ = client(schema_ready=False)
    response = api.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["detail"] == "database schema is not migrated"


def test_search_returns_ranked_evidence_and_diagnostics():
    api, _ = client()
    response = api.post("/v1/search", json={"query": "policy", "top_k": 3})
    body = response.json()
    assert response.status_code == 200
    assert body["reranked"] is True
    assert body["retrieved_count"] == 25
    assert body["evidence"][0] == {
        "evidence_id": "E1",
        "chunk_id": 7,
        "doc_id": "policy.md",
        "heading_path": "Data handling",
        "content": "Documents must remain local.",
        "fusion_score": 0.03,
        "rerank_score": 0.94,
        "lex_rank": 1,
        "dense_rank": 2,
    }


def test_query_returns_answer_citations_and_claim_verification():
    api, _ = client()
    response = api.post("/v1/query", json={"query": "Where?"})
    body = response.json()
    assert response.status_code == 200
    assert body["answer"] == "Documents must remain local [E1]."
    assert body["verified"] is True
    assert body["claims"][0]["support_score"] == 0.97
    assert body["citations"][0]["chunk_id"] == 7
    assert body["generation_duration_ms"] == 1.25
    assert body["model"] == "qwen3:4b"


def test_feedback_is_persisted_and_returns_identifier():
    api, conn = client()
    response = api.post(
        "/v1/feedback",
        json={
            "query": "policy",
            "chunk_id": 7,
            "shown_rank": 1,
            "relevant": True,
            "stage": "rerank",
        },
    )
    assert response.status_code == 201
    assert response.json() == {"id": 41}
    query, params = conn.fake_cursor.executions[-1]
    assert "INSERT INTO feedback" in query
    assert params == ("policy", 7, 1, True, "rerank")


def test_request_validation_rejects_empty_query_and_large_top_k():
    api, _ = client()
    assert api.post("/v1/query", json={"query": ""}).status_code == 422
    assert api.post("/v1/search", json={"query": "ok", "top_k": 21}).status_code == 422


def test_unknown_fields_are_rejected():
    api, _ = client()
    response = api.post("/v1/query", json={"query": "ok", "secret": "ignored"})
    assert response.status_code == 422
