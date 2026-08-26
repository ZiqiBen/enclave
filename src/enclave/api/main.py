"""FastAPI application exposing Enclave's retrieval and answer pipeline."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Literal, Protocol

import psycopg
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from enclave import db
from enclave.answer import VerifiedAnswer, answer_with_verification
from enclave.rank import RankedEvidence, retrieve_and_rank


class SearchRunner(Protocol):
    def __call__(self, conn, query: str, limit: int) -> RankedEvidence: ...


class QueryRunner(Protocol):
    def __call__(
        self, conn, query: str, limit: int
    ) -> tuple[RankedEvidence, VerifiedAnswer]: ...


@dataclass(frozen=True, slots=True)
class Services:
    connect: Callable = db.connect
    search: SearchRunner | None = None
    query: QueryRunner | None = None


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)


class EvidenceResponse(BaseModel):
    evidence_id: str
    chunk_id: int
    doc_id: str
    heading_path: str | None
    content: str
    fusion_score: float
    rerank_score: float | None
    lex_rank: int | None
    dense_rank: int | None


class TimingResponse(BaseModel):
    retrieval_ms: float | None
    rerank_ms: float | None
    generation_ms: float | None = None
    verification_ms: float | None = None
    total_ms: float


class SearchResponse(BaseModel):
    query: str
    evidence: list[EvidenceResponse]
    reranked: bool
    retrieved_count: int
    timings: TimingResponse


class CitationResponse(BaseModel):
    evidence_id: str
    chunk_id: int
    doc_id: str
    heading_path: str | None


class ClaimResponse(BaseModel):
    claim: str
    citation_ids: list[str]
    support_score: float
    supported: bool
    reason: str | None


class QueryResponse(SearchResponse):
    answer: str
    citations: list[CitationResponse]
    insufficient_evidence: bool
    verified: bool
    claims: list[ClaimResponse]
    model: str | None
    generation_duration_ms: float | None


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4000)
    chunk_id: int = Field(gt=0)
    shown_rank: int = Field(gt=0)
    relevant: bool
    stage: Literal["retrieval", "rerank"]


class FeedbackResponse(BaseModel):
    id: int


def _default_search(conn, query: str, limit: int) -> RankedEvidence:
    return retrieve_and_rank(conn, query, limit=limit)


def _default_query(
    conn, query: str, limit: int
) -> tuple[RankedEvidence, VerifiedAnswer]:
    ranked = retrieve_and_rank(conn, query, limit=limit)
    verified = answer_with_verification(query, ranked.candidates)
    return ranked, verified


def _evidence(ranked: RankedEvidence) -> list[EvidenceResponse]:
    return [
        EvidenceResponse(
            evidence_id=f"E{index}",
            chunk_id=item.chunk_id,
            doc_id=item.doc_id,
            heading_path=item.heading_path,
            content=item.content,
            fusion_score=item.fusion_score,
            rerank_score=item.rerank_score,
            lex_rank=item.lex_rank,
            dense_rank=item.dense_rank,
        )
        for index, item in enumerate(ranked.candidates, start=1)
    ]


def create_app(services: Services | None = None) -> FastAPI:
    services = services or Services()
    app = FastAPI(
        title="Enclave",
        version="0.1.0",
        description="Zero-egress retrieval and grounded answer engine.",
    )
    app.state.services = services

    def get_conn(request: Request):
        try:
            with request.app.state.services.connect() as conn:
                yield conn
        except psycopg.OperationalError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="database unavailable",
            ) from exc

    Connection = Annotated[object, Depends(get_conn)]

    @app.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready(conn: Connection) -> dict[str, object]:
        try:
            with conn.cursor() as cur:  # type: ignore[attr-defined]
                cur.execute("SELECT to_regclass('public.chunks') IS NOT NULL")
                schema_ready = bool(cur.fetchone()[0])
        except psycopg.Error as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="database check failed",
            ) from exc
        if not schema_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="database schema is not migrated",
            )
        return {"status": "ready", "database": "ok"}

    @app.post("/v1/search", response_model=SearchResponse)
    def search(body: QueryRequest, conn: Connection) -> SearchResponse:
        runner = app.state.services.search or _default_search
        started = time.perf_counter()
        ranked = runner(conn, body.query, body.top_k)
        total_ms = (time.perf_counter() - started) * 1000
        return SearchResponse(
            query=body.query,
            evidence=_evidence(ranked),
            reranked=ranked.reranked,
            retrieved_count=ranked.retrieved_count,
            timings=TimingResponse(
                retrieval_ms=ranked.retrieval_duration_ms,
                rerank_ms=ranked.rerank_duration_ms,
                total_ms=total_ms,
            ),
        )

    @app.post("/v1/query", response_model=QueryResponse)
    def query(body: QueryRequest, conn: Connection) -> QueryResponse:
        runner = app.state.services.query or _default_query
        started = time.perf_counter()
        ranked, verified = runner(conn, body.query, body.top_k)
        total_ms = (time.perf_counter() - started) * 1000
        answer = verified.answer
        duration = (
            answer.total_duration_ns / 1_000_000
            if answer.total_duration_ns is not None
            else None
        )
        return QueryResponse(
            query=body.query,
            evidence=_evidence(ranked),
            reranked=ranked.reranked,
            retrieved_count=ranked.retrieved_count,
            timings=TimingResponse(
                retrieval_ms=ranked.retrieval_duration_ms,
                rerank_ms=ranked.rerank_duration_ms,
                generation_ms=verified.generation_duration_ms,
                verification_ms=verified.verification_duration_ms,
                total_ms=total_ms,
            ),
            answer=answer.text,
            citations=[
                CitationResponse(
                    evidence_id=item.evidence_id,
                    chunk_id=item.chunk_id,
                    doc_id=item.doc_id,
                    heading_path=item.heading_path,
                )
                for item in answer.citations
            ],
            insufficient_evidence=answer.insufficient_evidence,
            verified=verified.verified,
            claims=[
                ClaimResponse(
                    claim=item.claim,
                    citation_ids=list(item.citation_ids),
                    support_score=item.support_score,
                    supported=item.supported,
                    reason=item.reason,
                )
                for item in verified.claims
            ],
            model=answer.model,
            generation_duration_ms=duration,
        )

    @app.post(
        "/v1/feedback",
        response_model=FeedbackResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def feedback(body: FeedbackRequest, conn: Connection) -> FeedbackResponse:
        try:
            with conn.cursor() as cur:  # type: ignore[attr-defined]
                cur.execute(
                    "INSERT INTO feedback "
                    "(query, chunk_id, shown_rank, relevant, stage) "
                    "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                    (
                        body.query,
                        body.chunk_id,
                        body.shown_rank,
                        body.relevant,
                        body.stage,
                    ),
                )
                feedback_id = cur.fetchone()[0]
        except psycopg.errors.ForeignKeyViolation as exc:
            raise HTTPException(status_code=404, detail="chunk not found") from exc
        return FeedbackResponse(id=feedback_id)

    return app


app = create_app()


def cli() -> None:
    """Run the local development API."""
    uvicorn.run("enclave.api.main:app", host="127.0.0.1", port=8000, reload=False)
