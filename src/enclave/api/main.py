"""FastAPI application exposing Enclave's retrieval and answer pipeline."""

import asyncio
import os
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, Protocol

import psycopg
import uvicorn
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

from enclave import db
from enclave.answer import VerifiedAnswer, answer_with_verification
from enclave.config import settings
from enclave.ingest.jobs import (
    IngestJob,
    create_job,
    delete_job,
    fail_interrupted_jobs,
    get_job,
    list_jobs,
    run_ingest_job,
    safe_filename,
)
from enclave.rank import RankedEvidence, retrieve_and_rank

WEB_DIR = Path(__file__).resolve().parents[1] / "web"


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


class IngestJobResponse(BaseModel):
    job_id: str
    filename: str
    status: str
    progress: int
    doc_id: str | None
    chunks_written: int
    error: str | None
    created_at: datetime
    updated_at: datetime


def _job_response(job: IngestJob) -> IngestJobResponse:
    return IngestJobResponse(**asdict(job))


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
    injected_services = services is not None
    services = services or Services()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.warmup = None
        if not injected_services:
            with db.connect() as conn:
                fail_interrupted_jobs(conn)
        if settings().warm_models and not injected_services:
            from enclave.models.warmup import warm_local_models

            app.state.warmup = await asyncio.to_thread(warm_local_models)
        yield

    app = FastAPI(
        title="Enclave",
        version="0.1.0",
        description="Zero-egress retrieval and grounded answer engine.",
        lifespan=lifespan,
    )
    app.state.services = services
    app.state.warmup = None
    app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")

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

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready(request: Request, conn: Connection) -> dict[str, object]:
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
        warmup = request.app.state.warmup
        return {
            "status": "ready",
            "database": "ok",
            "models": "warm" if warmup is not None else "on-demand",
            "warmup": asdict(warmup) if warmup is not None else None,
        }

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

    @app.post(
        "/v1/documents",
        response_model=IngestJobResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def upload_document(
        background_tasks: BackgroundTasks,
        conn: Connection,
        file: Annotated[UploadFile, File()],
    ) -> IngestJobResponse:
        cfg = settings()
        try:
            filename = safe_filename(file.filename or "")
        except ValueError as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc

        job_id = uuid.uuid4().hex
        job_dir = cfg.upload_dir.resolve() / job_id
        stored_path = job_dir / f"{job_id}_{filename}"
        job_dir.mkdir(parents=True, exist_ok=False)
        size = 0
        max_bytes = cfg.max_upload_mb * 1024 * 1024
        try:
            with stored_path.open("xb") as output:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > max_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"file exceeds {cfg.max_upload_mb} MB limit",
                        )
                    output.write(chunk)
            if size == 0:
                raise HTTPException(status_code=400, detail="file is empty")
            create_job(conn, job_id, filename, stored_path)
        except Exception:
            stored_path.unlink(missing_ok=True)
            job_dir.rmdir()
            raise
        finally:
            await file.close()

        background_tasks.add_task(run_ingest_job, job_id, stored_path, cfg.database_url)
        job = get_job(conn, job_id)
        assert job is not None
        return _job_response(job)

    @app.get("/v1/documents", response_model=list[IngestJobResponse])
    def documents(conn: Connection) -> list[IngestJobResponse]:
        return [_job_response(job) for job in list_jobs(conn)]

    @app.get("/v1/documents/{job_id}", response_model=IngestJobResponse)
    def document_status(job_id: str, conn: Connection) -> IngestJobResponse:
        job = get_job(conn, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="document not found")
        return _job_response(job)

    @app.delete("/v1/documents/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
    def remove_document(job_id: str, conn: Connection) -> None:
        try:
            deleted = delete_job(conn, job_id, settings().upload_dir)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="document not found")

    return app


app = create_app()


def cli() -> None:
    """Run the local development API."""
    uvicorn.run("enclave.api.main:app", host="127.0.0.1", port=8000, reload=False)


def cli_local() -> None:
    """Migrate, warm every model, and serve the complete local application."""
    os.environ["ENCLAVE_WARM_MODELS"] = "1"
    settings.cache_clear()
    db.migrate()
    uvicorn.run("enclave.api.main:app", host="127.0.0.1", port=8000, reload=False)
