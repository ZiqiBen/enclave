"""Local model warm-up used before accepting interactive traffic."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from enclave.config import settings


@dataclass(frozen=True, slots=True)
class WarmupResult:
    embedding_ms: float
    reranker_ms: float
    ollama_ms: float
    total_ms: float


def _ollama_keep_alive() -> None:
    cfg = settings()
    with httpx.Client(timeout=120.0, trust_env=False) as client:
        response = client.post(
            f"{cfg.ollama_url.rstrip('/')}/api/generate",
            json={
                "model": cfg.resolved_llm_model,
                "prompt": "",
                "stream": False,
                "keep_alive": -1,
            },
        )
        response.raise_for_status()


def warm_local_models(
    *,
    embedder_loader: Callable | None = None,
    reranker_loader: Callable | None = None,
    ollama_loader: Callable[[], None] | None = None,
) -> WarmupResult:
    """Load every local model and execute the kernels used by a real query."""
    if embedder_loader is None or reranker_loader is None:
        from enclave.models.encoders import get_embedder, get_reranker

        embedder_loader = embedder_loader or get_embedder
        reranker_loader = reranker_loader or get_reranker

    total_started = time.perf_counter()

    started = time.perf_counter()
    embedder_loader().encode_queries(["PostgreSQL documentation warm-up"])
    embedding_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    reranker_loader().score(
        "PostgreSQL documentation warm-up",
        ["PostgreSQL is a relational database management system."],
        batch_size=1,
    )
    reranker_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    (ollama_loader or _ollama_keep_alive)()
    ollama_ms = (time.perf_counter() - started) * 1000

    return WarmupResult(
        embedding_ms=embedding_ms,
        reranker_ms=reranker_ms,
        ollama_ms=ollama_ms,
        total_ms=(time.perf_counter() - total_started) * 1000,
    )
