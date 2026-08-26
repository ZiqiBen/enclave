"""Repeatable evaluation for a locally ingested documentation corpus."""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

import typer

from enclave.config import settings
from enclave.db import connect
from enclave.rank import RankedEvidence, rank_candidates
from enclave.retrieval.hybrid import dense_search, hybrid_search, lexical_search

DEFAULT_DATASET = Path(__file__).with_name("postgres_core.json")
EvalMode = Literal["lexical", "dense", "hybrid", "selective-rerank", "hybrid-rerank"]
ALL_MODES: tuple[EvalMode, ...] = (
    "lexical",
    "dense",
    "hybrid",
    "selective-rerank",
    "hybrid-rerank",
)


@dataclass(frozen=True, slots=True)
class EvalCase:
    query: str
    relevant_terms: tuple[str, ...]
    answerable: bool = True


@dataclass(frozen=True, slots=True)
class CaseResult:
    mode: str
    query: str
    answerable: bool
    hit: bool
    reciprocal_rank: float
    ndcg_at_10: float
    first_relevant_rank: int | None
    latency_ms: float
    reranked: bool
    top_heading: str | None


def load_cases(path: Path) -> list[EvalCase]:
    """Load and validate a small, human-reviewed golden question set."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = [
        EvalCase(
            query=item["query"],
            relevant_terms=tuple(item.get("relevant_terms", ())),
            answerable=item.get("answerable", True),
        )
        for item in raw
    ]
    if not cases:
        raise ValueError("evaluation dataset is empty")
    for case in cases:
        if case.answerable and not case.relevant_terms:
            raise ValueError(f"answerable case has no relevance terms: {case.query}")
    return cases


def relevance(candidate, terms: tuple[str, ...]) -> int:
    """Binary relevance judged from curated phrases in heading or content."""
    text = f"{candidate.heading_path or ''}\n{candidate.content}".casefold()
    return int(any(term.casefold() in text for term in terms))


def score_case(
    case: EvalCase, ranked: RankedEvidence, latency_ms: float, mode: str = "hybrid"
) -> CaseResult:
    labels = [
        relevance(candidate, case.relevant_terms) for candidate in ranked.candidates
    ]
    relevant_ranks = [index for index, label in enumerate(labels, start=1) if label]
    first = relevant_ranks[0] if relevant_ranks else None

    dcg = sum(label / math.log2(index + 1) for index, label in enumerate(labels, 1))
    ideal_count = min(sum(labels), 10)
    ideal = sum(1 / math.log2(index + 1) for index in range(1, ideal_count + 1))

    return CaseResult(
        mode=mode,
        query=case.query,
        answerable=case.answerable,
        hit=first is not None,
        reciprocal_rank=0.0 if first is None else 1.0 / first,
        ndcg_at_10=0.0 if ideal == 0 else dcg / ideal,
        first_relevant_rank=first,
        latency_ms=latency_ms,
        reranked=ranked.reranked,
        top_heading=ranked.candidates[0].heading_path if ranked.candidates else None,
    )


def summarize(results: list[CaseResult]) -> dict[str, float | int]:
    answerable = [result for result in results if result.answerable]
    latencies = [result.latency_ms for result in results]
    ordered = sorted(latencies)
    warm_latencies = latencies[1:] or latencies
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "cases": len(results),
        "answerable_cases": len(answerable),
        "hit_rate": statistics.fmean(result.hit for result in answerable),
        "mrr": statistics.fmean(result.reciprocal_rank for result in answerable),
        "ndcg_at_10": statistics.fmean(result.ndcg_at_10 for result in answerable),
        "rerank_rate": statistics.fmean(result.reranked for result in results),
        "mean_latency_ms": statistics.fmean(latencies),
        "warm_mean_latency_ms": statistics.fmean(warm_latencies),
        "p95_latency_ms": ordered[p95_index],
    }


def run_pipeline(conn, query: str, mode: EvalMode, top_k: int) -> RankedEvidence:
    """Run one retrieval ablation while keeping all other inputs constant."""
    depth = settings().resolved_rerank_depth
    if mode == "lexical":
        candidates = lexical_search(conn, query, limit=top_k)
        return rank_candidates(query, candidates, limit=top_k, force=False)
    if mode == "dense":
        candidates = dense_search(conn, query, limit=top_k)
        return rank_candidates(query, candidates, limit=top_k, force=False)

    candidates = hybrid_search(conn, query, limit=depth)
    force = {"hybrid": False, "selective-rerank": None, "hybrid-rerank": True}[mode]
    return rank_candidates(
        query,
        candidates,
        limit=top_k,
        force=force,
    )


def run_evaluation(
    cases: list[EvalCase], *, modes: tuple[EvalMode, ...], top_k: int = 10
) -> list[CaseResult]:
    results: list[CaseResult] = []
    with connect() as conn:
        total = len(cases) * len(modes)
        completed = 0
        for mode in modes:
            for case in cases:
                completed += 1
                started = time.perf_counter()
                ranked = run_pipeline(conn, case.query, mode, top_k)
                elapsed_ms = (time.perf_counter() - started) * 1000
                result = score_case(case, ranked, elapsed_ms, mode)
                results.append(result)
                typer.echo(
                    f"[{completed}/{total}] {mode:<15} "
                    f"rank={result.first_relevant_rank or '-'} "
                    f"{elapsed_ms:.0f}ms  {case.query}"
                )
    return results


def _command(
    dataset: Annotated[
        Path, typer.Option(help="JSON golden question set.")
    ] = DEFAULT_DATASET,
    top_k: Annotated[int, typer.Option(min=1, max=100)] = 10,
    mode: Annotated[
        str,
        typer.Option(help="Ablation mode or 'all'."),
    ] = "all",
    output: Annotated[
        Path | None, typer.Option(help="Optional JSON result path.")
    ] = None,
) -> None:
    """Measure retrieval quality and latency against a golden dataset."""
    if mode != "all" and mode not in ALL_MODES:
        raise typer.BadParameter(
            f"choose all or one of: {', '.join(ALL_MODES)}", param_hint="mode"
        )
    modes = ALL_MODES if mode == "all" else (mode,)
    cases = load_cases(dataset)
    results = run_evaluation(cases, modes=modes, top_k=top_k)  # type: ignore[arg-type]
    summaries = {
        selected: summarize([result for result in results if result.mode == selected])
        for selected in modes
    }
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "configuration": settings().describe(),
        "dataset": str(dataset),
        "top_k": top_k,
        "summaries": summaries,
        "results": [asdict(result) for result in results],
    }
    typer.echo("\n" + json.dumps(summaries, indent=2))
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        typer.echo(f"wrote {output}")


def cli() -> None:
    """Package entry point."""
    typer.run(_command)


if __name__ == "__main__":
    cli()
