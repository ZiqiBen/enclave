"""Evaluation harness for conversational query resolution and retrieval."""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from enclave.config import settings
from enclave.context import resolve_question
from enclave.db import connect
from enclave.eval.run import EvalMode, relevance, run_pipeline

DEFAULT_DATASET = Path(__file__).with_name("postgres_conversations.json")


@dataclass(frozen=True, slots=True)
class ConversationCase:
    name: str
    history: tuple[str, ...]
    query: str
    should_contextualize: bool
    expected_anchor: str | None
    relevant_terms: tuple[str, ...]
    answerable: bool = True


@dataclass(frozen=True, slots=True)
class ConversationCaseResult:
    name: str
    contextualized: bool
    expected_contextualized: bool
    decision_correct: bool
    anchor_correct: bool | None
    resolved_query: str
    first_relevant_rank: int | None
    reciprocal_rank: float
    reranked: bool
    latency_ms: float
    answerable: bool


def load_cases(path: Path) -> list[ConversationCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = [
        ConversationCase(
            name=item["name"],
            history=tuple(item.get("history", ())),
            query=item["query"],
            should_contextualize=item["should_contextualize"],
            expected_anchor=item.get("expected_anchor"),
            relevant_terms=tuple(item.get("relevant_terms", ())),
            answerable=item.get("answerable", True),
        )
        for item in raw
    ]
    if not cases:
        raise ValueError("conversation evaluation dataset is empty")
    for case in cases:
        if case.should_contextualize and not case.expected_anchor:
            raise ValueError(f"context case has no expected anchor: {case.name}")
        if case.answerable and not case.relevant_terms:
            raise ValueError(f"answerable case has no relevance terms: {case.name}")
    return cases


def evaluate_case(conn, case: ConversationCase, mode: EvalMode, top_k: int):
    resolved = resolve_question(case.query, list(case.history))
    started = time.perf_counter()
    effective_mode = (
        "hybrid-rerank"
        if resolved.contextualized and mode == "selective-rerank"
        else mode
    )
    ranked = run_pipeline(conn, resolved.retrieval_query, effective_mode, top_k)
    latency_ms = (time.perf_counter() - started) * 1000
    ranks = [
        index
        for index, candidate in enumerate(ranked.candidates, start=1)
        if relevance(candidate, case.relevant_terms)
    ]
    first = ranks[0] if ranks else None
    anchor_correct = (
        resolved.anchor == case.expected_anchor if case.should_contextualize else None
    )
    return ConversationCaseResult(
        name=case.name,
        contextualized=resolved.contextualized,
        expected_contextualized=case.should_contextualize,
        decision_correct=resolved.contextualized == case.should_contextualize,
        anchor_correct=anchor_correct,
        resolved_query=resolved.retrieval_query,
        first_relevant_rank=first,
        reciprocal_rank=0.0 if first is None else 1.0 / first,
        reranked=ranked.reranked,
        latency_ms=latency_ms,
        answerable=case.answerable,
    )


def summarize(results: list[ConversationCaseResult]) -> dict[str, float | int]:
    answerable = [item for item in results if item.answerable]
    anchored = [item for item in results if item.anchor_correct is not None]
    latencies = [item.latency_ms for item in results]
    return {
        "cases": len(results),
        "decision_accuracy": statistics.fmean(
            item.decision_correct for item in results
        ),
        "anchor_accuracy": statistics.fmean(item.anchor_correct for item in anchored),
        "retrieval_hit_rate": statistics.fmean(
            item.first_relevant_rank is not None for item in answerable
        ),
        "mrr": statistics.fmean(item.reciprocal_rank for item in answerable),
        "rerank_rate": statistics.fmean(item.reranked for item in results),
        "mean_latency_ms": statistics.fmean(latencies),
        "p95_latency_ms": sorted(latencies)[
            max(0, math.ceil(len(latencies) * 0.95) - 1)
        ],
    }


def _command(
    dataset: Annotated[
        Path, typer.Option(help="Conversation JSON dataset.")
    ] = DEFAULT_DATASET,
    mode: Annotated[EvalMode, typer.Option()] = "selective-rerank",
    top_k: Annotated[int, typer.Option(min=1, max=100)] = 10,
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    cases = load_cases(dataset)
    results = []
    with connect() as conn:
        for index, case in enumerate(cases, start=1):
            result = evaluate_case(conn, case, mode, top_k)
            results.append(result)
            decision = "ok" if result.decision_correct else "FAIL"
            anchor = "-" if result.anchor_correct is None else result.anchor_correct
            typer.echo(
                f"[{index}/{len(cases)}] decision={decision} "
                f"anchor={anchor} "
                f"rank={result.first_relevant_rank or '-'} {case.name}"
            )
    summary = summarize(results)
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "configuration": settings().describe(),
        "dataset": str(dataset),
        "mode": mode,
        "summary": summary,
        "results": [asdict(item) for item in results],
    }
    typer.echo("\n" + json.dumps(summary, indent=2))
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        typer.echo(f"wrote {output}")


def cli() -> None:
    typer.run(_command)


if __name__ == "__main__":
    cli()
