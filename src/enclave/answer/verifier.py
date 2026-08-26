"""Claim-level verification of generated answers against cited passages."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from enclave.answer.service import Answer
from enclave.retrieval.hybrid import Candidate

_CITATION = re.compile(r"\[(E\d+)\]")
# A model commonly writes "claim. [E1]". The citation still belongs to that
# claim, so do not split on whitespace when an evidence marker follows.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s+(?!\[E\d+\])")
_VERIFY_INSTRUCTION = (
    "Determine whether the cited document directly supports the claim. "
    "Preserve negation, quantities, and restrictions exactly."
)


class ClaimVerifier(Protocol):
    def score(
        self,
        query: str,
        documents: list[str],
        instruction: str,
        batch_size: int = 8,
    ) -> list[float]: ...


@dataclass(frozen=True, slots=True)
class ClaimVerification:
    claim: str
    citation_ids: tuple[str, ...]
    support_score: float
    supported: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class VerifiedAnswer:
    answer: Answer
    claims: tuple[ClaimVerification, ...]
    verified: bool
    generation_duration_ms: float | None = None
    verification_duration_ms: float | None = None


def _claims(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]


def verify_answer(
    answer: Answer,
    evidence: list[Candidate] | tuple[Candidate, ...],
    *,
    verifier: ClaimVerifier | None = None,
    threshold: float = 0.5,
) -> VerifiedAnswer:
    """Verify every cited claim; one unsupported claim fails the answer."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if answer.insufficient_evidence:
        return VerifiedAnswer(answer, (), True)

    evidence_by_id = {
        f"E{index}": candidate for index, candidate in enumerate(evidence, start=1)
    }
    results: list[ClaimVerification] = []

    for sentence in _claims(answer.text):
        citation_ids = tuple(dict.fromkeys(_CITATION.findall(sentence)))
        if not citation_ids:
            results.append(
                ClaimVerification(
                    claim=sentence,
                    citation_ids=(),
                    support_score=0.0,
                    supported=False,
                    reason="claim has no inline citation",
                )
            )
            continue

        unknown = [item for item in citation_ids if item not in evidence_by_id]
        if unknown:
            results.append(
                ClaimVerification(
                    claim=sentence,
                    citation_ids=citation_ids,
                    support_score=0.0,
                    supported=False,
                    reason=f"unknown citations: {unknown}",
                )
            )
            continue

        if verifier is None:
            from enclave.models.encoders import get_reranker

            verifier = get_reranker()

        clean_claim = _CITATION.sub("", sentence).strip()
        passages = [evidence_by_id[item].content for item in citation_ids]
        scores = verifier.score(
            clean_claim,
            passages,
            instruction=_VERIFY_INSTRUCTION,
            batch_size=len(passages),
        )
        if len(scores) != len(passages):
            raise ValueError("verifier returned the wrong number of scores")
        support_score = max((float(score) for score in scores), default=0.0)
        supported = support_score >= threshold
        results.append(
            ClaimVerification(
                claim=sentence,
                citation_ids=citation_ids,
                support_score=support_score,
                supported=supported,
                reason=None if supported else "support score below threshold",
            )
        )

    return VerifiedAnswer(
        answer=answer,
        claims=tuple(results),
        verified=bool(results) and all(result.supported for result in results),
    )
