"""Complete grounded-answer pipeline."""

from __future__ import annotations

import time

from enclave.answer.service import AnswerClient, generate_answer
from enclave.answer.verifier import ClaimVerifier, VerifiedAnswer, verify_answer
from enclave.retrieval.hybrid import Candidate


def answer_with_verification(
    query: str,
    evidence: list[Candidate] | tuple[Candidate, ...],
    *,
    client: AnswerClient | None = None,
    verifier: ClaimVerifier | None = None,
    verification_threshold: float = 0.5,
) -> VerifiedAnswer:
    """Generate a local answer, then verify every claim against its citations."""
    started = time.perf_counter()
    answer = generate_answer(query, evidence, client=client)
    generation_duration_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    verified = verify_answer(
        answer,
        evidence,
        verifier=verifier,
        threshold=verification_threshold,
    )
    verification_duration_ms = (time.perf_counter() - started) * 1000
    return VerifiedAnswer(
        answer=verified.answer,
        claims=verified.claims,
        verified=verified.verified,
        generation_duration_ms=generation_duration_ms,
        verification_duration_ms=verification_duration_ms,
    )
