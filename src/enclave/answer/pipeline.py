"""Complete grounded-answer pipeline."""

from __future__ import annotations

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
    answer = generate_answer(query, evidence, client=client)
    return verify_answer(
        answer,
        evidence,
        verifier=verifier,
        threshold=verification_threshold,
    )
