"""Claim-level answer verification without real model weights."""

from __future__ import annotations

import pytest

from enclave.answer import Answer, Citation, answer_with_verification, verify_answer
from enclave.answer.service import Generation
from enclave.retrieval.hybrid import Candidate


def evidence() -> Candidate:
    return Candidate(
        chunk_id=1,
        doc_id="policy.md",
        heading_path="External services",
        content="Confidential documents must not be uploaded to external services.",
        fusion_score=1.0,
        lex_rank=1,
        dense_rank=1,
    )


def answer(text: str, *, insufficient: bool = False) -> Answer:
    citations = ()
    if not insufficient:
        citations = (Citation("E1", 1, "policy.md", "External services"),)
    return Answer(text, citations, insufficient, "test-model")


class FakeVerifier:
    def __init__(self, scores: list[list[float]]):
        self.scores = scores
        self.calls: list[tuple[str, list[str], str, int]] = []

    def score(
        self,
        query: str,
        documents: list[str],
        instruction: str,
        batch_size: int = 8,
    ) -> list[float]:
        self.calls.append((query, documents, instruction, batch_size))
        return self.scores[len(self.calls) - 1]


def test_supported_claim_passes_and_uses_only_its_citation():
    verifier = FakeVerifier([[0.97]])

    result = verify_answer(
        answer("No, documents cannot be uploaded [E1]."),
        [evidence()],
        verifier=verifier,
        threshold=0.8,
    )

    assert result.verified is True
    assert result.claims[0].support_score == 0.97
    query, documents, instruction, batch_size = verifier.calls[0]
    assert "[E1]" not in query
    assert documents == [evidence().content]
    assert "negation" in instruction
    assert batch_size == 1


def test_low_support_score_fails_the_answer():
    result = verify_answer(
        answer("Yes, documents can be uploaded [E1]."),
        [evidence()],
        verifier=FakeVerifier([[0.05]]),
        threshold=0.8,
    )
    assert result.verified is False
    assert result.claims[0].supported is False
    assert result.claims[0].reason == "support score below threshold"


def test_one_uncited_sentence_fails_a_multi_claim_answer():
    result = verify_answer(
        answer("Documents stay local [E1]. Uploading is prohibited."),
        [evidence()],
        verifier=FakeVerifier([[0.95]]),
    )
    assert result.verified is False
    assert [claim.supported for claim in result.claims] == [True, False]
    assert result.claims[1].reason == "claim has no inline citation"


def test_unknown_inline_citation_fails_without_loading_model():
    result = verify_answer(answer("Invented claim [E9]."), [evidence()])
    assert result.verified is False
    assert result.claims[0].reason == "unknown citations: ['E9']"


def test_insufficient_evidence_refusal_needs_no_verifier():
    result = verify_answer(answer("Not enough evidence.", insufficient=True), [])
    assert result.verified is True
    assert result.claims == ()


def test_multiple_citations_use_best_support_score():
    second = evidence()
    second.chunk_id = 2
    second.doc_id = "handbook.md"
    result = verify_answer(
        answer("Documents stay local [E1] [E2]."),
        [evidence(), second],
        verifier=FakeVerifier([[0.2, 0.9]]),
        threshold=0.8,
    )
    assert result.verified is True
    assert result.claims[0].support_score == 0.9


def test_rejects_invalid_threshold():
    with pytest.raises(ValueError, match="threshold"):
        verify_answer(answer("Claim [E1]."), [evidence()], threshold=1.1)


def test_rejects_wrong_score_count():
    with pytest.raises(ValueError, match="wrong number"):
        verify_answer(answer("Claim [E1]."), [evidence()], verifier=FakeVerifier([[]]))


def test_complete_pipeline_generates_then_verifies():
    class Client:
        def generate(self, *, prompt, schema):
            return Generation(
                '{"answer":"Documents stay local [E1].",'
                '"citation_ids":["E1"],"insufficient_evidence":false}',
                "local-model",
            )

    result = answer_with_verification(
        "Where do documents stay?",
        [evidence()],
        client=Client(),
        verifier=FakeVerifier([[0.99]]),
        verification_threshold=0.8,
    )

    assert result.verified is True
    assert result.answer.model == "local-model"
