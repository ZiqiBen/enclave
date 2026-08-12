"""Grounded answer generation and Ollama request contracts."""

from __future__ import annotations

import json

import httpx
import pytest

from enclave.answer.service import Generation, OllamaClient, generate_answer
from enclave.retrieval.hybrid import Candidate


def evidence(chunk_id: int = 7) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        doc_id="security-policy.md",
        heading_path="Data handling",
        content="Confidential documents must stay on local infrastructure.",
        fusion_score=0.9,
        lex_rank=1,
        dense_rank=1,
    )


class FakeClient:
    def __init__(self, payload: dict[str, object]):
        self.payload = payload
        self.prompts: list[str] = []

    def generate(self, *, prompt: str, schema: dict[str, object]) -> Generation:
        self.prompts.append(prompt)
        assert schema["type"] == "object"
        return Generation(json.dumps(self.payload), "local-test-model", 42)


def test_generates_answer_and_resolves_citation_metadata():
    client = FakeClient(
        {
            "answer": "Documents must remain local [E1].",
            "citation_ids": ["E1"],
            "insufficient_evidence": False,
        }
    )

    answer = generate_answer("Where must documents stay?", [evidence()], client=client)

    assert answer.text == "Documents must remain local [E1]."
    assert answer.model == "local-test-model"
    assert answer.total_duration_ns == 42
    assert answer.citations[0].chunk_id == 7
    assert answer.citations[0].heading_path == "Data handling"
    assert "[E1] document=security-policy.md" in client.prompts[0]


def test_no_evidence_refuses_without_calling_model():
    answer = generate_answer("Unknown?", [])
    assert answer.insufficient_evidence is True
    assert answer.citations == ()
    assert answer.model is None


def test_rejects_unknown_citations():
    client = FakeClient(
        {
            "answer": "Invented [E9].",
            "citation_ids": ["E9"],
            "insufficient_evidence": False,
        }
    )
    with pytest.raises(ValueError, match="unknown evidence"):
        generate_answer("Question?", [evidence()], client=client)


def test_rejects_citations_on_insufficient_answer():
    client = FakeClient(
        {
            "answer": "Not enough information.",
            "citation_ids": ["E1"],
            "insufficient_evidence": True,
        }
    )
    with pytest.raises(ValueError, match="must not include citations"):
        generate_answer("Question?", [evidence()], client=client)


def test_rejects_grounded_answer_without_citations():
    client = FakeClient(
        {
            "answer": "Unsupported answer.",
            "citation_ids": [],
            "insufficient_evidence": False,
        }
    )
    with pytest.raises(ValueError, match="must include citations"):
        generate_answer("Question?", [evidence()], client=client)


def test_duplicate_citations_are_resolved_once():
    client = FakeClient(
        {
            "answer": "Local [E1].",
            "citation_ids": ["E1", "E1"],
            "insufficient_evidence": False,
        }
    )
    answer = generate_answer("Question?", [evidence()], client=client)
    assert [citation.evidence_id for citation in answer.citations] == ["E1"]


def test_declared_citation_is_attached_to_uncited_answer_sentence():
    client = FakeClient(
        {
            "answer": "PostgreSQL is a relational database management system.",
            "citation_ids": ["E1"],
            "insufficient_evidence": False,
        }
    )
    answer = generate_answer("What is PostgreSQL?", [evidence()], client=client)
    assert answer.text == (
        "PostgreSQL is a relational database management system. [E1]"
    )


def test_declared_citations_are_attached_to_each_uncited_sentence():
    client = FakeClient(
        {
            "answer": "First claim. Second claim.",
            "citation_ids": ["E1"],
            "insufficient_evidence": False,
        }
    )
    answer = generate_answer("Question?", [evidence()], client=client)
    assert answer.text == "First claim. [E1] Second claim. [E1]"


def test_rejects_inline_citation_missing_from_declared_ids():
    client = FakeClient(
        {
            "answer": "Mismatched citation [E2].",
            "citation_ids": ["E1"],
            "insufficient_evidence": False,
        }
    )
    with pytest.raises(ValueError, match="undeclared citations"):
        generate_answer("Question?", [evidence()], client=client)


def test_rejects_empty_query():
    with pytest.raises(ValueError, match="query must not be empty"):
        generate_answer("  ", [evidence()])


def test_ollama_client_sends_structured_non_streaming_request():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        assert request.url == "http://127.0.0.1:11434/api/chat"
        return httpx.Response(
            200,
            json={
                "model": "qwen3:4b",
                "message": {"role": "assistant", "content": '{"answer":"ok"}'},
                "total_duration": 123,
            },
        )

    client = OllamaClient(
        model="qwen3:4b", transport=httpx.MockTransport(handler), timeout_s=1
    )
    result = client.generate(prompt="prompt", schema={"type": "object"})

    assert captured["stream"] is False
    assert captured["think"] is False
    assert captured["format"] == {"type": "object"}
    assert captured["options"] == {"temperature": 0}
    assert result.total_duration_ns == 123


def test_ollama_client_rejects_malformed_response():
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={}))
    client = OllamaClient(model="test", transport=transport)
    with pytest.raises(ValueError, match="invalid chat response"):
        client.generate(prompt="prompt", schema={"type": "object"})
