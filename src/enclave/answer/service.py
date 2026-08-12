"""Grounded answer generation from ranked local evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import httpx
from pydantic import BaseModel, ConfigDict

from enclave.config import settings
from enclave.retrieval.hybrid import Candidate


class _ModelAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    citation_ids: list[str]
    insufficient_evidence: bool


@dataclass(frozen=True, slots=True)
class Generation:
    content: str
    model: str
    total_duration_ns: int | None = None


class AnswerClient(Protocol):
    def generate(self, *, prompt: str, schema: dict[str, object]) -> Generation: ...


@dataclass(frozen=True, slots=True)
class Citation:
    evidence_id: str
    chunk_id: int
    doc_id: str
    heading_path: str | None


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    citations: tuple[Citation, ...]
    insufficient_evidence: bool
    model: str | None
    total_duration_ns: int | None = None


class OllamaClient:
    """Small client for Ollama's local structured-chat endpoint."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ):
        cfg = settings()
        self.base_url = (base_url or cfg.ollama_url).rstrip("/")
        self.model = model or cfg.resolved_llm_model
        self.timeout_s = timeout_s
        self.transport = transport

    def generate(self, *, prompt: str, schema: dict[str, object]) -> Generation:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "format": schema,
            "options": {"temperature": 0},
        }
        with httpx.Client(
            timeout=self.timeout_s, transport=self.transport, trust_env=False
        ) as client:
            response = client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
        data = response.json()
        try:
            content = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise ValueError("Ollama returned an invalid chat response") from exc
        return Generation(
            content=content,
            model=data.get("model", self.model),
            total_duration_ns=data.get("total_duration"),
        )


def _prompt(query: str, evidence: list[Candidate]) -> str:
    passages = []
    for index, candidate in enumerate(evidence, start=1):
        heading = candidate.heading_path or "Untitled section"
        passages.append(
            f"[E{index}] document={candidate.doc_id}; heading={heading}\n"
            f"{candidate.content}"
        )
    context = "\n\n".join(passages)
    return f"""Answer the question using only the evidence below.

Rules:
- Do not use outside knowledge or invent facts.
- Preserve negation and restrictions exactly. For a yes/no question, begin with
  "Yes" only when the evidence supports the proposition in the question;
  otherwise begin with "No".
- Cite every factual claim with evidence IDs such as [E1].
- Return only JSON matching the supplied schema.
- citation_ids must contain every evidence ID used in the answer.
- If the evidence cannot answer the question, set insufficient_evidence to true,
  give a brief refusal, and return an empty citation_ids list.

Question:
{query}

Evidence:
{context}
"""


def generate_answer(
    query: str,
    evidence: list[Candidate] | tuple[Candidate, ...],
    *,
    client: AnswerClient | None = None,
) -> Answer:
    """Generate a cited answer and reject citations outside supplied evidence."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if not evidence:
        return Answer(
            text="The indexed documents do not contain enough evidence to answer.",
            citations=(),
            insufficient_evidence=True,
            model=None,
        )

    generation = (client or OllamaClient()).generate(
        prompt=_prompt(query, list(evidence)), schema=_ModelAnswer.model_json_schema()
    )
    parsed = _ModelAnswer.model_validate_json(generation.content)

    by_id = {f"E{index}": item for index, item in enumerate(evidence, start=1)}
    unknown = set(parsed.citation_ids) - set(by_id)
    if unknown:
        raise ValueError(f"model cited unknown evidence: {sorted(unknown)}")
    if parsed.insufficient_evidence and parsed.citation_ids:
        raise ValueError("an insufficient-evidence answer must not include citations")

    citations = tuple(
        Citation(
            evidence_id=evidence_id,
            chunk_id=by_id[evidence_id].chunk_id,
            doc_id=by_id[evidence_id].doc_id,
            heading_path=by_id[evidence_id].heading_path,
        )
        for evidence_id in dict.fromkeys(parsed.citation_ids)
    )
    return Answer(
        text=parsed.answer,
        citations=citations,
        insufficient_evidence=parsed.insufficient_evidence,
        model=generation.model,
        total_duration_ns=generation.total_duration_ns,
    )
