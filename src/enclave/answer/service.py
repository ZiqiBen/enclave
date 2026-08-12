"""Grounded answer generation from ranked local evidence."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

import httpx
from pydantic import BaseModel, ConfigDict

from enclave.config import settings
from enclave.retrieval.hybrid import Candidate

_INLINE_CITATION = re.compile(r"\[(E\d+)\]")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s+(?!\[E\d+\])")


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


def _normalize_inline_citations(text: str, citation_ids: list[str]) -> str:
    """Attach declared citations to uncited claims for deterministic checking.

    Small local models occasionally populate ``citation_ids`` correctly but
    omit the same marker from the answer text. Attaching all declared evidence
    to an uncited sentence lets the verifier judge that sentence; it does not
    make the sentence trusted or supported by itself.
    """
    declared = tuple(dict.fromkeys(citation_ids))
    declared_set = set(declared)
    inline = set(_INLINE_CITATION.findall(text))
    undeclared = inline - declared_set
    if undeclared:
        raise ValueError(f"answer used undeclared citations: {sorted(undeclared)}")

    suffix = "".join(f"[{item}]" for item in declared)
    sentences = [
        part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()
    ]
    return " ".join(
        sentence if _INLINE_CITATION.search(sentence) else f"{sentence} {suffix}"
        for sentence in sentences
    )


def _prompt(query: str, evidence: list[Candidate], schema: dict[str, object]) -> str:
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
- Answer the question directly. Do not repeat or paraphrase the question as the
  answer.
- Preserve negation and restrictions exactly. For a yes/no question, begin with
  "Yes" only when the evidence supports the proposition in the question;
  otherwise begin with "No".
- End every sentence containing a factual claim with one or more evidence IDs,
  such as [E1]. Do not place one citation only at the end of a multi-sentence
  paragraph.
- Every non-refusal answer must contain at least one inline [E#] citation and
  citation_ids must list those same IDs.
- Return only JSON matching the supplied schema.
- citation_ids must contain every evidence ID used in the answer.
- If the evidence cannot answer the question, set insufficient_evidence to true,
  give a brief refusal, and return an empty citation_ids list.

Question:
{query}

Evidence:
{context}

Output JSON schema:
{json.dumps(schema)}

Example output shape:
{{"answer":"A direct answer supported by the evidence [E1].",\
"citation_ids":["E1"],"insufficient_evidence":false}}

Now answer the question. Return JSON only.
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

    schema = _ModelAnswer.model_json_schema()
    generation = (client or OllamaClient()).generate(
        prompt=_prompt(query, list(evidence), schema), schema=schema
    )
    parsed = _ModelAnswer.model_validate_json(generation.content)

    by_id = {f"E{index}": item for index, item in enumerate(evidence, start=1)}
    unknown = set(parsed.citation_ids) - set(by_id)
    if unknown:
        raise ValueError(f"model cited unknown evidence: {sorted(unknown)}")
    if parsed.insufficient_evidence and parsed.citation_ids:
        raise ValueError("an insufficient-evidence answer must not include citations")
    if not parsed.insufficient_evidence and not parsed.citation_ids:
        raise ValueError("a grounded answer must include citations")

    answer_text = parsed.answer
    if not parsed.insufficient_evidence:
        answer_text = _normalize_inline_citations(answer_text, parsed.citation_ids)

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
        text=answer_text,
        citations=citations,
        insufficient_evidence=parsed.insufficient_evidence,
        model=generation.model,
        total_duration_ns=generation.total_duration_ns,
    )
