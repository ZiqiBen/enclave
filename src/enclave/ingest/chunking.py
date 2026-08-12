"""Deterministic, structure-aware document chunking."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from enclave.ingest.parsers import ParsedDocument


@dataclass(frozen=True, slots=True)
class Chunk:
    ordinal: int
    heading_path: str | None
    content: str
    content_hash: str
    token_count: int


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_document(
    document: ParsedDocument, *, max_words: int = 220, overlap_words: int = 40
) -> list[Chunk]:
    """Split within structural sections, with a small context overlap.

    Word counts are stored as a portable approximation. Model-specific token
    limits are enforced again by the reranker tokenizer at inference time.
    """
    if max_words < 1:
        raise ValueError("max_words must be positive")
    if overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("overlap_words must be between 0 and max_words - 1")

    chunks: list[Chunk] = []
    step = max_words - overlap_words
    for section in document.sections:
        words = section.text.split()
        for start in range(0, len(words), step):
            window = words[start : start + max_words]
            if not window:
                continue
            content = " ".join(window)
            chunks.append(
                Chunk(
                    ordinal=len(chunks),
                    heading_path=section.heading_path,
                    content=content,
                    content_hash=_hash(content),
                    token_count=len(window),
                )
            )
            if start + max_words >= len(words):
                break
    return chunks
