"""Deterministic resolution of ambiguous conversational follow-up questions."""

from __future__ import annotations

import re
from dataclasses import dataclass

_REFERENCE = re.compile(
    r"\b(it|its|that|this|they|them|their|those|these|former|latter)\b",
    re.IGNORECASE,
)
_CONTINUATION = re.compile(
    r"^(and|also|but|what about|how about|which one|which ones|why is that|how so|"
    r"tell me more|explain more)\b",
    re.IGNORECASE,
)
_EXPLICIT_DEICTIC_TOPIC = re.compile(
    r"^(what|who)\s+(is|are)\s+(this|that|these|those)\s+\w+",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ResolvedQuestion:
    original: str
    retrieval_query: str
    contextualized: bool
    anchor: str | None = None


def is_followup(query: str) -> bool:
    text = query.strip()
    if _EXPLICIT_DEICTIC_TOPIC.search(text):
        return False
    return bool(_REFERENCE.search(text) or _CONTINUATION.search(text))


def resolve_question(query: str, previous_user_queries: list[str]) -> ResolvedQuestion:
    """Attach the latest standalone topic only when the new query refers back."""
    original = query.strip()
    if not previous_user_queries or not is_followup(original):
        return ResolvedQuestion(original, original, False)

    anchor = next(
        (
            previous.strip()
            for previous in reversed(previous_user_queries)
            if previous.strip() and not is_followup(previous)
        ),
        previous_user_queries[-1].strip(),
    )
    retrieval_query = f"Previous topic: {anchor}\nFollow-up question: {original}"
    return ResolvedQuestion(original, retrieval_query, True, anchor)
