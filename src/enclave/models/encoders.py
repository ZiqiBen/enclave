"""Encoder wrappers: dense embedder and cross-encoder reranker.

Both load onto whatever device resolve_device() found, so the same code
runs on CUDA, MPS and CPU with no branching beyond the device string.

Two details here are easy to get wrong, which is why they are in the
scaffold rather than left as an exercise:

1. Qwen3-Embedding is *instruction-aware*. Queries and documents must be
   encoded differently -- queries carry a task instruction, documents do
   not. Skipping this silently costs real retrieval quality.

2. Qwen3-Reranker is not a classification head. It is a causal LM that
   you prompt, and the relevance score is the probability it assigns to
   the token "yes" versus "no" at the final position.

VERIFY the prompt template and pooling behaviour against the current
model cards before Sprint 1; both are model-specific contracts.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from enclave.config import settings

# Qwen3-Reranker chat scaffolding. Taken from the model card -- re-check it.
_RERANK_SYSTEM = (
    "<|im_start|>system\nJudge whether the Document meets the requirements "
    "based on the Query and the Instruct provided. Note that the answer can "
    'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
_RERANK_ASSISTANT = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

_DEFAULT_INSTRUCTION = (
    "Given a user question, retrieve documentation passages that answer it"
)


def _enforce_offline() -> None:
    """Zero-egress guarantee, enforced at the library level.

    With offline_only set, transformers and huggingface_hub refuse network
    access and fail loudly if the local cache is cold -- which is what we
    want, rather than a silent download in production.
    """
    # `uv run` does not export arbitrary keys from .env into os.environ.
    # Give native development the documented project-local cache by default;
    # release containers explicitly set HF_HOME=/models and win via setdefault.
    os.environ.setdefault("HF_HOME", str(Path(".models").resolve()))
    if settings().offline_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class Embedder:
    """Dense retrieval encoder with Matryoshka truncation."""

    def __init__(self, model_name: str | None = None, dim: int | None = None):
        _enforce_offline()
        from sentence_transformers import SentenceTransformer

        cfg = settings()
        self.dim = dim or cfg.resolved_embed_dim
        self.model = SentenceTransformer(
            model_name or cfg.embed_model,
            device=cfg.resolved_device,
            model_kwargs={"dtype": cfg.torch_dtype},
            # Qwen3-Embedding pools the last token, so padding must be left.
            processor_kwargs={"padding_side": "left"},
            truncate_dim=self.dim,  # Matryoshka: 1024 -> self.dim
        )

    def encode_queries(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        # prompt_name="query" applies the model's registered query
        # instruction. Documents deliberately get no prompt.
        return self.model.encode(
            texts,
            prompt_name="query",
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def encode_documents(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )


class Reranker:
    """Cross-encoder relevance scorer.

    Also reused as the answer verifier: scoring (generated sentence,
    cited passage) is the same operation as scoring (query, passage), so
    the system needs no fourth model.
    """

    def __init__(self, model_name: str | None = None):
        _enforce_offline()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = settings()
        name = model_name or cfg.rerank_model
        self.device = cfg.resolved_device
        self.max_tokens = cfg.resolved_max_passage_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
        self.model = (
            AutoModelForCausalLM.from_pretrained(name, dtype=cfg.torch_dtype)
            .to(self.device)
            .eval()
        )
        # The two tokens whose logits become the score.
        self.no_id = self.tokenizer.convert_tokens_to_ids("no")
        self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self._torch = torch

    def _build_prompt(self, query: str, document: str, instruction: str) -> str:
        return (
            f"{_RERANK_SYSTEM}<Instruct>: {instruction}\n"
            f"<Query>: {query}\n<Document>: {document}{_RERANK_ASSISTANT}"
        )

    def score(
        self,
        query: str,
        documents: list[str],
        instruction: str = _DEFAULT_INSTRUCTION,
        batch_size: int = 8,
    ) -> list[float]:
        """Relevance in [0, 1] per document, input order preserved.

        Cost is one forward pass per document, which is why rerank depth
        is the dominant latency knob in the whole system.
        """
        if not documents:
            return []

        torch = self._torch
        prompts = [self._build_prompt(query, d, instruction) for d in documents]
        scores: list[float] = []

        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_tokens,
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**enc).logits[:, -1, :]

            pair = torch.stack([logits[:, self.no_id], logits[:, self.yes_id]], dim=1)
            probs = torch.softmax(pair.float(), dim=1)[:, 1]
            scores.extend(probs.cpu().tolist())

        return scores


@lru_cache(maxsize=2)
def get_embedder(model_name: str | None = None) -> Embedder:
    return Embedder(model_name)


@lru_cache(maxsize=2)
def get_reranker(model_name: str | None = None) -> Reranker:
    return Reranker(model_name)
