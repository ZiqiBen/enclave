"""Configuration and platform resolution.

Two profiles exist. `portable` is the DEFAULT and is tuned for a machine
with no accelerator -- that is the whole point: the product must be
serviceable on the weakest machine in the fleet, and CUDA is an opt-in
speedup rather than a requirement.

Select with ENCLAVE_PROFILE=accelerated (or set individual overrides).
"""

from __future__ import annotations

import platform
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Profile = Literal["portable", "accelerated"]
Device = Literal["cuda", "mps", "cpu"]


@lru_cache(maxsize=1)
def resolve_device(preferred: str | None = None) -> Device:
    """The entire platform abstraction.

    Windows/Linux with NVIDIA -> cuda. Apple Silicon -> mps. Everything
    else -> cpu. Intel Macs land on cpu because MPS is unavailable there,
    which is correct rather than a bug.
    """
    import torch

    if preferred in ("cuda", "mps", "cpu"):
        return preferred  # type: ignore[return-value]
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def platform_tag() -> str:
    """Stable label for benchmark rows, e.g. 'darwin-mps', 'windows-cuda'."""
    return f"{platform.system().lower()}-{resolve_device()}"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ENCLAVE_", env_file=".env", extra="ignore"
    )

    profile: Profile = "portable"

    # --- models -------------------------------------------------------
    embed_model: str = "Qwen/Qwen3-Embedding-0.6B"
    rerank_model: str = "Qwen/Qwen3-Reranker-0.6B"
    # Baselines for the model-selection study (Sprint 1).
    embed_baseline: str = "BAAI/bge-m3"
    rerank_baseline: str = "BAAI/bge-reranker-v2-m3"

    # Synthesis runs through Ollama, whose HTTP API is identical on
    # Windows, macOS and Linux.
    ollama_url: str = "http://127.0.0.1:11434"
    llm_model: str = ""  # empty -> profile default, see resolved_llm_model

    # --- retrieval knobs ----------------------------------------------
    # Matryoshka truncation. MUST match the vector(N) column in
    # sql/001_schema.sql; changing it means re-embedding + reindexing.
    embed_dim: int = 0  # 0 -> profile default
    candidates_k: int = 100  # per channel, before fusion
    rrf_k: int = 60  # RRF smoothing constant
    rerank_depth: int = 0  # 0 -> profile default
    max_passage_tokens: int = 0  # 0 -> profile default
    conditional_rerank: bool | None = None  # None -> profile default
    # --- infrastructure -----------------------------------------------
    database_url: str = "postgresql://enclave:enclave@127.0.0.1:5433/enclave"
    redis_url: str = "redis://127.0.0.1:6380/0"
    device: str | None = None  # override; normally auto-resolved
    warm_models: bool = False  # enclave-local enables this before startup
    upload_dir: Path = Path("data/uploads")
    max_upload_mb: int = 20
    session_hours: int = 24
    cookie_secure: bool = False  # enable behind HTTPS in a deployed environment
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # --- guarantees ---------------------------------------------------
    # When true, model loaders refuse any network fetch and require a
    # populated local cache. The egress test sets this.
    offline_only: bool = True

    interactive_p99_budget_s: float = 2.0

    # ------------------------------------------------------------------
    @property
    def resolved_device(self) -> Device:
        return resolve_device(self.device)

    @property
    def is_accelerated(self) -> bool:
        return self.profile == "accelerated" and self.resolved_device == "cuda"

    @property
    def resolved_embed_dim(self) -> int:
        if self.embed_dim:
            return self.embed_dim
        return 512 if self.is_accelerated else 256

    @property
    def resolved_rerank_depth(self) -> int:
        if self.rerank_depth:
            return self.rerank_depth
        # Conservative on purpose. Sprint 1 replaces these with the
        # measured knee for each machine -- do not guess in production.
        return 100 if self.is_accelerated else 25

    @property
    def resolved_max_passage_tokens(self) -> int:
        if self.max_passage_tokens:
            return self.max_passage_tokens
        return 448 if self.is_accelerated else 288

    @property
    def resolved_conditional_rerank(self) -> bool:
        if self.conditional_rerank is not None:
            return self.conditional_rerank
        return not self.is_accelerated

    @property
    def vram_gb(self) -> float:
        """Total VRAM on the resolved device, 0.0 when there is none."""
        if self.resolved_device != "cuda":
            return 0.0
        import torch

        return torch.cuda.get_device_properties(0).total_memory / 1024**3

    @property
    def resolved_llm_model(self) -> str:
        if self.llm_model:
            return self.llm_model
        # Having a GPU is not the same as having room for an 8B model.
        # Llama-3.1-8B at Q4 is ~4.9 GB and the encoders want ~2.4 GB in
        # fp16, so the two only coexist above roughly 10 GB of VRAM. On a
        # 6 GB laptop card the 8B model either OOMs or evicts the encoders,
        # which ends up slower than simply using the 4B model.
        # Sprint 3 compares them properly; this is only the default.
        if self.vram_gb >= 10:
            return "llama3.1:8b-instruct-q4_K_M"
        return "qwen3:4b"

    @property
    def torch_dtype(self):
        import torch

        return torch.float16 if self.resolved_device == "cuda" else torch.float32

    def describe(self) -> dict[str, object]:
        """Config snapshot recorded with every benchmark run."""
        return {
            "profile": self.profile,
            "platform": platform_tag(),
            "device": self.resolved_device,
            "embed_model": self.embed_model,
            "rerank_model": self.rerank_model,
            "llm_model": self.resolved_llm_model,
            "embed_dim": self.resolved_embed_dim,
            "rerank_depth": self.resolved_rerank_depth,
            "max_passage_tokens": self.resolved_max_passage_tokens,
            "conditional_rerank": self.resolved_conditional_rerank,
        }


@lru_cache(maxsize=1)
def settings() -> Settings:
    return Settings()
