"""Platform abstraction and profile resolution.

These tests carry no marker, so they run on all three operating systems in
CI. They need no model weights, no database and no network -- which is the
point: the cross-platform layer must be assertable everywhere.
"""

from __future__ import annotations

import pytest

from enclave.config import Settings, platform_tag, resolve_device


class TestDeviceResolution:
    def test_returns_a_known_device(self):
        """Must never raise, on any platform, and must return one of three
        values. This is the whole cross-platform abstraction, so if it can
        fail the project has no portability story."""
        assert resolve_device() in {"cuda", "mps", "cpu"}

    @pytest.mark.parametrize("preferred", ["cuda", "mps", "cpu"])
    def test_explicit_override_is_honoured(self, preferred):
        resolve_device.cache_clear()
        assert resolve_device(preferred) == preferred
        resolve_device.cache_clear()

    def test_nonsense_override_falls_back_to_detection(self):
        resolve_device.cache_clear()
        assert resolve_device("tpu") in {"cuda", "mps", "cpu"}
        resolve_device.cache_clear()

    def test_platform_tag_is_a_stable_label(self):
        """Benchmark rows are keyed on this, so it must be non-empty and
        contain the device -- otherwise results from a Mac and a CUDA box
        become indistinguishable in the committed tables."""
        tag = platform_tag()
        assert tag
        assert tag.split("-")[-1] in {"cuda", "mps", "cpu"}


class TestProfiles:
    def test_portable_is_the_default(self):
        """The GPU-less profile is the default on purpose. If this ever
        flips, the Mac silently gets a configuration it cannot serve."""
        assert Settings().profile == "portable"
        assert Settings().warm_models is False
        assert Settings().api_host == "127.0.0.1"
        assert Settings().api_port == 8000

    def test_portable_is_conservative_even_on_a_cuda_host(self):
        cfg = Settings(profile="portable", device="cuda")
        assert cfg.is_accelerated is False
        assert cfg.resolved_rerank_depth == 25
        assert cfg.resolved_conditional_rerank is True
        assert cfg.resolved_embed_dim == 256

    def test_accelerated_requires_cuda_not_just_the_flag(self):
        """Asking for the accelerated profile on a CPU or MPS host must not
        produce CUDA-sized defaults -- the machine could not serve them."""
        for device in ("cpu", "mps"):
            cfg = Settings(profile="accelerated", device=device)
            assert cfg.is_accelerated is False
            assert cfg.resolved_rerank_depth == 25

    def test_accelerated_on_cuda_raises_the_budget(self):
        cfg = Settings(profile="accelerated", device="cuda")
        assert cfg.is_accelerated is True
        assert cfg.resolved_rerank_depth == 100
        assert cfg.resolved_conditional_rerank is False
        assert cfg.resolved_max_passage_tokens > 288

    def test_explicit_overrides_beat_the_profile(self):
        cfg = Settings(profile="portable", rerank_depth=77, conditional_rerank=False)
        assert cfg.resolved_rerank_depth == 77
        assert cfg.resolved_conditional_rerank is False

    def test_embed_dim_must_match_the_schema_column(self):
        """sql/001_schema.sql declares vector(256). A mismatch fails at
        insert time with an opaque error, so pin the default here."""
        assert Settings(profile="portable").resolved_embed_dim == 256

    def test_small_vram_does_not_get_an_8b_model(self, monkeypatch):
        """Having a GPU is not the same as having room for an 8B model.
        A 6 GB laptop card cannot hold Llama-3.1-8B plus the encoders."""
        cfg = Settings(profile="accelerated", device="cuda")
        monkeypatch.setattr(type(cfg), "vram_gb", property(lambda self: 6.0))
        assert "8b" not in cfg.resolved_llm_model.lower()

        monkeypatch.setattr(type(cfg), "vram_gb", property(lambda self: 24.0))
        assert "8b" in cfg.resolved_llm_model.lower()

    def test_describe_records_what_a_benchmark_needs(self):
        """Every committed benchmark row needs enough config to be
        reproducible; a result without its rerank depth is meaningless."""
        d = Settings().describe()
        for key in (
            "profile",
            "platform",
            "device",
            "embed_model",
            "rerank_model",
            "embed_dim",
            "rerank_depth",
            "conditional_rerank",
        ):
            assert key in d


class TestConditionalRerank:
    """Pure logic, so it needs neither a database nor models."""

    @staticmethod
    def _candidate(score, lex, dense):
        from enclave.retrieval.hybrid import Candidate

        return Candidate(
            chunk_id=1,
            doc_id="d",
            heading_path=None,
            content="c",
            fusion_score=score,
            lex_rank=lex,
            dense_rank=dense,
        )

    def test_channel_winner_skips_reranking(self, monkeypatch):
        from enclave.retrieval import hybrid

        monkeypatch.setattr(hybrid, "settings", lambda: Settings(profile="portable"))
        cands = [self._candidate(1.0, 3, 1), self._candidate(0.99, 1, 2)]
        assert hybrid.should_rerank(cands) is False

    def test_top_hit_that_won_neither_channel_gets_reranked(self, monkeypatch):
        from enclave.retrieval import hybrid

        monkeypatch.setattr(hybrid, "settings", lambda: Settings(profile="portable"))
        cands = [self._candidate(1.0, 3, 2), self._candidate(0.95, 1, 4)]
        assert hybrid.should_rerank(cands) is True

    def test_lexical_winner_without_dense_match_is_trusted(self, monkeypatch):
        from enclave.retrieval import hybrid

        monkeypatch.setattr(hybrid, "settings", lambda: Settings(profile="portable"))
        cands = [self._candidate(1.0, 1, None), self._candidate(0.2, None, 1)]
        assert hybrid.should_rerank(cands) is False

    def test_accelerated_profile_always_reranks(self, monkeypatch):
        from enclave.retrieval import hybrid

        monkeypatch.setattr(
            hybrid, "settings", lambda: Settings(profile="accelerated", device="cuda")
        )
        cands = [self._candidate(1.0, 1, 1), self._candidate(0.1, 2, 2)]
        assert hybrid.should_rerank(cands) is True
