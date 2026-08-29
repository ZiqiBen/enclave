"""Static production-deployment safety contracts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _preflight_module():
    path = ROOT / "scripts" / "production_preflight.py"
    spec = importlib.util.spec_from_file_location("production_preflight", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_production_compose_exposes_only_https_gateway():
    compose = yaml.safe_load(
        (ROOT / "docker-compose.production.yml").read_text(encoding="utf-8")
    )
    services = compose["services"]
    assert "ports" not in services["db"]
    assert "ports" not in services["redis"]
    assert "ports" not in services["api"]
    assert services["caddy"]["ports"][:2] == ["80:80", "443:443"]
    assert compose["networks"]["backend"]["internal"] is True
    assert services["api"]["environment"]["ENCLAVE_COOKIE_SECURE"] == "true"


def test_container_runs_as_non_root_and_never_copies_model_cache():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert "USER enclave" in dockerfile
    assert ".models" in ignore
    assert ".env" in ignore


def test_ci_builds_the_production_container_without_publishing_it():
    workflow = (ROOT / ".github/workflows/container.yml").read_text(encoding="utf-8")
    assert "docker/build-push-action@10e90e" in workflow
    assert "platforms: linux/amd64" in workflow
    assert "push: false" in workflow


def test_preflight_rejects_placeholder_configuration(monkeypatch, tmp_path):
    monkeypatch.setenv("ENCLAVE_DOMAIN", "enclave.example.com")
    monkeypatch.setenv("POSTGRES_PASSWORD", "replace-with-a-password")
    monkeypatch.setenv("ENCLAVE_MODEL_CACHE", str(tmp_path / "missing"))
    assert _preflight_module().main() == 1


def test_preflight_accepts_complete_configuration(monkeypatch, tmp_path):
    cache = tmp_path / "models"
    cache.mkdir()
    monkeypatch.setenv("ENCLAVE_DOMAIN", "knowledge.company.test")
    monkeypatch.setenv("POSTGRES_PASSWORD", "a-very-long-url-safe-password-42")
    monkeypatch.setenv("ENCLAVE_MODEL_CACHE", str(cache))
    assert _preflight_module().main() == 0
