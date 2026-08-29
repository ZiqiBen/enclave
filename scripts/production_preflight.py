"""Fail fast when production deployment inputs are incomplete or unsafe."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import quote


def main() -> int:
    errors: list[str] = []
    domain = os.getenv("ENCLAVE_DOMAIN", "").strip().lower()
    password = os.getenv("POSTGRES_PASSWORD", "")
    model_cache_value = os.getenv("ENCLAVE_MODEL_CACHE", "").strip()
    model_cache = Path(model_cache_value) if model_cache_value else None

    if not domain or domain in {"localhost", "example.com", "enclave.example.com"}:
        errors.append("ENCLAVE_DOMAIN must be the real DNS name for this server")
    if len(password) < 24 or password.startswith("replace-"):
        errors.append(
            "POSTGRES_PASSWORD must be a unique value of at least 24 characters"
        )
    elif quote(password, safe="") != password:
        errors.append("POSTGRES_PASSWORD must use URL-safe characters")
    if model_cache is None or not model_cache.is_dir():
        errors.append("ENCLAVE_MODEL_CACHE must be an existing model-cache directory")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("production configuration passed preflight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
