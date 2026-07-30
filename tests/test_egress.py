"""The zero-egress guarantee, as a test rather than a claim.

Two layers, deliberately:

1. Unmarked tests install a socket guard and assert that no code path even
   *attempts* an outbound connection. These run on all three operating
   systems in CI, because "we never try" is a stronger and far cheaper
   property to check than "the firewall stopped us".

2. Tests marked `egress` run inside the container on a Docker network with
   `internal: true` -- no default route at all -- and answer real queries.
   That is the end-to-end proof, and it needs models plus a database.

If the guard below ever fires, something in the stack acquired a network
dependency and the product promise is broken. Do not skip it.
"""

from __future__ import annotations

import socket

import pytest

# Loopback is legitimate: Postgres, Redis and Ollama all live on the host.
_ALLOWED_HOSTS = {
    "127.0.0.1",
    "::1",
    "localhost",
    "db",
    "redis",
    "host.docker.internal",
}


class EgressAttempted(AssertionError):
    """Raised the moment anything reaches for a non-local address."""


@pytest.fixture
def no_egress(monkeypatch):
    """Fail loudly on any outbound connection to a non-local address.

    Patches socket at the lowest useful level, so it catches httpx,
    requests, urllib and huggingface_hub alike without knowing about them.
    """
    attempts: list[str] = []
    real_connect = socket.socket.connect
    real_getaddrinfo = socket.getaddrinfo

    def guarded_connect(self, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else str(address)
        if host not in _ALLOWED_HOSTS:
            attempts.append(host)
            raise EgressAttempted(f"outbound connection attempted to {host!r}")
        return real_connect(self, address, *args, **kwargs)

    def guarded_getaddrinfo(host, *args, **kwargs):
        if host not in _ALLOWED_HOSTS:
            attempts.append(str(host))
            raise EgressAttempted(f"DNS resolution attempted for {host!r}")
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    return attempts


class TestOfflineEnforcement:
    def test_offline_flags_are_set_by_the_loader(self, monkeypatch):
        """transformers and huggingface_hub must be put into offline mode
        before any model loads, so a cold cache fails loudly instead of
        quietly downloading weights in production."""
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

        import os

        from enclave.models.encoders import _enforce_offline

        _enforce_offline()
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"

    def test_offline_only_defaults_to_on(self):
        """A deployment that has to remember to turn this on will forget."""
        from enclave.config import Settings

        assert Settings().offline_only is True


class TestGuardItself:
    """A guard that cannot fail proves nothing, so prove it can."""

    def test_guard_blocks_a_public_address(self, no_egress):
        with pytest.raises(EgressAttempted):
            socket.create_connection(("example.com", 443), timeout=1)

    def test_guard_permits_loopback(self, no_egress):
        # Nothing is necessarily listening; we only care that the guard lets
        # the attempt through rather than rejecting the address itself.
        with pytest.raises((ConnectionRefusedError, OSError)) as exc:
            socket.create_connection(("127.0.0.1", 59999), timeout=1)
        assert not isinstance(exc.value, EgressAttempted)


class TestNoEgressInImportPath:
    def test_importing_the_package_touches_no_network(self, no_egress):
        """Import-time network access is the classic accidental dependency
        -- a telemetry client or a model-card fetch in a module body."""
        import importlib

        for name in (
            "enclave.config",
            "enclave.db",
            "enclave.retrieval.hybrid",
            "enclave.models.encoders",
        ):
            importlib.import_module(name)
        assert no_egress == []

    def test_settings_and_device_resolution_touch_no_network(self, no_egress):
        from enclave.config import Settings, platform_tag

        Settings().describe()
        platform_tag()
        assert no_egress == []


@pytest.mark.egress
@pytest.mark.db
@pytest.mark.models
class TestEndToEndOffline:
    """Runs in the container on the `internal: true` network.

    TODO(Sprint 1): replace the placeholder with the golden query set once
    ingestion exists. Each case asserts (a) the expected passage is in the
    top-k and (b) no egress was attempted while answering.
    """

    def test_search_answers_correctly_with_no_route_out(
        self, db_conn, seeded_corpus, no_egress
    ):
        from enclave.retrieval.hybrid import hybrid_search

        results = hybrid_search(db_conn, seeded_corpus["query"], limit=5)
        assert results, "retrieval returned nothing with egress blocked"
        assert no_egress == []
