"""Shared fixtures.

The important trick here is `fake_embedder`: it lets the hybrid retrieval
SQL and the RRF fusion be tested with deterministic vectors and no model
weights at all. Downloading 2.5 GB of encoders to assert a SQL join would
be absurd, and CI runners should never need to.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

# Must be set before enclave.config is imported anywhere.
os.environ.setdefault("ENCLAVE_OFFLINE_ONLY", "1")

DIM = 256  # must match vector(N) in sql/001_schema.sql


def unit(*, axis: int, dim: int = DIM, noise: float = 0.0) -> np.ndarray:
    """A deterministic unit vector pointing mostly along one axis."""
    v = np.zeros(dim, dtype=np.float32)
    v[axis] = 1.0
    if noise:
        rng = np.random.default_rng(axis)
        v = v + noise * rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def fake_embedder(monkeypatch):
    """Replace the real encoder with a fixed query -> vector mapping.

    Queries not in the map fall back to a stable hash-derived vector, so
    tests never silently depend on network access or model weights.
    """
    query_vectors: dict[str, np.ndarray] = {}

    class FakeEmbedder:
        dim = DIM

        def encode_queries(self, texts, batch_size: int = 16):
            out = []
            for t in texts:
                if t in query_vectors:
                    out.append(query_vectors[t])
                else:
                    seed = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
                    out.append(unit(axis=seed % DIM))
            return np.stack(out)

        def encode_documents(self, texts, batch_size: int = 16):
            return self.encode_queries(texts)

    fake = FakeEmbedder()
    fake.query_vectors = query_vectors  # type: ignore[attr-defined]
    monkeypatch.setattr("enclave.models.encoders.get_embedder", lambda *a, **k: fake)
    return fake


@pytest.fixture
def db_conn():
    """A migrated, empty database. Skips cleanly when Postgres is absent."""
    psycopg = pytest.importorskip("psycopg")
    from enclave import db

    try:
        conn = db.connect()
    except psycopg.OperationalError as exc:  # pragma: no cover
        pytest.skip(f"Postgres not reachable: {exc}")

    db.reset(conn)
    db.migrate(conn)
    yield conn
    db.reset(conn)
    conn.close()


@pytest.fixture
def seeded_corpus(db_conn):
    """Three synthetic chunks chosen so the two retrieval channels disagree.

    axis 0 is the query direction, so:

      pg-both     lexical HIT + nearest in vector space -> found by both
      pg-lexical  lexical HIT + orthogonal vector       -> lexical only
      cooking     no lexical match, near the query      -> dense only

    That is exactly the shape RRF is supposed to handle, and it makes the
    `found_by_both` flag meaningful to assert.
    """
    from enclave.retrieval.hybrid import to_vector_literal

    rows = [
        ("pg-both", "postgres gin index inverted lookup", unit(axis=0)),
        ("pg-lexical", "postgres gin index", unit(axis=7)),
        ("cooking", "slow roasted tomato recipe", unit(axis=0, noise=0.35)),
    ]

    with db_conn.cursor() as cur:
        for key, content, vec in rows:
            cur.execute(
                "INSERT INTO documents (doc_id, source_path, title, doc_type) "
                "VALUES (%s, %s, %s, 'markdown')",
                (key, f"synthetic/{key}.md", key),
            )
            cur.execute(
                "INSERT INTO chunks (doc_id, ordinal, heading_path, content, "
                "content_hash, token_count, embedding) "
                "VALUES (%s, 0, %s, %s, %s, %s, %s::vector)",
                (
                    key,
                    f"Synthetic > {key}",
                    content,
                    hashlib.sha256(content.encode()).hexdigest(),
                    len(content.split()),
                    to_vector_literal(vec),
                ),
            )
    return {"query": "postgres gin index", "query_vector": unit(axis=0), "rows": rows}
