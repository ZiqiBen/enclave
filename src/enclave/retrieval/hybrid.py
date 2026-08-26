"""Stage 1: hybrid retrieval, both halves inside Postgres.

The lexical channel is Postgres full-text search over a GIN index; the
dense channel is pgvector over an HNSW index. Fusion is Reciprocal Rank
Fusion, computed in the same SQL statement.

Why one database instead of Postgres + FAISS:

  * identical behaviour on Windows, macOS and Linux -- one container
    image, no arm64 wheel roulette
  * filters, metadata and vectors in a single transaction, so there is no
    dual-write consistency problem to invent
  * pgvector is a stack an interviewer recognises immediately

FAISS and usearch still appear in the index benchmark (Sprint 2) as
comparison points. They are just not load-bearing dependencies.

Note: ts_rank_cd is NOT literally BM25. A rank_bm25 implementation is
kept in eval/ as a scoring comparison, so the difference is measured
rather than assumed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from enclave.config import settings


def to_vector_literal(vec: Sequence[float]) -> str:
    """pgvector's text input format, e.g. '[0.1,0.2,0.3]'.

    Passing the literal rather than a numpy array means the function does
    not depend on register_vector() having been called on the connection,
    which keeps it usable from a bare psycopg connection in tests.
    """
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"


# RRF over two channels. Each contributes 1/(k + rank); documents found by
# both accumulate both terms, which is the whole point.
#
# websearch_to_tsquery is used because it tolerates raw user input
# (quotes, OR, negation) instead of raising on punctuation the way
# to_tsquery does.
_HYBRID_SQL = """
WITH q AS (
    SELECT websearch_to_tsquery('english', %(query)s) AS tsq
),
lexical AS (
    SELECT c.id,
           ROW_NUMBER() OVER (ORDER BY ts_rank_cd(c.tsv, q.tsq) DESC) AS rnk
    FROM chunks c, q
    WHERE c.tsv @@ q.tsq
    ORDER BY ts_rank_cd(c.tsv, q.tsq) DESC
    LIMIT %(k)s
),
dense AS (
    SELECT c.id,
           ROW_NUMBER() OVER (ORDER BY c.embedding <=> %(vec)s::vector) AS rnk
    FROM chunks c
    WHERE c.embedding IS NOT NULL
    ORDER BY c.embedding <=> %(vec)s::vector
    LIMIT %(k)s
),
fused AS (
    SELECT COALESCE(l.id, d.id) AS id,
           COALESCE(1.0 / (%(rrf_k)s + l.rnk), 0.0)
         + COALESCE(1.0 / (%(rrf_k)s + d.rnk), 0.0) AS score,
           l.rnk AS lex_rank,
           d.rnk AS dense_rank
    FROM lexical l
    FULL OUTER JOIN dense d ON d.id = l.id
)
SELECT f.id, f.score, f.lex_rank, f.dense_rank,
       c.doc_id, c.heading_path, c.content
FROM fused f
JOIN chunks c ON c.id = f.id
ORDER BY f.score DESC
LIMIT %(limit)s;
"""

_LEXICAL_SQL = """
WITH ranked AS (
    SELECT c.id,
           ROW_NUMBER() OVER (
               ORDER BY ts_rank_cd(
                   c.tsv, websearch_to_tsquery('english', %(query)s)
               ) DESC
           ) AS rnk
    FROM chunks c
    WHERE c.tsv @@ websearch_to_tsquery('english', %(query)s)
    ORDER BY ts_rank_cd(
        c.tsv, websearch_to_tsquery('english', %(query)s)
    ) DESC
    LIMIT %(limit)s
)
SELECT r.id, 1.0 / (%(rrf_k)s + r.rnk), r.rnk, NULL,
       c.doc_id, c.heading_path, c.content
FROM ranked r
JOIN chunks c ON c.id = r.id
ORDER BY r.rnk;
"""

_DENSE_SQL = """
WITH ranked AS (
    SELECT c.id,
           ROW_NUMBER() OVER (ORDER BY c.embedding <=> %(vec)s::vector) AS rnk
    FROM chunks c
    WHERE c.embedding IS NOT NULL
    ORDER BY c.embedding <=> %(vec)s::vector
    LIMIT %(limit)s
)
SELECT r.id, 1.0 / (%(rrf_k)s + r.rnk), NULL, r.rnk,
       c.doc_id, c.heading_path, c.content
FROM ranked r
JOIN chunks c ON c.id = r.id
ORDER BY r.rnk;
"""


@dataclass(slots=True)
class Candidate:
    chunk_id: int
    doc_id: str
    heading_path: str | None
    content: str
    fusion_score: float
    lex_rank: int | None
    dense_rank: int | None
    rerank_score: float | None = None

    @property
    def found_by_both(self) -> bool:
        """Channel agreement. Disagreements are the interesting cases --
        they become hard negatives for Sprint 4 mining."""
        return self.lex_rank is not None and self.dense_rank is not None


def _candidates(rows) -> list[Candidate]:
    return [
        Candidate(
            chunk_id=row[0],
            fusion_score=float(row[1]),
            lex_rank=row[2],
            dense_rank=row[3],
            doc_id=row[4],
            heading_path=row[5],
            content=row[6],
        )
        for row in rows
    ]


def lexical_search(conn, query: str, limit: int = 10) -> list[Candidate]:
    """Postgres full-text retrieval alone, used as an evaluation baseline."""
    with conn.cursor() as cur:
        cur.execute(
            _LEXICAL_SQL,
            {"query": query, "limit": limit, "rrf_k": settings().rrf_k},
        )
        return _candidates(cur.fetchall())


def dense_search(conn, query: str, limit: int = 10) -> list[Candidate]:
    """pgvector retrieval alone, used as an evaluation baseline."""
    from enclave.models.encoders import get_embedder

    vec = get_embedder().encode_queries([query])[0]
    with conn.cursor() as cur:
        cur.execute(
            _DENSE_SQL,
            {
                "vec": to_vector_literal(vec),
                "limit": limit,
                "rrf_k": settings().rrf_k,
            },
        )
        return _candidates(cur.fetchall())


def hybrid_search(conn, query: str, limit: int | None = None) -> list[Candidate]:
    """Lexical + dense retrieval fused by RRF.

    `conn` is a psycopg connection. Embedding the query is one forward
    pass of the 0.6B encoder; everything else is one round trip.
    """
    from enclave.models.encoders import get_embedder

    cfg = settings()
    vec = get_embedder().encode_queries([query])[0]

    with conn.cursor() as cur:
        cur.execute(
            _HYBRID_SQL,
            {
                "query": query,
                "vec": to_vector_literal(vec),
                "k": cfg.candidates_k,
                "rrf_k": cfg.rrf_k,
                "limit": limit or cfg.resolved_rerank_depth,
            },
        )
        rows = cur.fetchall()

    return _candidates(rows)


def should_rerank(candidates: list[Candidate]) -> bool:
    """Conditional reranking -- the cheapest large latency win.

    RRF scores are sums of reciprocal ranks, so adjacent scores are naturally
    close and a conventional score-margin rule almost never fires. Instead,
    trust the top fused result when either underlying channel independently
    ranked it first. If neither channel did, the fusion order is uncertain and
    the cross-encoder earns its cost.
    """
    cfg = settings()
    if not cfg.resolved_conditional_rerank:
        return True
    if not candidates:
        return False
    top = candidates[0]
    return top.lex_rank != 1 and top.dense_rank != 1
