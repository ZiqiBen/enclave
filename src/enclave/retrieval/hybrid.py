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

    return [
        Candidate(
            chunk_id=r[0],
            fusion_score=float(r[1]),
            lex_rank=r[2],
            dense_rank=r[3],
            doc_id=r[4],
            heading_path=r[5],
            content=r[6],
        )
        for r in rows
    ]


def should_rerank(candidates: list[Candidate]) -> bool:
    """Conditional reranking -- the cheapest large latency win.

    When stage 1 already separates the top result clearly, stage 2 rarely
    changes the ordering, so a cross-encoder pass over N passages is
    wasted. Most queries in a documentation corpus are easy.

    The margin threshold is a tunable that Sprint 2 measures: report how
    often reranking is skipped and what it costs in NDCG@10.
    """
    cfg = settings()
    if not cfg.resolved_conditional_rerank:
        return True
    if len(candidates) < 2:
        return False

    top, second = candidates[0].fusion_score, candidates[1].fusion_score
    if top <= 0:
        return True
    relative_margin = (top - second) / top
    # A decisive stage 1 that both channels agree on is trustworthy.
    return not (
        relative_margin >= cfg.conditional_rerank_margin and candidates[0].found_by_both
    )
