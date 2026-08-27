# Enclave engineering log

This is the durable source for the eventual resume and interview package. It
records what was built, why decisions were made, what was measured, and which
claims still need stronger evidence.

## Product goal

Enclave is a zero-egress retrieval and grounded-answer engine for private
document collections. Documents, embeddings, reranking, and answer generation
stay on infrastructure controlled by the operator. The intended product is a
private knowledge assistant, not merely a localhost demo.

## Working architecture

1. Parse HTML, Markdown, text, and PDF documents.
2. Split them into structure-aware, overlapping chunks and deduplicate content.
3. Generate 256-dimensional Matryoshka-truncated Qwen3 embeddings.
4. Store documents, chunks, full-text vectors, and dense vectors in PostgreSQL.
5. Retrieve through PostgreSQL full-text search and pgvector HNSW search.
6. Fuse channel ranks with reciprocal rank fusion (RRF).
7. Conditionally rerank uncertain candidates with Qwen3-Reranker-0.6B.
8. Generate a grounded answer through local Ollama `qwen3:4b`.
9. Normalize inline evidence IDs and verify claims against cited chunks.
10. Store explicit thumbs-up/down retrieval feedback for later hard-negative
    mining.

## Implemented milestones

### Ingestion and storage

- PostgreSQL schema includes documents, chunks, `tsvector`, pgvector, HNSW and
  GIN indexes, feedback, and evaluation-run support.
- Ingestion batches embeddings across documents instead of invoking the model
  once per document. On the PostgreSQL core corpus this reduced ingestion from
  712.57 seconds to 336.28 seconds (2.12x).
- Real corpus: PostgreSQL 16.14 core documentation, 223 files, 2,032 chunks,
  all 2,032 embedded. Three duplicate chunks were skipped and there were zero
  ingestion failures.
- Local database storage was moved from `/tmp` to ignored `data/postgres17`
  after a machine restart demonstrated that `/tmp` was not durable.
- Database tests use random temporary schemas, so running pytest cannot delete
  the real development corpus.

### Retrieval and ranking

- Lexical retrieval uses PostgreSQL full-text search and a GIN index.
- Dense retrieval uses Qwen3-Embedding-0.6B and pgvector cosine distance over
  an HNSW index.
- Hybrid retrieval performs RRF in one SQL statement, avoiding dual-store
  consistency problems.
- Qwen3 reranker input truncation preserves the model-specific prefix and
  suffix; only passage content consumes the remaining token budget.
- Evaluation baselines can run lexical-only, dense-only, hybrid-only,
  selective-rerank, or always-rerank pipelines.

### Answers and verification

- Ollama synthesis is local and currently uses `qwen3:4b`.
- Grounded answers must declare and include inline citations such as `[E1]`.
- A deterministic normalization step attaches declared evidence IDs when a
  model omits inline placement; undeclared citations are rejected.
- Sentence parsing preserves citations written after punctuation.
- Claim-level verification checks that cited evidence supports each claim.
- Verified real example: “What is PostgreSQL?” produced a cited answer with a
  support score of 0.9991.
- An out-of-domain vacation question correctly returned insufficient evidence.

### API and validation

- FastAPI exposes `/health/live`, `/health/ready`, `/v1/search`, `/v1/query`,
  and `/v1/feedback`.
- Search and query responses expose wall-clock stage timings for retrieval,
  conditional reranking, generation, verification, and the total request. The
  pre-existing Ollama-reported generation duration is retained separately.
- Swagger is available locally at `http://127.0.0.1:8000/docs`.
- Current suite: 96 passing tests plus Ruff lint and formatting checks.
- One non-blocking dependency warning remains: Starlette's current test client
  compatibility layer recommends migration to `httpx2`.

## Retrieval experiments

The initial golden set has ten PostgreSQL questions: nine answerable and one
out-of-domain. Relevance is judged through human-curated phrases, so these are
regression numbers, not a claim of general 100% accuracy.

| Pipeline | Hit rate | MRR | NDCG@10 | Warm mean latency |
|---|---:|---:|---:|---:|
| Lexical only | 44.4% | 0.444 | 0.434 | 0.52 ms |
| Dense only | 100% | 0.944 | 0.941 | 39.68 ms |
| Hybrid only | 100% | 0.901 | 0.899 | 35.00 ms |
| Selective reranker | 100% | 1.000 | 0.976 | 476.88 ms |
| Always rerank | 100% | 1.000 | 0.973 | 3,848.35 ms |

Selective reranking reduced the rerank rate from 100% to 20%. Across all ten
queries, mean retrieval latency fell from 3,846.74 ms to 953.79 ms (75.2%)
while MRR stayed at 1.0 and NDCG@10 did not regress.

The live `/v1/search` route was restarted and checked after the change. A warm,
trusted database-cluster query returned with `reranked=false` in 167 ms over
HTTP. The uncertain “What is PostgreSQL?” path returned `reranked=true`, put
the Concepts evidence first with a reranker score of 0.9979, and took 4.66
seconds. This confirms that the policy is active in the served product path,
not only inside the offline benchmark.

### Why the first conditional rule failed

The original policy required a 35% relative gap between the first two RRF
scores. Real observed gaps were only 0.8%--2.4%, because reciprocal-rank scores
are compressed by construction. The condition therefore never fired.

The replacement policy trusts the fused winner when either lexical or dense
retrieval independently ranked that chunk first. Otherwise it reranks. On the
current set this reranks “What is PostgreSQL?” (whose correct hybrid evidence
was initially rank 9) and the GIN question, while skipping eight queries.

## Honest limitations and next evidence needed

- Ten questions are insufficient for a production accuracy claim. Expand to
  30--50 manually reviewed questions, then to a larger judged dataset.
- Phrase-based relevance can miss semantically correct passages. Add stable
  chunk/document judgments reviewed by a human.
- The benchmark runs modes sequentially, so cold-start time depends on order;
  warm means are reported separately for fair interaction comparisons.
- Full `/v1/query` latency also includes Ollama generation and claim
  verification. It is measured separately below rather than inferred from
  retrieval-only benchmarks.
- Multi-user authentication, tenant isolation, upload UI, background jobs,
  HTTPS, monitoring, backups, and production deployment remain future work.

## Interview evidence to preserve

- Explain why one PostgreSQL system was chosen over PostgreSQL plus FAISS:
  transactional consistency, filters and vectors together, recognizable
  operations, and cross-platform simplicity.
- Explain why `ts_rank_cd` is not called BM25; the project measures alternatives
  rather than making an inaccurate claim.
- Explain the latency/quality experiment that rejected RRF score margins and
  produced the channel-winner selective policy.
- Explain how a real restart exposed ephemeral database storage and motivated a
  durable ignored data directory.
- Explain why model-specific token framing must survive reranker truncation.
- Use measured numbers only, and identify the ten-question set as a regression
  benchmark rather than broad accuracy evidence.

## Local end-to-end deployment measurements

The native macOS development deployment was restarted after instrumentation
with PostgreSQL on port 5433, Ollama on 11434, and FastAPI on 8000. All three
real requests returned verified cited answers.

| Request | Retrieval | Rerank | Generation | Verification | Total |
|---|---:|---:|---:|---:|---:|
| Cold, trusted cluster query | 4,504.7 ms | 0 ms | 7,796.0 ms | 1,267.3 ms | 13,568.0 ms |
| Warm, uncertain PostgreSQL query | 296.6 ms | 4,249.3 ms | 5,070.3 ms | 503.5 ms | 10,120.1 ms |
| Warm, trusted cluster query | 408.2 ms | 0 ms | 2,033.1 ms | 576.7 ms | 3,018.1 ms |

The warm trusted path is now about 3.0 seconds end to end. Generation is the
largest remaining cost on that path; reranking dominates only uncertain
queries. Cold model initialization remains a separate deployment concern and
should be handled by startup warm-up before production traffic.

## Local startup warm-up

`uv run enclave-local` is the production-like local entry point. It applies
pending database migrations, executes a real query embedding, executes a
one-document reranker forward pass, asks Ollama to load and pin the configured
LLM, and only then completes FastAPI startup. `/health/ready` reports `models:
warm` plus the per-model startup durations.

The first verified cluster query after enabling warm-up took 6.81 seconds,
versus 13.57 seconds on the earlier fully cold path. Retrieval fell from 4.50
seconds to 0.49 seconds. Generation still took 5.91 seconds because loading LLM
weights does not eliminate evaluation of the real evidence prompt; subsequent
generation remains the next optimization target.

## Local end-user interface

The FastAPI process now serves a zero-dependency chat interface at `/`; the
Swagger page remains available at `/docs`. The interface reports whether local
models are warm, offers PostgreSQL example questions, sends real `/v1/query`
requests, renders verified answers without injecting model-authored HTML,
expands the top five evidence passages, shows retrieval/generation/total stage
timings, and persists useful/not-useful votes through `/v1/feedback`.

The layout is responsive and keyboard accessible: Enter submits, Shift+Enter
adds a line, controls have accessible labels, and reduced-motion preferences
are respected. It loads no fonts, scripts, styles, images, analytics, or CDN
resources from outside the local service. A built wheel was inspected to
confirm `index.html`, `app.css`, and `app.js` are included in the distributable
package. At this milestone the suite contains 99 passing tests.

## Upload and knowledge-base management

The local interface accepts PDF, Markdown, HTML, and text uploads up to 20 MB.
Names are Unicode-normalized, flattened to a basename, filtered to safe
characters, and checked against the parser allowlist before a directory is
created. Every job receives a random ID and its own directory beneath the
configured ignored upload root, so same-name files cannot overwrite each
other.

An upload returns HTTP 202 with a persistent PostgreSQL job immediately. A
FastAPI background task then records parsing, embedding, completion, or failure
state. The interface polls only while work is active. Jobs interrupted by a
service restart are marked failed on startup instead of remaining permanently
in progress. Active jobs cannot be deleted; completed deletion cascades through
the document's chunks and embeddings, removes the job record, and removes only
the job directory after verifying it is inside the configured upload root.

A real README upload completed with nine embedded chunks and no error. Its API
deletion returned 204, the document/job list returned to zero, and the isolated
upload directory was removed. Unsupported-extension behavior, path flattening,
active-job protection, cascade deletion, restart recovery, UI routes, and the
existing retrieval stack are covered by the automated suite.
