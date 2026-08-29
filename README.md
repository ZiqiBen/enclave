# Enclave

[![CI](https://github.com/ZiqiBen/enclave/actions/workflows/ci.yml/badge.svg)](https://github.com/ZiqiBen/enclave/actions/workflows/ci.yml)
[![Production container](https://github.com/ZiqiBen/enclave/actions/workflows/container.yml/badge.svg)](https://github.com/ZiqiBen/enclave/actions/workflows/container.yml)

**Zero-egress retrieval and answer engine. Open weights only. Windows, macOS and Linux from one codebase.**

Every model weight runs on the machine serving the request. No inference API, no managed vector database, no telemetry. That is a product requirement — the target customers cannot let documents leave their network — not a budget compromise.

| | |
|---|---|
| Retrieval | Postgres full-text search + `pgvector` (HNSW), fused with Reciprocal Rank Fusion |
| Ranking | `Qwen3-Reranker-0.6B` cross-encoder |
| Embeddings | `Qwen3-Embedding-0.6B`, Matryoshka-truncated |
| Synthesis | Ollama → `Llama-3.1-8B-Instruct` or `Qwen3-4B-Instruct` |
| Serving | FastAPI · Redis · Docker Compose |

## Two profiles, and the slow one is the default

The development fleet is deliberately uneven: a Windows workstation with an NVIDIA GPU, and a Mac without one. So `portable` is the default profile and is tuned for the machine with **no accelerator**. `accelerated` is opt-in.

Tuning defaults for CUDA would be the obvious mistake: it produces a configuration the Mac cannot serve, and since GitHub-hosted runners have no GPU, CI would never catch the drift.

```bash
ENCLAVE_PROFILE=portable      # default — CPU or Apple MPS
ENCLAVE_PROFILE=accelerated   # opt in on the CUDA box
```

| | `portable` | `accelerated` |
|---|---|---|
| Encoders | MPS (Apple Silicon) or CPU / ONNX INT8 | CUDA fp16 |
| Rerank depth | 25 | 100 |
| Passage tokens | 288 | 448 |
| Conditional rerank | on | off |
| Embedding dims | 256 | 512 |
| Synthesis | `qwen3:4b` | `llama3.1:8b-instruct-q4_K_M` |

Those depth numbers are conservative placeholders. **Sprint 1 replaces them with the measured knee on each machine** — do not ship a guess.

## Quickstart

Identical on Windows and macOS. Models run natively on the host; only Postgres and Redis are containerized (see *Why models are not containerized* below).

```bash
git clone <your-fork> && cd enclave
cp .env.example .env
uv sync --extra dev
docker compose up -d db redis
```

Pull the weights once — after this the runtime never touches the network:

```bash
ollama pull qwen3:4b
uv run python -c "from enclave.models.encoders import get_embedder, get_reranker; get_embedder(); get_reranker()"
```

Then:

```bash
uv run enclave-migrate
uv run enclave-create-user admin --admin --claim-existing
uv run enclave-ingest ./corpora/postgres-docs --owner admin
uv run enclave-api
```

For the complete interactive local deployment, use the warm entry point. It
applies pending migrations and loads the embedding, reranking, and Ollama models
before readiness succeeds:

```bash
uv run enclave-local
```

Open `http://127.0.0.1:8000/` for the end-user chat interface. Swagger remains
available at `http://127.0.0.1:8000/docs` for API-level testing.

Enclave has no public sign-up route. Create accounts locally with
`enclave-create-user`; passwords are scrypt-hashed and the command prompts
without placing them in shell history. Use `--claim-existing` once when
upgrading an existing installation so its legacy corpus belongs to that
account. Documents, search results, feedback, upload jobs, and conversations
are isolated by account. Set `ENCLAVE_COOKIE_SECURE=true` when serving over
HTTPS; leave it false for `127.0.0.1` development.

Use **Knowledge base** in the top bar to upload PDF, Markdown, HTML, or text
documents (20 MB maximum), follow background parsing/embedding progress, and
delete an uploaded document together with its indexed chunks. Uploaded source
files stay under the ignored local `data/uploads/` directory.

Every successful query is saved to PostgreSQL as a local conversation. Use
**History** to reopen the exact answer/evidence snapshot, continue the same
conversation, start a new one, or delete it. Reopening history never reruns a
model call.

Pronoun and continuation follow-ups such as “How does it prevent blocking?”
are resolved against the latest standalone user question in that conversation.
The resolved retrieval query is exposed in the API and the interface marks the
answer with **Context used**; standalone questions are never rewritten.
Contextualized queries always use the reranker because ambiguity makes a
single-channel winner less trustworthy; standalone queries retain the measured
selective-rerank policy.

Evaluate the resolver and its downstream retrieval against positive and
negative multi-turn cases with:

```bash
uv run enclave-eval-context \
  --output benchmarks/raw/postgres-conversations.json
```

The query response includes server-side stage timings so local deployments can
be tuned without guessing:

```json
{
  "timings": {
    "retrieval_ms": 408.2,
    "rerank_ms": 0.0,
    "generation_ms": 2033.1,
    "verification_ms": 576.7,
    "total_ms": 3018.1
  }
}
```

### Windows with an NVIDIA GPU

The PyPI `torch` wheel is CPU-only on Windows, so CUDA is an explicit extra step. This asymmetry is intentional — the default install yields the `portable` profile everywhere.

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Then set `ENCLAVE_PROFILE=accelerated` in `.env`.

### macOS

Nothing extra. The PyPI `torch` wheel ships with MPS support, so Apple Silicon is accelerated out of the box. Intel Macs fall back to CPU and are best-effort only.

## Why models are not containerized

**Docker on macOS cannot reach the GPU.** Docker Desktop runs a Linux VM and Metal is not passed through, so containerizing the encoders silently discards Apple Silicon acceleration and makes the reranker unusable. On Windows, containers *can* reach an NVIDIA GPU through WSL2, but only with NVIDIA hardware and extra setup.

So the split is by concern, not uniform:

| Component | Windows / macOS dev | Linux release |
|---|---|---|
| Encoders, reranker, LLM | native (uv venv + Ollama app) | container |
| API, worker | native | container |
| Postgres + pgvector, Redis | container | container |

`docker compose --profile release up` runs the API and encoders in a Linux
container while connecting to an Ollama process on the host. The separate
production compose file below adds the HTTPS gateway and hardened networking.

## Self-hosted HTTPS deployment

The production stack is separate from the laptop development compose file. It
builds the FastAPI application as a non-root container, keeps PostgreSQL and
Redis off public host ports, persists uploads and database state, and exposes
only Caddy on ports 80/443. Caddy obtains and renews the certificate for the
configured DNS name. The application container is restricted to the internal
backend network; model weights remain a read-only host mount.

On a Linux server with Docker Compose, a DNS record pointing to the server, a
populated Hugging Face cache, and Ollama already serving the configured model:

```bash
cp .env.production.example .env.production
# Edit every value, then validate it without starting services.
set -a; source .env.production; set +a
uv run python scripts/production_preflight.py
docker compose --env-file .env.production \
  -f docker-compose.production.yml up -d --build
docker compose --env-file .env.production \
  -f docker-compose.production.yml exec api enclave-create-user admin --admin
```

Ollama must listen on the Docker host interface rather than only localhost;
keep port 11434 blocked at the public firewall. The production API enables the
Secure session-cookie flag automatically. `POSTGRES_PASSWORD` must be long and
URL-safe because it is embedded in the internal connection URL. Never commit
`.env.production`.

Create a private database and upload-source backup with:

```bash
./scripts/backup_postgres.sh /srv/enclave/backups
```

Restore is intentionally explicit and destructive; stop user traffic first,
then pass the confirmation word:

```bash
./scripts/restore_postgres.sh /srv/enclave/backups/enclave-TIMESTAMP RESTORE
```

## Two guarantees, both tested

**Zero egress.** `docker compose --profile egress-test up` runs the stack on a Docker network with `internal: true` — no default route — executes a golden query set, and asserts both correct answers and zero outbound connection attempts. `ENCLAVE_OFFLINE_ONLY=1` also makes `transformers` refuse network access, so a cold cache fails loudly instead of downloading in production.

**Cross-platform.** CI runs the suite on `ubuntu-latest`, `windows-latest`, and `macos-14` (arm64). A platform-specific bug fails a build instead of surfacing during a demo.

> **CI proves correctness, not speed.** Hosted runners have no GPU. Latency is measured locally on both machines and the result tables are committed under `benchmarks/`, so a performance regression shows up in a diff. A green matrix is not a performance claim.

## Layout

```
sql/001_schema.sql          documents, chunks (tsvector + vector), feedback, eval_runs
src/enclave/
  config.py                 profiles + device resolution (the whole platform abstraction)
  models/encoders.py        Qwen3 embedder & reranker wrappers
  retrieval/hybrid.py       RRF over Postgres FTS + pgvector, in one SQL statement
  ingest/                   parse → structure-aware chunk → dedup → embed
  rank/                     conditional rerank orchestration
  answer/                   grounded synthesis + claim-level verification
  api/                      health, search, query, and feedback routes
  eval/                     golden question set + quality/latency harness
benchmarks/                 committed result tables (regressions visible in diffs)
```

The evaluation harness compares lexical-only, dense-only, hybrid, selective
reranking, and always-rerank retrieval. It measures hit rate, MRR, NDCG@10,
reranking frequency, and latency against a human-reviewed question set:

```bash
uv run enclave-eval --output benchmarks/raw/postgres-core.json
```

Real embedding and reranker integration requires locally cached weights; tests
use deterministic fakes so CI never downloads models. Ollama synthesis has been
verified locally.

## Notes worth reading before Sprint 1

- **`ENCLAVE_EMBED_DIM` must match `vector(N)`** in `sql/001_schema.sql`. Changing it means re-embedding the corpus and rebuilding the HNSW index, so pick one dimension per deployment.
- **`ts_rank_cd` is not BM25.** `rank_bm25` is kept in `eval/` as a scoring comparison so the difference is measured, not assumed.
- **Qwen3-Reranker is a causal LM, not a classifier.** The score is the probability it assigns to `yes` versus `no` at the final position. Verify the prompt template against the current model card — it is a model-specific contract.
- **Qwen3-Embedding is instruction-aware.** Queries get a task instruction, documents do not. Skipping this quietly costs retrieval quality.
