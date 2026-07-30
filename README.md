# Enclave

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
uv run enclave-ingest ./corpora/postgres-docs
uv run enclave-api
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

`docker compose --profile release up` runs everything containerized for the Linux target, so production stays fully containerized without hurting laptop development.

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
  ingest/                   parse → structure-aware chunk → dedup → embed     [TODO]
  rank/                     rerank orchestration                              [TODO]
  answer/                   synthesis + reranker-based verifier               [TODO]
  api/                      FastAPI routes                                    [TODO]
  eval/                     BEIR harness, ablations, index benchmark          [TODO]
benchmarks/                 committed result tables (regressions visible in diffs)
```

`[TODO]` modules are intentionally unwritten. The scaffold provides the pieces where the API is non-obvious or a design decision is encoded; the rest is the build.

## Notes worth reading before Sprint 1

- **`ENCLAVE_EMBED_DIM` must match `vector(N)`** in `sql/001_schema.sql`. Changing it means re-embedding the corpus and rebuilding the HNSW index, so pick one dimension per deployment.
- **`ts_rank_cd` is not BM25.** `rank_bm25` is kept in `eval/` as a scoring comparison so the difference is measured, not assumed.
- **Qwen3-Reranker is a causal LM, not a classifier.** The score is the probability it assigns to `yes` versus `no` at the final position. Verify the prompt template against the current model card — it is a model-specific contract.
- **Qwen3-Embedding is instruction-aware.** Queries get a task instruction, documents do not. Skipping this quietly costs retrieval quality.
