# PostgreSQL core retrieval ablation

Measured on the `portable` profile against the 10-case PostgreSQL core golden
set. Nine questions are answerable from the corpus; one is deliberately
out-of-domain. The first request in each mode can include model loading, so warm
latency is the useful interactive comparison.

| Pipeline | Hit rate | MRR | NDCG@10 | Warm mean | Main observation |
|---|---:|---:|---:|---:|---|
| Lexical only | 44.4% | 0.444 | 0.434 | <1 ms | Exact terms are fast but brittle |
| Dense only | 100% | 0.944 | 0.941 | ~40 ms | Best quality/latency baseline |
| Hybrid, no reranker | 100% | 0.901 | 0.899 | ~35 ms | One easy query fell to rank 9 |
| Selective reranker | 100% | 1.000 | 0.976 | ~477 ms | Reranked 20% of queries |
| Always rerank | 100% | 1.000 | 0.973 | ~3.85 s | High quality, high latency cost |

Raw machine-specific output is written to `benchmarks/raw/` and intentionally
ignored by Git. The committed table is a reviewable baseline, not a broad
accuracy claim: ten phrase-judged questions are too small for that.

The selective policy trusts a fused top result when either lexical or dense
retrieval independently ranked it first. RRF score margins were rejected as a
gate because observed margins were only 0.8%--2.4%, so the old 35% threshold
never skipped the cross-encoder.

Run it again with:

```bash
uv run enclave-eval --mode all \
  --output benchmarks/raw/postgres-ablation.json
```
