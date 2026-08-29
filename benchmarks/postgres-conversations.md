# PostgreSQL conversational retrieval evaluation

Measured on the `portable` profile against 16 reviewed multi-turn cases: eight
contextual follow-ups, five explicit topic switches, one ambiguous query without
history, and two lexical traps containing `this`/`it`.

| Metric | Initial rule | Corrected decisions | Context always reranked |
|---|---:|---:|---:|
| Context decision accuracy | 87.5% | 100% | 100% |
| Anchor accuracy | 87.5% | 100% | 100% |
| Retrieval hit rate | 100% | 100% | 100% |
| MRR | 0.901 | 0.901 | 0.964 |
| Rerank rate | 18.8% | 12.5% | 62.5% |
| Mean retrieval latency | 1,259 ms | 2,270 ms | 3,088 ms |
| P95 retrieval latency | 5,999 ms | 9,836 ms | 10,980 ms |

The initial rule missed “Which one is the default?” and incorrectly treated
“What is this database cluster concept?” as a reference to earlier history.
Explicit `which one(s)` handling and a named deictic-topic exception fixed both
without making `Git` match the standalone word `it`.

After the decision fixes, relevant EXPLAIN ANALYZE evidence still appeared at
rank 9. Always reranking contextual queries moved it to rank 2 and moved the
MVCC follow-up from rank 2 to rank 1. Standalone queries retain selective
reranking. The quality improvement costs latency and is an explicit product
tradeoff, not a free optimization.

Run with:

```bash
uv run enclave-eval-context \
  --output benchmarks/raw/postgres-conversations.json
```
