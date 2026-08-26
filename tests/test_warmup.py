from enclave.models.warmup import warm_local_models


class FakeEmbedder:
    def __init__(self, calls):
        self.calls = calls

    def encode_queries(self, texts):
        self.calls.append(("embed", texts))


class FakeReranker:
    def __init__(self, calls):
        self.calls = calls

    def score(self, query, documents, batch_size=8):
        self.calls.append(("rerank", query, documents, batch_size))
        return [1.0]


def test_warmup_exercises_all_three_model_paths():
    calls = []

    result = warm_local_models(
        embedder_loader=lambda: FakeEmbedder(calls),
        reranker_loader=lambda: FakeReranker(calls),
        ollama_loader=lambda: calls.append(("ollama",)),
    )

    assert [call[0] for call in calls] == ["embed", "rerank", "ollama"]
    assert result.embedding_ms >= 0
    assert result.reranker_ms >= 0
    assert result.ollama_ms >= 0
    assert result.total_ms >= 0
