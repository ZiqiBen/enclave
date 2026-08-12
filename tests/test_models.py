"""Real local Qwen model contracts; never downloads weights."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.models


def test_real_embedder_shape_normalization_and_semantic_order():
    from enclave.models.encoders import get_embedder

    embedder = get_embedder()
    query = embedder.encode_queries(["Where must confidential documents stay?"])
    documents = embedder.encode_documents(
        [
            "Confidential documents must stay on local infrastructure.",
            "Tomatoes grow well in warm weather.",
        ]
    )

    assert query.shape == (1, 256)
    assert documents.shape == (2, 256)
    assert np.linalg.norm(query[0]) == pytest.approx(1.0, abs=1e-5)
    assert np.linalg.norm(documents[0]) == pytest.approx(1.0, abs=1e-5)
    similarities = query @ documents.T
    assert similarities[0, 0] > similarities[0, 1]


def test_real_reranker_prefers_relevant_passage():
    from enclave.models.encoders import get_reranker

    reranker = get_reranker()
    scores = reranker.score(
        "Where must confidential documents stay?",
        [
            "Confidential documents must stay on local infrastructure.",
            "Tomatoes grow well in warm weather.",
        ],
        batch_size=2,
    )

    assert len(scores) == 2
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert scores[0] > scores[1]
