import numpy as np

from app.embeddings import embed_text


def test_embed_text_returns_normalized_vector_of_correct_dim():
    vec = embed_text("hello world", dim=64)
    assert vec.shape == (64,)
    assert np.isclose(np.linalg.norm(vec), 1.0)


def test_embed_text_deterministic_and_similar_text_scores_higher():
    v1 = embed_text("python is great for data science")
    v2 = embed_text("python is great for data science")
    v3 = embed_text("bananas taste good in smoothies")

    assert np.array_equal(v1, v2)  # deterministic

    sim_same = float(np.dot(v1, v2))
    sim_diff = float(np.dot(v1, v3))
    assert sim_same > sim_diff
