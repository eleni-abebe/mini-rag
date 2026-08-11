"""
Similarity + ranking layer.

This is the "R" in RAG: given a query and a pool of chunks, return the
chunks most likely to help answer the query, best first.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1]. Returns 0.0 for a zero vector instead
    of raising, since an empty/stopword-only chunk shouldn't crash retrieval.
    """
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def retrieve(
    query: str,
    documents: list[str],
    embed_fn: Callable[[str], np.ndarray],
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """Rank `documents` by similarity to `query`, best match first.

    Returns a list of (document, score) tuples, length <= top_k.
    """
    q_vec = embed_fn(query)
    scored = [(doc, cosine_similarity(q_vec, embed_fn(doc))) for doc in documents]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]
