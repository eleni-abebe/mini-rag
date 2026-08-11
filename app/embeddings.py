"""
A tiny, dependency-free "embedding" model.

Real projects call OpenAI/Cohere/local sentence-transformers here. We use a
deterministic hashing-trick bag-of-words vectorizer instead so that:
  1. CI never needs an API key or network access.
  2. Tests are 100% reproducible.
  3. You can see exactly why two texts are "similar" -- shared words hash
     into the same bucket.

Swap `embed_text` for a real embedding call later; nothing else in the
pipeline needs to change, which is the whole point of separating this out.
"""
from __future__ import annotations

import hashlib

import numpy as np

DEFAULT_DIM = 128


def embed_text(text: str, dim: int = DEFAULT_DIM) -> np.ndarray:
    """Turn text into a fixed-length, L2-normalized vector.

    Each word is hashed into one of `dim` buckets (the "hashing trick"),
    so the same word always lands in the same bucket -> deterministic.
    """
    vec = np.zeros(dim, dtype=np.float64)
    words = text.lower().split()
    for word in words:
        idx = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % dim
        vec[idx] += 1.0

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec
