"""
Text chunking utilities for RAG.

The retriever can't embed a whole document as one vector and expect good
matches -- long documents need to be split into smaller overlapping chunks
first. Overlap matters because it stops us from cutting a sentence in half
right where the answer lives.
"""
from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 50, overlap: int = 10) -> list[str]:
    """Split `text` into word chunks of `chunk_size` words, with `overlap`
    words repeated between consecutive chunks.

    Args:
        text: the raw document text.
        chunk_size: number of words per chunk.
        overlap: number of words shared between consecutive chunks.

    Returns:
        A list of chunk strings. Empty list if text is empty/whitespace.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks
