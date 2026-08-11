"""
The "agentic" part of agentic RAG.

A plain RAG pipeline does one retrieval pass and hands the top chunk to an
LLM. An *agentic* pipeline adds a decision loop: retrieve, judge whether the
result is good enough, and if not, reformulate the query and try again --
up to a max number of iterations.

We keep the "judge" and "reformulate" steps rule-based (no LLM call) so this
stays free, fast, and deterministic in CI. In a real system you'd swap
`_is_good_enough` and `_reformulate` for LLM calls without touching the loop
structure.
"""
from __future__ import annotations

from collections.abc import Callable

from app.retrieval import retrieve

CONFIDENCE_THRESHOLD = 0.2


def _reformulate(query: str) -> str:
    """Naive reformulation: drop the leading word (often a low-signal
    word like "what"/"how") and try again."""
    words = query.split()
    return " ".join(words[1:]) if len(words) > 1 else query


def agentic_answer(
    query: str,
    documents: list[str],
    embed_fn: Callable,
    max_iterations: int = 2,
) -> dict:
    """Run the retrieve -> judge -> reformulate loop.

    Returns a dict with the best answer found, how many iterations it took,
    and the confidence score, so the caller (and tests) can inspect the
    agent's behavior, not just its final answer.
    """
    current_query = query
    best_results: list[tuple[str, float]] = []

    for iteration in range(1, max_iterations + 1):
        results = retrieve(current_query, documents, embed_fn, top_k=3)
        if results:
            best_results = results
        if results and results[0][1] >= CONFIDENCE_THRESHOLD:
            return {
                "answer": results[0][0],
                "score": results[0][1],
                "iterations": iteration,
                "final_query": current_query,
            }
        current_query = _reformulate(current_query)

    # Ran out of iterations -- return the best we saw, flagged as low confidence.
    if best_results:
        return {
            "answer": best_results[0][0],
            "score": best_results[0][1],
            "iterations": max_iterations,
            "final_query": current_query,
        }
    return {"answer": None, "score": 0.0, "iterations": max_iterations, "final_query": current_query}
