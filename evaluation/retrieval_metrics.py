"""
Retrieval-quality metrics for RAG evaluation.

All metrics operate on the list of retrieved chunks (in rank order) plus
an :class:`~evaluation.dataset.EvalExample` that defines what counts as a
relevant chunk via ``expected_sources`` and/or ``expected_substrings``.

Implemented metrics
-------------------
* ``hit_at_k``     — 1.0 iff any of the top-K chunks is relevant, else 0.0
* ``recall_at_k``  — |relevant ∩ top-K| / total expected relevance signals
* ``precision_at_k`` — |relevant ∩ top-K| / K
* ``mrr``          — mean reciprocal rank of the first relevant chunk
* ``ndcg_at_k``    — normalised discounted cumulative gain (binary gain)

All functions are pure, side-effect free, and depend only on NumPy.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List

from evaluation.dataset import EvalExample


def _relevance_vector(example: EvalExample, chunks: List[Dict[str, Any]]) -> List[int]:
    """Return a 0/1 list of length ``len(chunks)`` marking relevance."""
    return [1 if example.is_relevant(c) else 0 for c in chunks]


def hit_at_k(example: EvalExample, chunks: List[Dict[str, Any]], k: int) -> float:
    """1.0 iff at least one of the top-K retrieved chunks is relevant."""
    if k <= 0 or not chunks:
        return 0.0
    rel = _relevance_vector(example, chunks[:k])
    return 1.0 if any(rel) else 0.0


def recall_at_k(example: EvalExample, chunks: List[Dict[str, Any]], k: int) -> float:
    """
    Fraction of expected relevance signals recovered in the top-K.

    The denominator is :meth:`EvalExample.total_relevant` — typically the
    number of distinct expected source files. We count *unique* sources
    hit in the top-K to avoid inflating recall when several chunks share
    a single relevant source.
    """
    if k <= 0 or not chunks:
        return 0.0
    denom = example.total_relevant()
    if denom <= 0:
        return 0.0
    seen: set = set()
    for c in chunks[:k]:
        if not example.is_relevant(c):
            continue
        src = (c.get("metadata", {}) or {}).get("source", "")
        # Fall back to content hash when source missing (substring-only datasets).
        key = src or hash((c.get("content") or "")[:200])
        seen.add(key)
    return min(1.0, len(seen) / denom)


def precision_at_k(example: EvalExample, chunks: List[Dict[str, Any]], k: int) -> float:
    """Fraction of the top-K chunks that are relevant."""
    if k <= 0 or not chunks:
        return 0.0
    top = chunks[:k]
    rel = _relevance_vector(example, top)
    return sum(rel) / float(len(top))


def mrr(example: EvalExample, chunks: List[Dict[str, Any]]) -> float:
    """Reciprocal rank of the first relevant chunk (1-indexed). 0 if none."""
    for rank, c in enumerate(chunks, 1):
        if example.is_relevant(c):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(example: EvalExample, chunks: List[Dict[str, Any]], k: int) -> float:
    """
    Normalised DCG @K with binary gain (1 for relevant, 0 for not).

    DCG = Σ_{i=1..K} rel_i / log2(i + 1)
    IDCG = DCG of the ideal ranking (all relevant first).
    """
    if k <= 0 or not chunks:
        return 0.0
    rel = _relevance_vector(example, chunks[:k])
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    n_rel = min(sum(rel), k)
    if n_rel == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
    return dcg / idcg if idcg > 0 else 0.0


def compute_retrieval_metrics(
    example: EvalExample,
    chunks: List[Dict[str, Any]],
    ks: Iterable[int] = (1, 3, 5, 10),
) -> Dict[str, float]:
    """
    Compute the full retrieval-metrics suite for one (example, retrieved) pair.

    Returns a flat dict with keys like ``hit@1``, ``recall@5``,
    ``precision@10``, ``mrr``, ``ndcg@5``.
    """
    metrics: Dict[str, float] = {"mrr": mrr(example, chunks)}
    for k in ks:
        metrics[f"hit@{k}"] = hit_at_k(example, chunks, k)
        metrics[f"recall@{k}"] = recall_at_k(example, chunks, k)
        metrics[f"precision@{k}"] = precision_at_k(example, chunks, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(example, chunks, k)
    return metrics


def aggregate_metrics(per_example: List[Dict[str, float]]) -> Dict[str, float]:
    """Mean-aggregate a list of per-example metric dicts."""
    if not per_example:
        return {}
    keys = set()
    for d in per_example:
        keys.update(d.keys())
    agg: Dict[str, float] = {}
    for k in sorted(keys):
        vals = [d.get(k, 0.0) for d in per_example]
        agg[k] = sum(vals) / len(vals)
    return agg
