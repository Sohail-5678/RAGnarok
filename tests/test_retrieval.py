"""Unit tests for ``core.retrieval`` — RRF math and BM25 plumbing."""
from __future__ import annotations

import pytest

from core.retrieval import reciprocal_rank_fusion, BM25Retriever


class TestReciprocalRankFusion:
    def test_empty_inputs_returns_empty(self):
        assert reciprocal_rank_fusion([], [], top_k=5) == []

    def test_dense_only_preserves_order(self):
        dense = [
            {"id": "a", "content": "A", "metadata": {}, "similarity": 0.9},
            {"id": "b", "content": "B", "metadata": {}, "similarity": 0.8},
            {"id": "c", "content": "C", "metadata": {}, "similarity": 0.7},
        ]
        out = reciprocal_rank_fusion(dense, [], dense_weight=1.0, sparse_weight=0.0,
                                     top_k=10)
        assert [r["id"] for r in out] == ["a", "b", "c"]
        # Fusion scores must be strictly decreasing.
        scores = [r["fusion_score"] for r in out]
        assert scores == sorted(scores, reverse=True)

    def test_sparse_only_preserves_order(self):
        sparse = [
            {"id": "x", "content": "X", "metadata": {}, "bm25_score": 5.0},
            {"id": "y", "content": "Y", "metadata": {}, "bm25_score": 3.0},
        ]
        out = reciprocal_rank_fusion([], sparse, dense_weight=0.0, sparse_weight=1.0,
                                     top_k=10)
        assert [r["id"] for r in out] == ["x", "y"]

    def test_fusion_combines_dense_and_sparse(self):
        dense = [
            {"id": "shared", "content": "S", "metadata": {}, "similarity": 0.9},
            {"id": "dense_only", "content": "D", "metadata": {}, "similarity": 0.5},
        ]
        sparse = [
            {"id": "shared", "content": "S", "metadata": {}, "bm25_score": 4.0},
            {"id": "sparse_only", "content": "P", "metadata": {}, "bm25_score": 2.0},
        ]
        out = reciprocal_rank_fusion(dense, sparse, dense_weight=0.5,
                                     sparse_weight=0.5, top_k=10)
        ids = [r["id"] for r in out]
        # Shared doc appears once and should rank first (boosted by both).
        assert ids.count("shared") == 1
        assert ids[0] == "shared"
        # Both unique docs are still represented.
        assert set(ids) == {"shared", "dense_only", "sparse_only"}

    def test_top_k_caps_output(self):
        dense = [
            {"id": str(i), "content": str(i), "metadata": {}, "similarity": 1.0 - i * 0.1}
            for i in range(10)
        ]
        out = reciprocal_rank_fusion(dense, [], top_k=3)
        assert len(out) == 3


class TestBM25Retriever:
    def test_empty_index_returns_empty(self):
        r = BM25Retriever()
        assert r.search("anything") == []

    def test_build_and_search(self):
        r = BM25Retriever()
        docs = [
            {"id": "1", "content": "Multimodal RAG with hybrid search and reranking",
             "metadata": {"source": "a"}},
            {"id": "2", "content": "Cooking pasta with tomato sauce",
             "metadata": {"source": "b"}},
            {"id": "3", "content": "Reciprocal rank fusion combines dense and sparse",
             "metadata": {"source": "c"}},
        ]
        r.build_index(docs)
        results = r.search("hybrid reranking", top_k=3)
        assert results, "should find at least one result"
        # Doc 1 mentions both 'hybrid' and 'reranking' → must rank first.
        assert results[0]["metadata"]["source"] == "a"
        # Scores must be non-negative.
        assert all(res["bm25_score"] >= 0 for res in results)

    def test_search_with_unrelated_query_can_be_empty(self):
        r = BM25Retriever()
        r.build_index([{"id": "1", "content": "alpha beta", "metadata": {}}])
        # rank-bm25 returns 0 for completely unrelated queries; results may be
        # filtered out because scores are not > 0.
        results = r.search("xxxxxx yyyyyy")
        for res in results:
            assert res["bm25_score"] > 0
