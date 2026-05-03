"""
Smart retrieval layer: Hybrid Search (Dense + BM25 Sparse) with Cross-Encoder Re-ranking.
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st

from config import (
    TOP_K_RETRIEVAL, TOP_K_RERANK,
    BM25_WEIGHT, DENSE_WEIGHT, SIMILARITY_THRESHOLD,
    RERANKER_MODEL,
)
from core.embeddings import generate_query_embedding
from core.vector_store import query_vector_store, get_all_documents, get_or_create_collection


# ─── BM25 Sparse Retrieval ────────────────────────────────────────────

class BM25Retriever:
    """BM25 sparse keyword retrieval."""

    def __init__(self):
        self.corpus: List[str] = []
        self.documents: List[Dict[str, Any]] = []
        self.bm25 = None

    def build_index(self, documents: List[Dict[str, Any]]):
        """Build BM25 index from documents."""
        from rank_bm25 import BM25Okapi
        
        self.documents = documents
        self.corpus = [doc.get("content", "") for doc in documents]
        
        # Tokenize
        tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        
        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
        """Search using BM25."""
        if not self.bm25 or not self.corpus:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "content": self.documents[idx].get("content", ""),
                    "metadata": self.documents[idx].get("metadata", {}),
                    "bm25_score": float(scores[idx]),
                    "id": self.documents[idx].get("id", ""),
                })

        return results


# ─── Cross-Encoder Re-ranking ─────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_reranker(model_name: str = RERANKER_MODEL):
    """Load and cache the cross-encoder reranker model."""
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(model_name)
    return model


def rerank_results(
    query: str,
    results: List[Dict[str, Any]],
    top_k: int = TOP_K_RERANK,
    model_name: str = RERANKER_MODEL,
) -> List[Dict[str, Any]]:
    """
    Re-rank results using a cross-encoder model.
    
    Takes the merged results from hybrid search and re-orders them
    by relevance to the query.
    """
    if not results:
        return []

    if len(results) <= 1:
        return results

    try:
        reranker = load_reranker(model_name)
        
        # Prepare query-document pairs
        pairs = [(query, r["content"]) for r in results]
        
        # Get re-ranking scores
        scores = reranker.predict(pairs)
        
        # Attach scores and sort
        for i, result in enumerate(results):
            result["rerank_score"] = float(scores[i])

        # Sort by reranker score (descending)
        results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        return results[:top_k]

    except Exception as e:
        # Fallback: return top results by original score
        return results[:top_k]


# ─── Hybrid Search (Dense + Sparse Fusion) ────────────────────────────

def hybrid_search(
    query: str,
    top_k_retrieval: int = TOP_K_RETRIEVAL,
    top_k_rerank: int = TOP_K_RERANK,
    dense_weight: float = DENSE_WEIGHT,
    bm25_weight: float = BM25_WEIGHT,
    modality_filter: Optional[str] = None,
    enable_reranking: bool = True,
) -> List[Dict[str, Any]]:
    """
    Perform hybrid search combining dense vector search with BM25 sparse search.
    Then apply cross-encoder re-ranking for maximum precision.
    
    Pipeline:
    1. Dense search via ChromaDB (semantic similarity)
    2. Sparse search via BM25 (keyword matching)
    3. Reciprocal Rank Fusion to merge results
    4. Cross-encoder re-ranking for precision
    
    Returns:
        List of top-K re-ranked results with content, metadata, and scores.
    """
    collection = get_or_create_collection()
    
    if collection.count() == 0:
        return []

    # ── Step 1: Dense Vector Search ───────────────────────────────────
    query_embedding = generate_query_embedding(query)
    
    where_filter = None
    if modality_filter and modality_filter != "all":
        where_filter = {"modality": modality_filter}

    dense_results = query_vector_store(
        query_embedding=query_embedding,
        collection=collection,
        top_k=top_k_retrieval,
        where_filter=where_filter,
    )

    # ── Step 2: BM25 Sparse Search ────────────────────────────────────
    all_docs = get_all_documents(collection)
    
    # Apply modality filter to BM25 corpus too
    if modality_filter and modality_filter != "all":
        all_docs = [d for d in all_docs if d.get("metadata", {}).get("modality") == modality_filter]

    bm25_retriever = BM25Retriever()
    bm25_retriever.build_index(all_docs)
    bm25_results = bm25_retriever.search(query, top_k=top_k_retrieval)

    # ── Step 3: Reciprocal Rank Fusion ────────────────────────────────
    fused_results = reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=bm25_results,
        dense_weight=dense_weight,
        sparse_weight=bm25_weight,
        top_k=top_k_retrieval,
    )

    # ── Step 4: Cross-Encoder Re-ranking ──────────────────────────────
    if enable_reranking and fused_results:
        final_results = rerank_results(
            query=query,
            results=fused_results,
            top_k=top_k_rerank,
        )
    else:
        final_results = fused_results[:top_k_rerank]

    return final_results


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    dense_weight: float = DENSE_WEIGHT,
    sparse_weight: float = BM25_WEIGHT,
    k: int = 60,
    top_k: int = TOP_K_RETRIEVAL,
) -> List[Dict[str, Any]]:
    """
    Merge dense and sparse results using Reciprocal Rank Fusion (RRF).
    
    RRF Score = Σ (weight / (k + rank))
    """
    # Build a mapping from content hash to result + score
    fused_scores: Dict[str, Dict[str, Any]] = {}

    # Score dense results
    for rank, result in enumerate(dense_results, 1):
        key = result.get("id", result["content"][:100])
        rrf_score = dense_weight / (k + rank)
        
        if key not in fused_scores:
            fused_scores[key] = {
                "content": result["content"],
                "metadata": result["metadata"],
                "id": result.get("id", ""),
                "fusion_score": 0.0,
                "dense_similarity": result.get("similarity", 0.0),
                "dense_rank": rank,
            }
        fused_scores[key]["fusion_score"] += rrf_score

    # Score sparse results
    for rank, result in enumerate(sparse_results, 1):
        key = result.get("id", result["content"][:100])
        rrf_score = sparse_weight / (k + rank)

        if key not in fused_scores:
            fused_scores[key] = {
                "content": result["content"],
                "metadata": result["metadata"],
                "id": result.get("id", ""),
                "fusion_score": 0.0,
                "bm25_score": result.get("bm25_score", 0.0),
                "sparse_rank": rank,
            }
        fused_scores[key]["fusion_score"] += rrf_score
        fused_scores[key]["bm25_score"] = result.get("bm25_score", 0.0)
        fused_scores[key]["sparse_rank"] = rank

    # Sort by fusion score
    fused_list = sorted(
        fused_scores.values(),
        key=lambda x: x["fusion_score"],
        reverse=True,
    )

    return fused_list[:top_k]
