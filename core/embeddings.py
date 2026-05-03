"""
Embedding generation and management.
Uses sentence-transformers for text embeddings with batch processing.
"""
import asyncio
import numpy as np
from typing import List, Optional
from functools import lru_cache

import streamlit as st

from config import EMBEDDING_MODEL, EMBEDDING_DIM, EMBEDDING_BATCH_SIZE


@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str = EMBEDDING_MODEL):
    """
    Load and cache the sentence-transformer embedding model.
    Uses st.cache_resource to persist across reruns.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model


def generate_embeddings(
    texts: List[str],
    model_name: str = EMBEDDING_MODEL,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Generate embeddings for a list of texts using batch processing.
    
    Returns:
        numpy array of shape (len(texts), embedding_dim)
    """
    if not texts:
        return np.array([])

    model = load_embedding_model(model_name)
    
    all_embeddings: List[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(
            batch,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        all_embeddings.append(batch_embeddings)

    return np.vstack(all_embeddings)


def generate_query_embedding(
    query: str,
    model_name: str = EMBEDDING_MODEL,
) -> np.ndarray:
    """Generate embedding for a single query."""
    model = load_embedding_model(model_name)
    return model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]


def compute_similarity(
    query_embedding: np.ndarray,
    document_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between query and document embeddings."""
    # Since embeddings are normalized, cosine similarity = dot product
    return np.dot(document_embeddings, query_embedding)
