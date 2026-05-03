"""
Vector store management using ChromaDB.
Handles storage, retrieval, and metadata filtering.
"""
import os
import uuid
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import chromadb
from chromadb.config import Settings as ChromaSettings

from config import EMBEDDING_DIM
from utils.chunking import Chunk


CHROMA_COLLECTION_NAME = "multimodal_rag"


@st.cache_resource(show_spinner=False)
def get_chroma_client():
    """Initialize and cache ChromaDB client."""
    db_path = os.path.join(os.getcwd(), "chroma_db")
    client = chromadb.PersistentClient(path=db_path, settings=ChromaSettings(
        anonymized_telemetry=False,
        allow_reset=True,
    ))
    return client


def get_or_create_collection(
    client: chromadb.Client = None,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> chromadb.Collection:
    """Get or create a ChromaDB collection."""
    if client is None:
        client = get_chroma_client()
    
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def add_chunks_to_store(
    chunks: List[Chunk],
    embeddings: np.ndarray,
    collection: Optional[chromadb.Collection] = None,
) -> int:
    """
    Add chunks with their embeddings to the vector store.
    
    Returns the number of chunks added.
    """
    if not chunks or len(embeddings) == 0:
        return 0

    if collection is None:
        collection = get_or_create_collection()

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    embedding_list: List[List[float]] = []

    for i, chunk in enumerate(chunks):
        chunk_uuid = f"{chunk.chunk_id}_{uuid.uuid4().hex[:8]}"
        ids.append(chunk_uuid)
        documents.append(chunk.content)
        
        # Ensure metadata values are ChromaDB-compatible (str, int, float, bool)
        clean_meta = {}
        for k, v in chunk.metadata.items():
            if isinstance(v, (str, int, float, bool)):
                clean_meta[k] = v
            else:
                clean_meta[k] = str(v)
        metadatas.append(clean_meta)
        embedding_list.append(embeddings[i].tolist())

    # Add in batches (ChromaDB has batch limits)
    batch_size = 100
    added = 0

    for j in range(0, len(ids), batch_size):
        end = min(j + batch_size, len(ids))
        collection.add(
            ids=ids[j:end],
            documents=documents[j:end],
            metadatas=metadatas[j:end],
            embeddings=embedding_list[j:end],
        )
        added += end - j

    return added


def query_vector_store(
    query_embedding: np.ndarray,
    collection: Optional[chromadb.Collection] = None,
    top_k: int = 20,
    where_filter: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Query the vector store for similar chunks.
    
    Returns list of results with content, metadata, and similarity score.
    """
    if collection is None:
        collection = get_or_create_collection()

    if collection.count() == 0:
        return []

    query_params = {
        "query_embeddings": [query_embedding.tolist()],
        "n_results": min(top_k, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }

    if where_filter:
        query_params["where"] = where_filter

    try:
        results = collection.query(**query_params)
    except Exception:
        # Fallback without filter
        query_params.pop("where", None)
        results = collection.query(**query_params)

    formatted_results: List[Dict[str, Any]] = []

    if results and results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            distance = results["distances"][0][i] if results["distances"] else 0.0
            # ChromaDB returns distance, convert to similarity (cosine)
            similarity = 1.0 - distance

            formatted_results.append({
                "content": doc,
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "similarity": similarity,
                "id": results["ids"][0][i] if results["ids"] else "",
            })

    return formatted_results


def get_all_documents(
    collection: Optional[chromadb.Collection] = None,
) -> List[Dict[str, Any]]:
    """Get all documents from the collection."""
    if collection is None:
        collection = get_or_create_collection()

    if collection.count() == 0:
        return []

    results = collection.get(include=["documents", "metadatas"])
    
    documents: List[Dict[str, Any]] = []
    for i, doc in enumerate(results["documents"]):
        documents.append({
            "content": doc,
            "metadata": results["metadatas"][i] if results["metadatas"] else {},
            "id": results["ids"][i],
        })

    return documents


def clear_collection(
    collection_name: str = CHROMA_COLLECTION_NAME,
):
    """Delete and recreate the collection."""
    client = get_chroma_client()
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    return get_or_create_collection(client, collection_name)


def get_collection_stats(
    collection: Optional[chromadb.Collection] = None,
) -> Dict[str, Any]:
    """Get statistics about the current collection."""
    if collection is None:
        collection = get_or_create_collection()

    count = collection.count()
    
    stats = {
        "total_chunks": count,
        "sources": set(),
        "modalities": set(),
    }

    if count > 0:
        all_docs = collection.get(include=["metadatas"])
        for meta in all_docs["metadatas"]:
            if "source" in meta:
                stats["sources"].add(meta["source"])
            if "modality" in meta:
                stats["modalities"].add(meta["modality"])

    stats["sources"] = list(stats["sources"])
    stats["modalities"] = list(stats["modalities"])
    stats["num_sources"] = len(stats["sources"])

    return stats
