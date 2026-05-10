"""
Pytest configuration and shared fixtures for RAGnarok.

Adds the project root to ``sys.path`` so test files can do
``from core.retrieval import ...`` etc. without needing an installed
package, and provides reusable sample chunks + a temporary logger.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest


# Make the project root importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@pytest.fixture
def sample_chunks() -> List[Dict[str, Any]]:
    """A small canned set of retrieval results for metric tests."""
    return [
        {
            "content": "RAGnarok supports multimodal RAG with PDF, audio, video, and images.",
            "metadata": {"source": "readme.md", "modality": "text"},
            "rerank_score": 0.95,
            "fusion_score": 0.81,
        },
        {
            "content": "It uses all-MiniLM-L6-v2 for embeddings and cross-encoder reranking.",
            "metadata": {"source": "readme.md", "modality": "text"},
            "rerank_score": 0.88,
        },
        {
            "content": "Hybrid search fuses dense vector search with BM25 via reciprocal rank fusion.",
            "metadata": {"source": "architecture.pdf", "modality": "text", "page_number": 3},
            "rerank_score": 0.71,
        },
        {
            "content": "An unrelated paragraph about cooking pasta.",
            "metadata": {"source": "cooking.md", "modality": "text"},
            "rerank_score": 0.12,
        },
    ]


@pytest.fixture
def tmp_logger(tmp_path):
    """A QueryLogger writing to a per-test temp directory."""
    from evaluation.logger import QueryLogger
    db = tmp_path / "queries.db"
    jsonl = tmp_path / "queries.jsonl"
    return QueryLogger(db_path=db, jsonl_path=jsonl)
