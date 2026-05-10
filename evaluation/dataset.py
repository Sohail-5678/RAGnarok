"""
Gold-set dataset loading for RAG evaluation.

Each example is one row in a JSONL file with the following schema::

    {
      "id": "q001",                                 # unique id (string)
      "query": "What did the speaker say about X?", # required
      "expected_sources": ["report.pdf"],           # optional: list[str]
      "expected_substrings": ["greenhouse gas"],    # optional: list[str]
      "gold_answer": "The speaker said...",         # optional: reference answer
      "modality_filter": null                       # optional: "text"|"audio"|"video"|"image"|null
    }

``expected_sources`` and ``expected_substrings`` define what counts as a
"relevant" retrieved chunk:

  * Source match — chunk's ``metadata.source`` is in ``expected_sources``.
  * Substring match — chunk's content contains any of ``expected_substrings``
    (case-insensitive).

A chunk is considered relevant if **either** check passes. At least one
of the two fields must be provided for retrieval metrics to be meaningful;
otherwise only the LLM-as-judge generation metrics are computable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class EvalExample:
    """One evaluation example loaded from a JSONL gold set."""
    id: str
    query: str
    expected_sources: List[str] = field(default_factory=list)
    expected_substrings: List[str] = field(default_factory=list)
    gold_answer: Optional[str] = None
    modality_filter: Optional[str] = None

    def is_relevant(self, chunk: Dict[str, Any]) -> bool:
        """Return True if ``chunk`` should be considered relevant for this example."""
        meta = chunk.get("metadata", {}) or {}
        source = (meta.get("source") or "").strip()
        if source and source in self.expected_sources:
            return True
        content = (chunk.get("content") or "").lower()
        for needle in self.expected_substrings:
            if needle and needle.lower() in content:
                return True
        return False

    def total_relevant(self) -> int:
        """Best-effort denominator for Recall@K — number of distinct relevance signals."""
        # When ``expected_sources`` is the primary signal, the denominator is the
        # number of distinct sources we expect to surface. Substring expectations
        # add to the count when no sources are listed.
        if self.expected_sources:
            return len(set(self.expected_sources))
        return max(1, len(self.expected_substrings))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_dataset(path: str | Path) -> List[EvalExample]:
    """Load a JSONL gold set into a list of :class:`EvalExample`."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")

    examples: List[EvalExample] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_no} of {p}: {e}") from e
            if "query" not in row:
                raise ValueError(f"Missing 'query' field at line {line_no} of {p}")
            examples.append(EvalExample(
                id=str(row.get("id", f"q{line_no:04d}")),
                query=row["query"],
                expected_sources=list(row.get("expected_sources", []) or []),
                expected_substrings=list(row.get("expected_substrings", []) or []),
                gold_answer=row.get("gold_answer"),
                modality_filter=row.get("modality_filter"),
            ))
    return examples


def save_dataset(examples: Iterable[EvalExample], path: str | Path) -> int:
    """Serialise examples back to JSONL. Returns the number of rows written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with p.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count
