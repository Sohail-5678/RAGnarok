"""
Persistent query logger for RAGnarok.

Captures every ``(query, retrieved_chunks, answer, sources, config,
latency)`` tuple to:

  * **SQLite** at ``evaluation/logs/ragnarok_queries.db``
        — structured, indexable, easy to query offline.
  * **JSONL** at ``evaluation/logs/ragnarok_queries.jsonl``
        — append-only, human-readable, easy to grep / pipe.

Both writes are best-effort: any I/O error is swallowed so logging
never breaks the live chat experience.

The logger is *opt-in* — the chat layer only writes when
``st.session_state.persist_queries`` is True (toggle exposed in the
sidebar's "Evaluation & Logging" expander).
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


_DEFAULT_LOG_DIR = Path(__file__).parent / "logs"
_DEFAULT_DB_PATH = _DEFAULT_LOG_DIR / "ragnarok_queries.db"
_DEFAULT_JSONL_PATH = _DEFAULT_LOG_DIR / "ragnarok_queries.jsonl"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS query_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL,
    session_id      TEXT,
    query           TEXT    NOT NULL,
    answer          TEXT,
    retrieved       TEXT,           -- JSON array
    sources         TEXT,           -- JSON array
    provider        TEXT,
    config          TEXT,           -- JSON object
    latency_ms      REAL,
    error           TEXT
);
CREATE INDEX IF NOT EXISTS idx_query_logs_ts ON query_logs(ts);
CREATE INDEX IF NOT EXISTS idx_query_logs_session ON query_logs(session_id);
"""


@dataclass
class QueryLogEntry:
    """One logged query → answer cycle."""
    query: str
    answer: str = ""
    retrieved: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    provider: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: Optional[str] = None
    session_id: str = ""
    ts: str = ""

    def to_row(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "session_id": self.session_id,
            "query": self.query,
            "answer": self.answer,
            "retrieved": json.dumps(self.retrieved, ensure_ascii=False, default=str),
            "sources": json.dumps(self.sources, ensure_ascii=False, default=str),
            "provider": self.provider,
            "config": json.dumps(self.config, ensure_ascii=False, default=str),
            "latency_ms": self.latency_ms,
            "error": self.error,
        }

    def to_jsonl(self) -> str:
        return json.dumps({
            "ts": self.ts,
            "session_id": self.session_id,
            "query": self.query,
            "answer": self.answer,
            "retrieved": self.retrieved,
            "sources": self.sources,
            "provider": self.provider,
            "config": self.config,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }, ensure_ascii=False, default=str)


class QueryLogger:
    """
    Thread-safe persistent logger writing to SQLite + JSONL.

    Use :func:`get_default_logger` to obtain a process-wide singleton.
    """

    def __init__(
        self,
        db_path: Path = _DEFAULT_DB_PATH,
        jsonl_path: Path = _DEFAULT_JSONL_PATH,
    ):
        self.db_path = Path(db_path)
        self.jsonl_path = Path(jsonl_path)
        self._lock = threading.Lock()
        self._ensure_storage()

    def _ensure_storage(self) -> None:
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                conn.executescript(_SCHEMA)
                conn.commit()
        except Exception:
            # Best-effort; don't crash if filesystem is read-only.
            pass

    def log(self, entry: QueryLogEntry) -> None:
        """Persist one entry to both stores. Never raises."""
        if not entry.ts:
            entry.ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        with self._lock:
            self._write_sqlite(entry)
            self._write_jsonl(entry)

    def _write_sqlite(self, entry: QueryLogEntry) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT INTO query_logs
                       (ts, session_id, query, answer, retrieved, sources,
                        provider, config, latency_ms, error)
                       VALUES (:ts, :session_id, :query, :answer, :retrieved,
                               :sources, :provider, :config, :latency_ms, :error)""",
                    entry.to_row(),
                )
                conn.commit()
        except Exception:
            pass

    def _write_jsonl(self, entry: QueryLogEntry) -> None:
        try:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self.jsonl_path.open("a", encoding="utf-8") as f:
                f.write(entry.to_jsonl() + "\n")
        except Exception:
            pass

    # ─── Read helpers (used by harness & UI) ───────────────────────────

    def fetch_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return the most recent N entries as plain dicts."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM query_logs ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def count(self) -> int:
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute("SELECT COUNT(*) FROM query_logs").fetchone()
            return int(row[0]) if row else 0
        except Exception:
            return 0


_default_logger: Optional[QueryLogger] = None
_default_lock = threading.Lock()


def get_default_logger() -> QueryLogger:
    """Process-wide singleton logger (lazy)."""
    global _default_logger
    with _default_lock:
        if _default_logger is None:
            _default_logger = QueryLogger()
        return _default_logger


def new_session_id() -> str:
    """Generate a short opaque session id."""
    return uuid.uuid4().hex[:12]
