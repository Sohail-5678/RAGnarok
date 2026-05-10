"""Unit tests for the ``evaluation`` package (retrieval metrics, dataset,
generation metrics that don't need network, logger, harness wiring)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.dataset import EvalExample, load_dataset, save_dataset
from evaluation.retrieval_metrics import (
    hit_at_k, recall_at_k, precision_at_k, mrr, ndcg_at_k,
    compute_retrieval_metrics, aggregate_metrics,
)
from evaluation.generation_metrics import (
    bleu_score, rouge_scores, citation_coverage,
)
from evaluation.logger import QueryLogEntry
from evaluation.harness import EvalConfig


# ─── Dataset round-trip ─────────────────────────────────────────────────

class TestDataset:
    def test_load_and_save_round_trip(self, tmp_path: Path):
        path = tmp_path / "ds.jsonl"
        examples = [
            EvalExample(id="q1", query="what is X?", expected_sources=["a.pdf"]),
            EvalExample(id="q2", query="what is Y?", expected_substrings=["foo"]),
        ]
        n = save_dataset(examples, path)
        assert n == 2
        loaded = load_dataset(path)
        assert len(loaded) == 2
        assert loaded[0].id == "q1"
        assert loaded[0].expected_sources == ["a.pdf"]

    def test_load_skips_blank_and_comment_lines(self, tmp_path: Path):
        path = tmp_path / "ds.jsonl"
        path.write_text(
            '\n'
            '# header comment\n'
            '{"id":"q1","query":"x"}\n'
            '\n',
            encoding="utf-8",
        )
        loaded = load_dataset(path)
        assert len(loaded) == 1
        assert loaded[0].id == "q1"

    def test_load_raises_on_missing_query(self, tmp_path: Path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"id":"q1"}\n', encoding="utf-8")
        with pytest.raises(ValueError):
            load_dataset(path)

    def test_is_relevant_source_match(self):
        ex = EvalExample(id="q", query="?", expected_sources=["doc.pdf"])
        assert ex.is_relevant({"metadata": {"source": "doc.pdf"}}) is True
        assert ex.is_relevant({"metadata": {"source": "other.pdf"}}) is False

    def test_is_relevant_substring_match_case_insensitive(self):
        ex = EvalExample(id="q", query="?", expected_substrings=["RAG", "fusion"])
        assert ex.is_relevant({"content": "hybrid Rag systems"}) is True
        assert ex.is_relevant({"content": "completely unrelated"}) is False


# ─── Retrieval metrics ──────────────────────────────────────────────────

class TestRetrievalMetrics:
    @pytest.fixture
    def example_md(self):
        return EvalExample(
            id="q", query="?",
            expected_sources=["readme.md"],
        )

    def test_hit_at_k(self, example_md, sample_chunks):
        assert hit_at_k(example_md, sample_chunks, k=1) == 1.0
        assert hit_at_k(example_md, [sample_chunks[3]], k=1) == 0.0
        assert hit_at_k(example_md, [], k=5) == 0.0

    def test_precision_at_k(self, example_md, sample_chunks):
        # Top 2 are both readme.md
        assert precision_at_k(example_md, sample_chunks, k=2) == 1.0
        # Top 4 → 2 relevant out of 4
        assert precision_at_k(example_md, sample_chunks, k=4) == pytest.approx(0.5)

    def test_recall_at_k_unique_sources(self, example_md, sample_chunks):
        # Only 1 unique expected source. Once we hit it, recall = 1.0.
        assert recall_at_k(example_md, sample_chunks, k=1) == 1.0
        assert recall_at_k(example_md, sample_chunks[3:], k=1) == 0.0

    def test_mrr_first_rank(self, example_md, sample_chunks):
        assert mrr(example_md, sample_chunks) == 1.0
        # Reverse the list → relevant doc is at rank 3.
        reordered = list(reversed(sample_chunks))
        assert mrr(example_md, reordered) == pytest.approx(1 / 3)

    def test_mrr_zero_when_none_relevant(self, example_md):
        chunks = [{"metadata": {"source": "no.pdf"}, "content": "x"}]
        assert mrr(example_md, chunks) == 0.0

    def test_ndcg_at_k(self, example_md, sample_chunks):
        v = ndcg_at_k(example_md, sample_chunks, k=4)
        assert 0.0 < v <= 1.0
        # Perfect ranking should give nDCG ≈ 1.0
        perfect = [
            c for c in sample_chunks if c["metadata"]["source"] == "readme.md"
        ] + [
            c for c in sample_chunks if c["metadata"]["source"] != "readme.md"
        ]
        assert ndcg_at_k(example_md, perfect, k=4) >= v

    def test_compute_retrieval_metrics_keys(self, example_md, sample_chunks):
        m = compute_retrieval_metrics(example_md, sample_chunks, ks=(1, 3))
        assert {"hit@1", "recall@1", "precision@1", "ndcg@1",
                "hit@3", "recall@3", "precision@3", "ndcg@3", "mrr"} <= set(m.keys())

    def test_aggregate_metrics_mean(self):
        a = {"hit@1": 1.0, "mrr": 1.0}
        b = {"hit@1": 0.0, "mrr": 0.5}
        out = aggregate_metrics([a, b])
        assert out["hit@1"] == 0.5
        assert out["mrr"] == 0.75


# ─── Generation metrics (no network) ────────────────────────────────────

class TestGenerationMetrics:
    def test_bleu_identical_is_high(self):
        s = "this is a perfectly matching sentence"
        assert bleu_score(s, s) > 0.5

    def test_bleu_empty_inputs(self):
        assert bleu_score("", "ref") == 0.0
        assert bleu_score("hyp", "") == 0.0

    def test_rouge_scores_keys(self):
        out = rouge_scores("hello world foo", "hello world bar")
        assert {"rouge1", "rouge2", "rougeL"} <= set(out.keys())
        assert 0 <= out["rouge1"] <= 1

    def test_citation_coverage_matches_sources(self, sample_chunks):
        answer = (
            "The system supports multimodal RAG [Source: readme.md, page 1]. "
            "Hybrid search is described [Source: architecture.pdf, page 3]."
        )
        out = citation_coverage(answer, sample_chunks)
        assert out["num_citations"] >= 2
        assert out["coverage"] > 0.0
        assert out["fake_citation_rate"] == 0.0

    def test_citation_coverage_detects_fakes(self, sample_chunks):
        answer = "Per [Source: hallucinated.pdf, page 99] this is wrong."
        out = citation_coverage(answer, sample_chunks)
        assert out["fake_citation_rate"] > 0.0

    def test_citation_coverage_handles_empty_answer(self):
        out = citation_coverage("", [])
        assert out["num_citations"] == 0


# ─── Logger ─────────────────────────────────────────────────────────────

class TestQueryLogger:
    def test_log_and_fetch(self, tmp_logger):
        entry = QueryLogEntry(
            query="what is RAG?",
            answer="Retrieval-augmented generation.",
            retrieved=[{"content": "ctx", "metadata": {"source": "a"}}],
            sources=[{"source": "a", "modality": "text"}],
            provider="groq",
            config={"top_k_rerank": 5},
            latency_ms=123.4,
            session_id="abc",
        )
        tmp_logger.log(entry)
        assert tmp_logger.count() == 1
        rows = tmp_logger.fetch_recent(limit=5)
        assert len(rows) == 1
        assert rows[0]["query"] == "what is RAG?"
        assert rows[0]["provider"] == "groq"

    def test_jsonl_written(self, tmp_logger, tmp_path):
        tmp_logger.log(QueryLogEntry(query="hello"))
        text = tmp_logger.jsonl_path.read_text(encoding="utf-8").strip()
        assert text, "JSONL file should not be empty"
        row = json.loads(text.splitlines()[-1])
        assert row["query"] == "hello"


# ─── Harness config ─────────────────────────────────────────────────────

class TestEvalConfig:
    def test_defaults_match_expectations(self):
        cfg = EvalConfig()
        assert cfg.name == "default"
        assert cfg.enable_reranking is True
        assert cfg.dense_weight + cfg.bm25_weight == pytest.approx(1.0)

    def test_to_dict_round_trip(self):
        cfg = EvalConfig(name="x", enable_reranking=False)
        d = cfg.to_dict()
        assert d["name"] == "x"
        assert d["enable_reranking"] is False
        assert isinstance(d["ks"], list)
