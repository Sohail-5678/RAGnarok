"""
End-to-end evaluation harness for RAGnarok.

Runs the full RAG pipeline (hybrid search + optional rerank + LLM
generation) on every example in a gold dataset, under one or more
configurations, and produces:

  * Per-example metrics (retrieval + generation) — written to JSONL.
  * Per-config aggregate report — written to JSON + Markdown.
  * Side-by-side A/B comparison across configs — written to Markdown.

Designed to be runnable both programmatically and from the CLI::

    python -m evaluation.cli \\
        --dataset docs/examples/sample_dataset.jsonl \\
        --output-dir evaluation/reports
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from evaluation.dataset import EvalExample, load_dataset
from evaluation.retrieval_metrics import (
    compute_retrieval_metrics, aggregate_metrics,
)
from evaluation.generation_metrics import compute_generation_metrics


# ─── Config ─────────────────────────────────────────────────────────────

@dataclass
class EvalConfig:
    """One pipeline configuration to evaluate."""
    name: str = "default"
    top_k_retrieval: int = 20
    top_k_rerank: int = 5
    dense_weight: float = 0.7
    bm25_weight: float = 0.3
    enable_reranking: bool = True
    modality_filter: Optional[str] = None        # overridden per-example if set on EvalExample
    provider: str = "groq"                       # LLM provider for generation
    run_llm_judge: bool = True
    run_bertscore: bool = False
    ks: Tuple[int, ...] = (1, 3, 5, 10)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ks"] = list(self.ks)
        return d


# ─── Single-example runner ──────────────────────────────────────────────

def _run_pipeline_once(
    example: EvalExample,
    cfg: EvalConfig,
    api_keys: Dict[str, str],
) -> Dict[str, Any]:
    """Execute hybrid_search + generate_response for one example."""
    # Lazy imports so ``import evaluation`` stays cheap and side-effect free.
    from core.retrieval import hybrid_search
    from core.generation import generate_response
    from utils.helpers import format_context_for_llm

    modality_filter = example.modality_filter or cfg.modality_filter

    t0 = time.perf_counter()
    try:
        retrieved = hybrid_search(
            query=example.query,
            top_k_retrieval=cfg.top_k_retrieval,
            top_k_rerank=cfg.top_k_rerank,
            dense_weight=cfg.dense_weight,
            bm25_weight=cfg.bm25_weight,
            modality_filter=modality_filter,
            enable_reranking=cfg.enable_reranking,
        )
    except Exception as e:
        return {
            "retrieved": [],
            "answer": "",
            "error": f"retrieval_error: {str(e)[:200]}",
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
        }

    context_str = format_context_for_llm(retrieved) if retrieved else ""

    # Stream → collect into a single string.
    answer_parts: List[str] = []
    try:
        for tok in generate_response(
            question=example.query,
            context=context_str,
            provider=cfg.provider,
            api_keys=api_keys,
            chat_history=None,
        ):
            answer_parts.append(tok)
    except Exception as e:
        return {
            "retrieved": retrieved,
            "answer": "".join(answer_parts),
            "error": f"generation_error: {str(e)[:200]}",
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
        }

    return {
        "retrieved": retrieved,
        "answer": "".join(answer_parts),
        "error": None,
        "latency_ms": (time.perf_counter() - t0) * 1000.0,
    }


# ─── Full run ───────────────────────────────────────────────────────────

def run_evaluation(
    dataset: Iterable[EvalExample] | str | Path,
    config: EvalConfig,
    api_keys: Optional[Dict[str, str]] = None,
    output_dir: str | Path = "evaluation/reports",
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Run one config over the full dataset and persist per-example +
    aggregate reports.

    Returns:
        A dict with keys: ``config``, ``aggregate``, ``per_example`` (list).
    """
    # Normalise inputs
    if isinstance(dataset, (str, Path)):
        examples = load_dataset(dataset)
    else:
        examples = list(dataset)
    if not examples:
        raise ValueError("Empty evaluation dataset")

    api_keys = api_keys or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_example: List[Dict[str, Any]] = []
    retrieval_metrics_list: List[Dict[str, float]] = []
    generation_metrics_list: List[Dict[str, float]] = []

    total = len(examples)
    for i, ex in enumerate(examples):
        run = _run_pipeline_once(ex, config, api_keys)
        retrieved = run["retrieved"]
        answer = run["answer"]

        r_metrics = compute_retrieval_metrics(ex, retrieved, ks=config.ks)
        retrieval_metrics_list.append(r_metrics)

        g_metrics = compute_generation_metrics(
            question=ex.query,
            answer=answer,
            retrieved_chunks=retrieved,
            gold_answer=ex.gold_answer,
            api_keys=api_keys,
            run_llm_judge=config.run_llm_judge,
            run_bertscore=config.run_bertscore,
        )
        # Drop None values from numeric aggregation later.
        numeric_g = {k: v for k, v in g_metrics.items()
                     if isinstance(v, (int, float)) and v is not None}
        generation_metrics_list.append(numeric_g)

        per_example.append({
            "id": ex.id,
            "query": ex.query,
            "expected_sources": ex.expected_sources,
            "expected_substrings": ex.expected_substrings,
            "gold_answer": ex.gold_answer,
            "answer": answer,
            "retrieved": [
                {
                    "source": (c.get("metadata", {}) or {}).get("source"),
                    "modality": (c.get("metadata", {}) or {}).get("modality"),
                    "rerank_score": c.get("rerank_score"),
                    "fusion_score": c.get("fusion_score"),
                    "content_preview": (c.get("content") or "")[:200],
                }
                for c in retrieved
            ],
            "retrieval_metrics": r_metrics,
            "generation_metrics": g_metrics,
            "latency_ms": run["latency_ms"],
            "error": run["error"],
        })

        if progress_callback:
            try:
                progress_callback((i + 1) / total, ex.id)
            except Exception:
                pass

    aggregate = {
        "retrieval": aggregate_metrics(retrieval_metrics_list),
        "generation": aggregate_metrics(generation_metrics_list),
        "num_examples": total,
        "num_errors": sum(1 for p in per_example if p["error"]),
        "mean_latency_ms": (
            sum(p["latency_ms"] for p in per_example) / total if total else 0.0
        ),
    }

    report = {
        "config": config.to_dict(),
        "aggregate": aggregate,
        "per_example": per_example,
    }

    # Persist artefacts
    _write_report(report, output_dir, config.name)

    return report


# ─── A/B comparison ─────────────────────────────────────────────────────

def compare_configs(
    dataset: Iterable[EvalExample] | str | Path,
    configs: List[EvalConfig],
    api_keys: Optional[Dict[str, str]] = None,
    output_dir: str | Path = "evaluation/reports",
    progress_callback=None,
) -> Dict[str, Any]:
    """Run each config sequentially and produce a side-by-side report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reports: Dict[str, Dict[str, Any]] = {}
    for i, cfg in enumerate(configs):
        def _cb(p: float, qid: str, _i=i, _n=len(configs), _name=cfg.name):
            if progress_callback:
                overall = (_i + p) / _n
                progress_callback(overall, f"{_name}:{qid}")

        reports[cfg.name] = run_evaluation(
            dataset=dataset,
            config=cfg,
            api_keys=api_keys,
            output_dir=output_dir,
            progress_callback=_cb,
        )

    # Side-by-side aggregate
    comparison = {
        name: r["aggregate"] for name, r in reports.items()
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "comparison.md").write_text(
        _render_comparison_md(comparison), encoding="utf-8"
    )
    return {"reports": reports, "comparison": comparison}


# ─── Report writers ─────────────────────────────────────────────────────

def _write_report(report: Dict[str, Any], output_dir: Path, cfg_name: str) -> None:
    safe = cfg_name.replace("/", "_").replace(" ", "_")
    (output_dir / f"{safe}_per_example.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, default=str)
                  for r in report["per_example"]),
        encoding="utf-8",
    )
    (output_dir / f"{safe}_aggregate.json").write_text(
        json.dumps(
            {"config": report["config"], "aggregate": report["aggregate"]},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (output_dir / f"{safe}_report.md").write_text(
        _render_single_md(report), encoding="utf-8"
    )


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _render_single_md(report: Dict[str, Any]) -> str:
    cfg = report["config"]
    agg = report["aggregate"]
    lines: List[str] = []
    lines.append(f"# Evaluation Report — `{cfg['name']}`\n")
    lines.append("## Configuration\n")
    lines.append("```json")
    lines.append(json.dumps(cfg, indent=2))
    lines.append("```\n")
    lines.append(f"**Examples:** {agg['num_examples']}  ")
    lines.append(f"**Errors:** {agg['num_errors']}  ")
    lines.append(f"**Mean latency:** {agg['mean_latency_ms']:.1f} ms\n")

    if agg["retrieval"]:
        lines.append("## Retrieval Metrics (mean)\n")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for k, v in sorted(agg["retrieval"].items()):
            lines.append(f"| `{k}` | {_fmt(v)} |")
        lines.append("")

    if agg["generation"]:
        lines.append("## Generation Metrics (mean)\n")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for k, v in sorted(agg["generation"].items()):
            lines.append(f"| `{k}` | {_fmt(v)} |")
        lines.append("")

    return "\n".join(lines)


def _render_comparison_md(comparison: Dict[str, Dict[str, Any]]) -> str:
    if not comparison:
        return "# Comparison\n\n(no configs)\n"

    # Collect every metric across all configs.
    metric_keys = set()
    for name, agg in comparison.items():
        for section in ("retrieval", "generation"):
            metric_keys.update(agg.get(section, {}).keys())
    metric_keys = sorted(metric_keys)
    config_names = list(comparison.keys())

    lines = ["# A/B Configuration Comparison\n"]
    header = "| Metric | " + " | ".join(f"`{n}`" for n in config_names) + " |"
    sep = "|---" + "|---" * len(config_names) + "|"
    lines.append(header)
    lines.append(sep)
    for m in metric_keys:
        row = [f"`{m}`"]
        for name in config_names:
            agg = comparison[name]
            v = agg.get("retrieval", {}).get(m, agg.get("generation", {}).get(m))
            row.append(_fmt(v) if v is not None else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Latency & Errors\n")
    lines.append("| Stat | " + " | ".join(f"`{n}`" for n in config_names) + " |")
    lines.append(sep)
    lines.append("| `num_examples` | " + " | ".join(
        _fmt(comparison[n].get("num_examples")) for n in config_names) + " |")
    lines.append("| `num_errors` | " + " | ".join(
        _fmt(comparison[n].get("num_errors")) for n in config_names) + " |")
    lines.append("| `mean_latency_ms` | " + " | ".join(
        _fmt(comparison[n].get("mean_latency_ms")) for n in config_names) + " |")
    lines.append("")
    return "\n".join(lines)
