"""
Command-line runner for the RAGnarok evaluation harness.

Examples
--------
Run the default config on the example dataset::

    python -m evaluation.cli --dataset docs/examples/sample_dataset.jsonl

A/B compare reranking on vs off::

    python -m evaluation.cli \\
        --dataset docs/examples/sample_dataset.jsonl \\
        --ab rerank_on rerank_off

Custom config via JSON::

    python -m evaluation.cli \\
        --dataset docs/examples/sample_dataset.jsonl \\
        --config-json '{"name":"dense_only","bm25_weight":0.0,"dense_weight":1.0}'
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

from evaluation.harness import EvalConfig, run_evaluation, compare_configs


_BUILTIN_CONFIGS: Dict[str, EvalConfig] = {
    "default": EvalConfig(name="default"),
    "rerank_on": EvalConfig(name="rerank_on", enable_reranking=True),
    "rerank_off": EvalConfig(name="rerank_off", enable_reranking=False),
    "dense_only": EvalConfig(
        name="dense_only", dense_weight=1.0, bm25_weight=0.0,
    ),
    "sparse_only": EvalConfig(
        name="sparse_only", dense_weight=0.0, bm25_weight=1.0,
    ),
    "no_judge": EvalConfig(name="no_judge", run_llm_judge=False),
}


def _resolve_api_keys() -> Dict[str, str]:
    """Pick up keys from env so the CLI works without Streamlit."""
    return {
        "groq": os.environ.get("GROQ_API_KEY", ""),
        "openai": os.environ.get("OPENAI_API_KEY", ""),
        "gemini": os.environ.get("GOOGLE_API_KEY", ""),
        "claude": os.environ.get("ANTHROPIC_API_KEY", ""),
    }


def _parse_config_json(s: str) -> EvalConfig:
    obj = json.loads(s)
    return EvalConfig(**obj)


def _progress_printer(p: float, label: str) -> None:
    bar = "█" * int(p * 30) + "·" * (30 - int(p * 30))
    sys.stdout.write(f"\r  [{bar}] {p*100:5.1f}%  {label[:40]:40s}")
    sys.stdout.flush()
    if p >= 1.0:
        sys.stdout.write("\n")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evaluation.cli",
        description="Run the RAGnarok evaluation harness on a JSONL gold set.",
    )
    parser.add_argument("--dataset", required=True, help="Path to JSONL gold set.")
    parser.add_argument(
        "--output-dir", default="evaluation/reports",
        help="Directory to write reports into.",
    )
    parser.add_argument(
        "--config", default="default",
        help=f"Built-in config name. One of: {', '.join(_BUILTIN_CONFIGS)}",
    )
    parser.add_argument(
        "--config-json", default=None,
        help="Inline JSON for a custom EvalConfig (overrides --config).",
    )
    parser.add_argument(
        "--ab", nargs="+", default=None,
        help="Two or more built-in config names to A/B compare.",
    )
    parser.add_argument(
        "--no-judge", action="store_true",
        help="Disable the LLM-as-judge metrics (faster, no API calls).",
    )
    args = parser.parse_args(argv)

    api_keys = _resolve_api_keys()
    dataset_path = Path(args.dataset)

    if args.ab:
        try:
            configs = [_BUILTIN_CONFIGS[name] for name in args.ab]
        except KeyError as e:
            print(f"Unknown config name: {e}. Available: {list(_BUILTIN_CONFIGS)}",
                  file=sys.stderr)
            return 2
        if args.no_judge:
            configs = [
                EvalConfig(**{**c.to_dict(), "run_llm_judge": False, "ks": tuple(c.ks)})
                for c in configs
            ]
        print(f"Running A/B comparison: {args.ab}")
        result = compare_configs(
            dataset=dataset_path,
            configs=configs,
            api_keys=api_keys,
            output_dir=args.output_dir,
            progress_callback=_progress_printer,
        )
        print(f"\nDone. Reports written to {args.output_dir}/")
        print(f"Comparison summary: {args.output_dir}/comparison.md")
        return 0

    # Single-config run
    if args.config_json:
        config = _parse_config_json(args.config_json)
    else:
        if args.config not in _BUILTIN_CONFIGS:
            print(f"Unknown config: {args.config}. Available: {list(_BUILTIN_CONFIGS)}",
                  file=sys.stderr)
            return 2
        config = _BUILTIN_CONFIGS[args.config]

    if args.no_judge:
        config = EvalConfig(**{**config.to_dict(), "run_llm_judge": False,
                               "ks": tuple(config.ks)})

    print(f"Running config '{config.name}' on {dataset_path}")
    report = run_evaluation(
        dataset=dataset_path,
        config=config,
        api_keys=api_keys,
        output_dir=args.output_dir,
        progress_callback=_progress_printer,
    )
    agg = report["aggregate"]
    print(f"\nDone. {agg['num_examples']} examples, "
          f"{agg['num_errors']} errors, "
          f"mean latency {agg['mean_latency_ms']:.1f} ms")
    print(f"Reports: {args.output_dir}/{config.name}_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
