# RAGnarok Evaluation Suite

End-to-end offline evaluation for the RAGnarok multimodal RAG pipeline.

## What's Included

| Concern | Implementation |
|---|---|
| **Retrieval quality** | `retrieval_metrics.py` — Hit@K, Recall@K, Precision@K, MRR, nDCG@K |
| **Reference-based generation** | `generation_metrics.py` — BLEU (NLTK), ROUGE-1/2/L (`rouge-score`), BERTScore (optional) |
| **Faithfulness / Relevancy** | `generation_metrics.py` — Groq Llama-3.3 70B LLM-as-judge |
| **Citation coverage** | `generation_metrics.py` — regex extraction + source matching |
| **RAGAS / DeepEval / TruLens** | `ragas_eval.py` — optional, lazy-loaded |
| **Query logging** | `logger.py` — SQLite + JSONL, wired into `ui/chat.py` |
| **A/B harness** | `harness.py` + `cli.py` — compare any set of pipeline configs |
| **Unit tests** | `tests/` — pytest fixtures for chunking, retrieval, ingestion, metrics |

## Dataset Format

JSONL, one example per line:

```json
{
  "id": "q001",
  "query": "What is RAGnarok?",
  "expected_sources": ["readme.md"],
  "expected_substrings": ["multimodal", "rag"],
  "gold_answer": "RAGnarok is a multimodal RAG application…",
  "modality_filter": null
}
```

* **`expected_sources`** and **`expected_substrings`** define retrieval relevance.
  A chunk is relevant if its `metadata.source` is in `expected_sources` **or** its content contains any of the substrings (case-insensitive).
* **`gold_answer`** enables BLEU / ROUGE / BERTScore. Optional.
* **`modality_filter`** restricts retrieval to one modality. Optional.

See `docs/examples/sample_dataset.jsonl` for a working example you can copy and adapt.

## Quick Start (CLI)

```bash
# Default run on the example dataset
export GROQ_API_KEY="gsk_..."
python -m evaluation.cli --dataset docs/examples/sample_dataset.jsonl

# A/B compare reranking on vs off
python -m evaluation.cli \
    --dataset docs/examples/sample_dataset.jsonl \
    --ab rerank_on rerank_off

# Skip LLM-as-judge metrics for a fast dry-run
python -m evaluation.cli \
    --dataset docs/examples/sample_dataset.jsonl \
    --no-judge

# Compare dense-only vs sparse-only vs hybrid
python -m evaluation.cli \
    --dataset docs/examples/sample_dataset.jsonl \
    --ab dense_only sparse_only default
```

Reports land in `evaluation/reports/`:

* `<config>_per_example.jsonl` — full retrieval + answers + metrics per query
* `<config>_aggregate.json` — mean metrics for the config
* `<config>_report.md` — human-readable summary
* `comparison.md` — side-by-side table when `--ab` is used

## Built-in Configurations

| Name | Description |
|---|---|
| `default` | Hybrid (0.7 dense / 0.3 BM25) + rerank + LLM judge |
| `rerank_on` / `rerank_off` | Toggles the cross-encoder reranker |
| `dense_only` / `sparse_only` | Pure dense / pure BM25 |
| `no_judge` | Skip LLM-as-judge for speed |

Define your own with `--config-json '{"name":"x","dense_weight":0.5,...}'`.

## Programmatic API

```python
from evaluation import EvalConfig, run_evaluation, compare_configs

report = run_evaluation(
    dataset="docs/examples/sample_dataset.jsonl",
    config=EvalConfig(name="my_run", enable_reranking=False),
    api_keys={"groq": "gsk_..."},
)
print(report["aggregate"])
```

## Persistent Query Logging

Every live query is captured (when the **"Persist Queries"** toggle is on in the sidebar) to:

* `evaluation/logs/ragnarok_queries.db` (SQLite)
* `evaluation/logs/ragnarok_queries.jsonl` (append-only JSONL)

You can later turn those logs into an eval dataset:

```bash
# Cherry-pick interesting queries, add expected_substrings + gold_answer,
# save as JSONL, then feed back into the harness.
```

## Optional Frameworks

These are **not** in `requirements.txt`. Install on demand:

```bash
pip install ragas datasets langchain-openai     # RAGAS
pip install bert-score                          # BERTScore
pip install deepeval                            # DeepEval
pip install trulens-eval                        # TruLens
```

`evaluation/ragas_eval.py` exposes `run_ragas(...)` and `run_deepeval(...)` —
both return `{"error": "...not installed..."}` gracefully if the package is missing.

## Running the Unit Tests

```bash
pytest tests/ -v
```

No API keys are required for the test suite; network-dependent paths are skipped or mocked.
