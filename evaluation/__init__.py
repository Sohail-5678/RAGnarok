"""
RAGnarok evaluation package.

Provides everything needed to evaluate the RAG pipeline offline:

  * Gold-set loading                       (``dataset``)
  * Retrieval metrics                      (``retrieval_metrics``)
  * Reference-based generation metrics     (``generation_metrics``)
  * LLM-as-judge metrics                   (``generation_metrics``)
  * Optional RAGAS / TruLens / DeepEval    (``ragas_eval``)
  * Persistent query→answer logger         (``logger``)
  * End-to-end evaluation harness          (``harness``)
  * A/B comparison of pipeline configs     (``harness``)
  * CLI runner                             (``cli``)
"""
from evaluation.dataset import EvalExample, load_dataset, save_dataset
from evaluation.retrieval_metrics import (
    hit_at_k, recall_at_k, precision_at_k, mrr, ndcg_at_k,
    compute_retrieval_metrics,
)
from evaluation.generation_metrics import (
    bleu_score, rouge_scores, bert_score_safe,
    citation_coverage, llm_judge_faithfulness, llm_judge_answer_relevancy,
    compute_generation_metrics,
)
from evaluation.logger import QueryLogger, get_default_logger
from evaluation.harness import EvalConfig, run_evaluation, compare_configs

__all__ = [
    "EvalExample", "load_dataset", "save_dataset",
    "hit_at_k", "recall_at_k", "precision_at_k", "mrr", "ndcg_at_k",
    "compute_retrieval_metrics",
    "bleu_score", "rouge_scores", "bert_score_safe",
    "citation_coverage", "llm_judge_faithfulness", "llm_judge_answer_relevancy",
    "compute_generation_metrics",
    "QueryLogger", "get_default_logger",
    "EvalConfig", "run_evaluation", "compare_configs",
]
