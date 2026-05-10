"""
Optional RAGAS / TruLens / DeepEval bridges.

These third-party evaluation frameworks are heavy and have conflicting
dependency chains, so we keep them strictly optional. None of the
imports run unless the caller explicitly invokes the corresponding
function and the package is installed.

Functions return a metrics dict on success, or a dict with an
``"error"`` key explaining why evaluation was skipped — never raise.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


def _is_installed(pkg: str) -> bool:
    try:
        __import__(pkg)
        return True
    except Exception:
        return False


# ─── RAGAS ──────────────────────────────────────────────────────────────

def run_ragas(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: Optional[List[str]] = None,
    groq_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run RAGAS on a batch of (question, answer, contexts) tuples.

    RAGAS metrics computed (when ``ground_truths`` is provided, the
    reference-requiring metrics are added automatically):

      * faithfulness
      * answer_relevancy
      * context_precision
      * context_recall (needs ground_truths)
      * answer_correctness (needs ground_truths)

    The default RAGAS LLM is OpenAI; we override it to use Groq via the
    OpenAI-compatible endpoint so the free Groq key works.
    """
    if not _is_installed("ragas") or not _is_installed("datasets"):
        return {"error": "ragas not installed — `pip install ragas datasets`"}

    try:
        from datasets import Dataset  # type: ignore
        from ragas import evaluate  # type: ignore
        from ragas.metrics import (  # type: ignore
            faithfulness, answer_relevancy, context_precision,
        )
        metrics = [faithfulness, answer_relevancy, context_precision]

        if ground_truths:
            try:
                from ragas.metrics import context_recall, answer_correctness  # type: ignore
                metrics.extend([context_recall, answer_correctness])
            except Exception:
                pass

        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        if ground_truths:
            data["ground_truth"] = ground_truths
        ds = Dataset.from_dict(data)

        # Wire RAGAS to use Groq via the OpenAI-compatible endpoint.
        # This avoids needing an OpenAI key on the free tier.
        groq_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
        llm = None
        if groq_key and _is_installed("langchain_openai"):
            from langchain_openai import ChatOpenAI  # type: ignore
            llm = ChatOpenAI(
                model="llama-3.3-70b-versatile",
                api_key=groq_key,
                base_url="https://api.groq.com/openai/v1",
                temperature=0.0,
            )

        result = evaluate(ds, metrics=metrics, llm=llm) if llm else evaluate(ds, metrics=metrics)
        # ``result`` is a ragas Result object; coerce to plain dict of means.
        return {k: float(v) for k, v in result.to_pandas().mean(numeric_only=True).items()}
    except Exception as e:
        return {"error": f"ragas evaluation failed: {str(e)[:200]}"}


# ─── DeepEval ───────────────────────────────────────────────────────────

def run_deepeval(
    question: str,
    answer: str,
    contexts: List[str],
    gold_answer: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a small DeepEval suite on a single example. Skipped if not installed."""
    if not _is_installed("deepeval"):
        return {"error": "deepeval not installed — `pip install deepeval`"}
    try:
        from deepeval.test_case import LLMTestCase  # type: ignore
        from deepeval.metrics import (  # type: ignore
            FaithfulnessMetric, AnswerRelevancyMetric, ContextualRelevancyMetric,
        )

        tc = LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=contexts,
            expected_output=gold_answer,
        )
        out: Dict[str, Any] = {}
        for cls, name in [
            (FaithfulnessMetric, "deepeval_faithfulness"),
            (AnswerRelevancyMetric, "deepeval_answer_relevancy"),
            (ContextualRelevancyMetric, "deepeval_context_relevancy"),
        ]:
            try:
                m = cls(threshold=0.5)
                m.measure(tc)
                out[name] = float(m.score)
            except Exception as e:
                out[name] = None
                out[f"{name}_error"] = str(e)[:120]
        return out
    except Exception as e:
        return {"error": f"deepeval failed: {str(e)[:200]}"}


# ─── TruLens ────────────────────────────────────────────────────────────

def trulens_feedback_stubs() -> Dict[str, Any]:
    """
    TruLens integration is best done at the application boundary (it
    wraps the LLM call and instruments feedback functions). This helper
    just reports whether TruLens is importable so callers can wire it in
    themselves.
    """
    if not _is_installed("trulens_eval"):
        return {"error": "trulens-eval not installed — `pip install trulens-eval`"}
    return {"available": True, "note": "wire TruLens at the generate_response boundary"}
