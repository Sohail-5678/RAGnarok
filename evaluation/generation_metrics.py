"""
Generation-quality metrics for RAG evaluation.

Two families of metrics are provided:

  1. **Reference-based** (require a ``gold_answer``):
       - BLEU (NLTK, smoothed)
       - ROUGE-1 / ROUGE-2 / ROUGE-L (``rouge-score``)
       - BERTScore (optional, lazy import; skipped if not installed)

  2. **Reference-free / RAG-specific**:
       - Citation coverage — fraction of retrieved sources that appear
         cited in the answer.
       - LLM-as-judge faithfulness — does the answer only contain claims
         supported by the retrieved context? (binary 0/1 from Groq Llama)
       - LLM-as-judge answer relevancy — does the answer address the
         question? (0/1)

All heavy / network-dependent dependencies are lazy-imported so that
``import evaluation`` stays cheap.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional


# ─── Reference-based metrics ────────────────────────────────────────────

def bleu_score(prediction: str, reference: str) -> float:
    """Smoothed sentence-level BLEU-4. Returns 0.0 on any error."""
    if not prediction or not reference:
        return 0.0
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smoother = SmoothingFunction().method1
        return float(sentence_bleu(
            [reference.split()],
            prediction.split(),
            smoothing_function=smoother,
        ))
    except Exception:
        return 0.0


def rouge_scores(prediction: str, reference: str) -> Dict[str, float]:
    """ROUGE-1 / ROUGE-2 / ROUGE-L F1 scores. Empty dict on error."""
    if not prediction or not reference:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        scores = scorer.score(reference, prediction)
        return {k: float(v.fmeasure) for k, v in scores.items()}
    except Exception:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def bert_score_safe(prediction: str, reference: str) -> Optional[float]:
    """
    BERTScore F1 between prediction and reference. Returns ``None`` if
    ``bert-score`` is not installed (it's a heavy optional dependency).
    """
    if not prediction or not reference:
        return 0.0
    try:
        from bert_score import score as _bs_score  # type: ignore
    except Exception:
        return None
    try:
        _, _, f1 = _bs_score([prediction], [reference], lang="en", verbose=False)
        return float(f1.mean().item())
    except Exception:
        return None


# ─── Citation coverage ──────────────────────────────────────────────────

_CITATION_PATTERN = re.compile(r"\[\s*Source\s*:\s*([^\],]+)", re.IGNORECASE)


def citation_coverage(
    answer: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Measure how well the answer's citations align with the retrieved sources.

    Returns:
      * ``coverage``   — fraction of retrieved sources that appear cited
                         at least once in the answer.
      * ``fake_citation_rate`` — fraction of distinct citations in the
                         answer that do NOT match any retrieved source
                         (a proxy for hallucinated citations).
      * ``num_citations`` — total count of citation markers found.
    """
    if not answer:
        return {"coverage": 0.0, "fake_citation_rate": 0.0, "num_citations": 0.0}

    cited = {m.group(1).strip() for m in _CITATION_PATTERN.finditer(answer)}
    retrieved_sources = {
        (c.get("metadata", {}) or {}).get("source", "")
        for c in retrieved_chunks
        if (c.get("metadata", {}) or {}).get("source")
    }

    if not retrieved_sources:
        return {
            "coverage": 0.0,
            "fake_citation_rate": 1.0 if cited else 0.0,
            "num_citations": float(len(cited)),
        }

    hits = {c for c in cited if any(c in s or s in c for s in retrieved_sources)}
    fakes = cited - hits
    coverage = len(hits & retrieved_sources) / len(retrieved_sources) \
        if retrieved_sources else 0.0
    # Loosen ``coverage`` to allow partial filename matches in citations:
    matched_sources = {
        s for s in retrieved_sources if any(c in s or s in c for c in cited)
    }
    coverage = len(matched_sources) / len(retrieved_sources)

    fake_rate = (len(fakes) / len(cited)) if cited else 0.0
    return {
        "coverage": coverage,
        "fake_citation_rate": fake_rate,
        "num_citations": float(len(cited)),
    }


# ─── LLM-as-judge ───────────────────────────────────────────────────────

_JUDGE_PROMPT_FAITHFULNESS = """You are a strict evaluator. Decide whether the ANSWER is fully supported by the CONTEXT.

CONTEXT:
{context}

ANSWER:
{answer}

Rules:
- Score 1 only if EVERY factual claim in the answer is directly supported by the context.
- Score 0 if any claim is missing from the context, contradicts it, or is fabricated.
- Generic disclaimers like "I don't know" count as 1 if no unsupported claim is made.

Respond with a single JSON object only, no markdown, no commentary:
{{"score": 0 or 1, "reason": "one short sentence"}}"""

_JUDGE_PROMPT_ANSWER_RELEVANCY = """You are a strict evaluator. Decide whether the ANSWER addresses the QUESTION.

QUESTION:
{question}

ANSWER:
{answer}

Rules:
- Score 1 if the answer directly addresses the question (even if it says it doesn't know).
- Score 0 if the answer is off-topic, evasive, or unrelated.

Respond with a single JSON object only, no markdown, no commentary:
{{"score": 0 or 1, "reason": "one short sentence"}}"""


def _call_groq_judge(prompt: str, api_key: str, model: str = "llama-3.3-70b-versatile") -> Dict[str, Any]:
    """Invoke Groq with a strict JSON-only judge prompt. Returns parsed dict or fallback."""
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        return json.loads(raw)
    except Exception as e:
        return {"score": None, "reason": f"judge_error: {str(e)[:120]}"}


def _resolve_judge_key(api_keys: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Find a Groq key from session keys or environment."""
    if api_keys:
        k = (api_keys.get("groq") or "").strip()
        if k:
            return k
    return (os.environ.get("GROQ_API_KEY") or "").strip() or None


def llm_judge_faithfulness(
    answer: str,
    retrieved_chunks: List[Dict[str, Any]],
    api_keys: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Faithfulness/groundedness via Groq LLM judge."""
    key = _resolve_judge_key(api_keys)
    if not key or not answer:
        return {"score": None, "reason": "no_judge_key_or_empty_answer"}
    context = "\n\n".join(
        f"[{(c.get('metadata', {}) or {}).get('source','?')}] {c.get('content','')[:800]}"
        for c in retrieved_chunks[:8]
    )
    prompt = _JUDGE_PROMPT_FAITHFULNESS.format(context=context, answer=answer)
    return _call_groq_judge(prompt, key)


def llm_judge_answer_relevancy(
    question: str,
    answer: str,
    api_keys: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Answer-relevancy via Groq LLM judge."""
    key = _resolve_judge_key(api_keys)
    if not key or not answer:
        return {"score": None, "reason": "no_judge_key_or_empty_answer"}
    prompt = _JUDGE_PROMPT_ANSWER_RELEVANCY.format(question=question, answer=answer)
    return _call_groq_judge(prompt, key)


# ─── Top-level convenience ──────────────────────────────────────────────

def compute_generation_metrics(
    question: str,
    answer: str,
    retrieved_chunks: List[Dict[str, Any]],
    gold_answer: Optional[str] = None,
    api_keys: Optional[Dict[str, str]] = None,
    run_llm_judge: bool = True,
    run_bertscore: bool = False,
) -> Dict[str, Any]:
    """
    Compute the full generation-metrics suite for one (question, answer) pair.

    Returns a flat dict suitable for logging / aggregation. Missing metrics
    (e.g. no gold answer, no Groq key, BERTScore not installed) are simply
    omitted or set to ``None`` rather than raising.
    """
    out: Dict[str, Any] = {}

    # Reference-based
    if gold_answer:
        out["bleu"] = bleu_score(answer, gold_answer)
        out.update(rouge_scores(answer, gold_answer))
        if run_bertscore:
            bs = bert_score_safe(answer, gold_answer)
            if bs is not None:
                out["bertscore_f1"] = bs

    # Citation coverage
    out.update(citation_coverage(answer, retrieved_chunks))

    # LLM-as-judge
    if run_llm_judge:
        faith = llm_judge_faithfulness(answer, retrieved_chunks, api_keys)
        rel = llm_judge_answer_relevancy(question, answer, api_keys)
        out["faithfulness"] = faith.get("score")
        out["faithfulness_reason"] = faith.get("reason")
        out["answer_relevancy"] = rel.get("score")
        out["answer_relevancy_reason"] = rel.get("reason")

    return out
