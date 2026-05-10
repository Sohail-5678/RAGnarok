"""
Chat interface: message rendering, streaming, and context display.
"""
import time
import streamlit as st
from typing import List, Dict, Any, Optional

from config import LLM_PROVIDERS, TOP_K_RERANK
from ui.styles import render_source_tag
from core.retrieval import hybrid_search
from core.generation import generate_response
from utils.helpers import format_context_for_llm
from evaluation.logger import (
    QueryLogEntry, get_default_logger, new_session_id,
)
from evaluation.generation_metrics import compute_generation_metrics


def render_chat():
    """Render the main chat interface."""
    # Initialize messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing messages (with index + paired question for scoring).
    for idx, msg in enumerate(st.session_state.messages):
        paired_question = None
        if msg.get("role") == "assistant" and idx > 0:
            prev = st.session_state.messages[idx - 1]
            if prev.get("role") == "user":
                paired_question = prev.get("content", "")
        _render_message(msg, idx, paired_question)

    # Chat input
    if prompt := st.chat_input("Ask anything about your uploaded documents...", key="chat_input"):
        # Add user message
        user_msg = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_msg)
        _render_message(user_msg, len(st.session_state.messages) - 1, None)

        # Generate response
        with st.chat_message("assistant", avatar="🔮"):
            _generate_and_stream(prompt)


def _render_message(msg: Dict[str, Any], idx: int = 0, paired_question: Optional[str] = None):
    """Render a single chat message, plus inline scoring for assistant turns."""
    role = msg["role"]
    avatar = "👤" if role == "user" else "🔮"

    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["content"])

        # Render sources + scoring controls for assistant messages.
        if role == "assistant":
            if "sources" in msg:
                _render_sources(msg["sources"])
            if msg.get("retrieved") is not None and paired_question:
                _render_score_controls(msg, idx, paired_question)


def _generate_and_stream(prompt: str):
    """Generate response with RAG retrieval and stream it."""
    api_keys = st.session_state.get("api_keys", {})
    provider = st.session_state.get("selected_provider", "groq")
    top_k = st.session_state.get("top_k", TOP_K_RERANK)
    enable_reranking = st.session_state.get("enable_reranking", True)
    modality_filter = st.session_state.get("modality_filter", "all")
    dense_weight = st.session_state.get("dense_weight", 0.7)
    bm25_weight = st.session_state.get("bm25_weight", 0.3)

    # ── Logging setup ─────────────────────────────────────
    t_start = time.perf_counter()
    if "session_id" not in st.session_state:
        st.session_state.session_id = new_session_id()
    retrieved_for_log: List[Dict[str, Any]] = []

    # ── Retrieve relevant context ─────────────────────────
    context = ""
    sources = []

    from core.vector_store import get_or_create_collection
    collection = get_or_create_collection()

    if collection.count() > 0:
        with st.status("🔍 Searching knowledge base...", expanded=False) as status:
            st.write("Running hybrid search (Dense + BM25)...")
            
            results = hybrid_search(
                query=prompt,
                top_k_retrieval=20,
                top_k_rerank=top_k,
                dense_weight=dense_weight,
                bm25_weight=bm25_weight,
                modality_filter=modality_filter if modality_filter != "all" else None,
                enable_reranking=enable_reranking,
            )

            if results:
                st.write(f"Found {len(results)} relevant chunks")
                if enable_reranking:
                    st.write("Applied cross-encoder re-ranking ✓")

                context = format_context_for_llm(results)
                retrieved_for_log = [
                    {
                        "content": r.get("content", ""),
                        "metadata": r.get("metadata", {}),
                        "rerank_score": r.get("rerank_score"),
                        "fusion_score": r.get("fusion_score"),
                        "dense_similarity": r.get("dense_similarity"),
                        "bm25_score": r.get("bm25_score"),
                    }
                    for r in results
                ]
                sources = [
                    {
                        "source": r.get("metadata", {}).get("source", "Unknown"),
                        "modality": r.get("metadata", {}).get("modality", "text"),
                        "score": r.get("rerank_score", r.get("fusion_score", 0)),
                        "metadata": r.get("metadata", {}),
                    }
                    for r in results
                ]

                status.update(label=f"✅ Retrieved {len(results)} chunks", state="complete")
            else:
                st.write("No matching chunks found")
                status.update(label="⚠️ No relevant results", state="complete")

    # ── Stream LLM response ───────────────────────────────
    provider_info = LLM_PROVIDERS.get(provider, {})
    model_name = provider_info.get("name", provider)

    # Get chat history (exclude current prompt)
    chat_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]  # Exclude the just-added user msg
    ]

    full_response = ""
    response_placeholder = st.empty()

    try:
        for token in generate_response(
            question=prompt,
            context=context,
            provider=provider,
            api_keys=api_keys,
            chat_history=chat_history,
        ):
            full_response += token
            response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

    except Exception as e:
        full_response = f"❌ Generation error: {str(e)[:300]}"
        response_placeholder.markdown(full_response)

    # ── Render sources ────────────────────────────────────
    if sources:
        _render_sources(sources)

    # ── Save assistant message ────────────────────────────
    assistant_msg = {
        "role": "assistant",
        "content": full_response,
        # Persist the full retrieved chunks so the user can later click
        # "📈 Score this answer" without re-running retrieval.
        "retrieved": retrieved_for_log,
    }
    if sources:
        assistant_msg["sources"] = sources

    st.session_state.messages.append(assistant_msg)

    # ── Persistent query log (opt-in) ─────────────────────
    if st.session_state.get("persist_queries", False):
        try:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            entry = QueryLogEntry(
                query=prompt,
                answer=full_response,
                retrieved=retrieved_for_log,
                sources=sources,
                provider=provider,
                config={
                    "top_k_rerank": top_k,
                    "dense_weight": dense_weight,
                    "bm25_weight": bm25_weight,
                    "enable_reranking": enable_reranking,
                    "modality_filter": modality_filter,
                },
                latency_ms=latency_ms,
                session_id=st.session_state.get("session_id", ""),
            )
            get_default_logger().log(entry)
        except Exception:
            # Never let logging break the chat experience.
            pass

    # Rerun so the freshly-saved message renders through ``_render_message``,
    # which is what attaches the "📈 Score this answer" button.
    st.rerun()


def _render_score_controls(msg: Dict[str, Any], idx: int, question: str):
    """
    Render the inline "Score this answer" UI under an assistant message.

    Runs three reference-free metrics on demand (no gold answer required):
      * Faithfulness        — LLM-as-judge against retrieved context (0/1)
      * Answer Relevancy    — LLM-as-judge against the question      (0/1)
      * Citation coverage   — regex-based, deterministic

    Results are cached on the message itself so re-renders are free.
    """
    # If already scored, just render the card.
    if "scores" in msg:
        _render_score_card(msg["scores"])
        # Allow user to re-score if they want.
        if st.button("🔄 Re-score", key=f"rescore_{idx}"):
            msg.pop("scores", None)
            st.rerun()
        return

    col_btn, col_hint = st.columns([1, 3])
    with col_btn:
        clicked = st.button("📈 Score this answer", key=f"score_{idx}")
    with col_hint:
        st.caption(
            "Runs faithfulness + relevancy (LLM judge) and citation coverage on this answer."
        )

    if clicked:
        api_keys = st.session_state.get("api_keys", {})
        with st.spinner("Scoring with Groq LLM judge…"):
            try:
                scores = compute_generation_metrics(
                    question=question,
                    answer=msg.get("content", ""),
                    retrieved_chunks=msg.get("retrieved", []),
                    gold_answer=None,
                    api_keys=api_keys,
                    run_llm_judge=True,
                    run_bertscore=False,
                )
            except Exception as e:
                scores = {"error": f"Scoring failed: {str(e)[:200]}"}
        msg["scores"] = scores
        st.rerun()


def _render_score_card(scores: Dict[str, Any]):
    """Render the scoring results as a compact glass card."""
    if scores.get("error"):
        st.error(scores["error"])
        return

    def _badge(label: str, value, kind: Optional[str] = None) -> str:
        """One badge: green for 1, amber for 0, grey for None/unknown."""
        if value is None:
            color, glyph = "#64748b", "—"
        elif isinstance(value, (int, float)) and value >= 0.999:
            color, glyph = "#10b981", "✓"
        elif isinstance(value, (int, float)) and value <= 0.001:
            color, glyph = "#f43f5e", "✗"
        else:
            color, glyph = "#f59e0b", f"{value:.2f}" if isinstance(value, float) else str(value)
        return (
            f"<div style='display:inline-flex;align-items:center;gap:0.4rem;"
            f"padding:0.35rem 0.7rem;border-radius:20px;"
            f"background:rgba(255,255,255,0.04);"
            f"border:1px solid {color}55;color:{color};"
            f"font-size:0.75rem;font-weight:600;margin:0.2rem;'>"
            f"<span>{label}</span><span>{glyph}</span></div>"
        )

    faith = scores.get("faithfulness")
    rel = scores.get("answer_relevancy")
    cov = scores.get("coverage")
    fake = scores.get("fake_citation_rate")
    num_cit = int(scores.get("num_citations") or 0)

    badges_html = (
        "<div style='margin-top:0.7rem;padding:0.6rem 0.8rem;"
        "background:rgba(255,255,255,0.03);"
        "border:1px solid rgba(255,255,255,0.08);border-radius:12px;'>"
        "<div style='font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;"
        "color:#94a3b8;margin-bottom:0.4rem;'>📈 Answer Quality</div>"
    )
    badges_html += _badge("Faithfulness", faith)
    badges_html += _badge("Relevancy", rel)
    badges_html += _badge(f"Citations ({num_cit})", cov)
    if fake is not None and fake > 0:
        badges_html += _badge("Fake citations", 1.0 - fake)  # invert so low is good
    badges_html += "</div>"
    st.markdown(badges_html, unsafe_allow_html=True)

    # Show judge reasoning in an expander for transparency.
    faith_reason = scores.get("faithfulness_reason")
    rel_reason = scores.get("answer_relevancy_reason")
    if faith_reason or rel_reason:
        with st.expander("Judge reasoning", expanded=False):
            if faith_reason:
                st.caption(f"**Faithfulness:** {faith_reason}")
            if rel_reason:
                st.caption(f"**Answer relevancy:** {rel_reason}")


def _render_sources(sources: List[Dict[str, Any]]):
    """Render source citations below a message."""
    if not sources:
        return

    # Deduplicate sources
    seen = set()
    unique_sources = []
    for s in sources:
        key = f"{s['source']}_{s['modality']}"
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    source_html = "<div style='margin-top: 0.8rem; display: flex; flex-wrap: wrap; gap: 0.3rem;'>"
    source_html += "<span style='font-size: 0.7rem; color: #64748b; margin-right: 0.3rem; align-self: center;'>Sources:</span>"

    for s in unique_sources:
        source_html += render_source_tag(s["source"], s["modality"])

    source_html += "</div>"

    st.markdown(source_html, unsafe_allow_html=True)


def render_welcome():
    """Render welcome screen when no messages exist."""
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 3rem 2rem;">
        <div style="font-size: 3.5rem; margin-bottom: 1rem;">🔮</div>
        <h2 style="background: linear-gradient(135deg, #8b5cf6, #3b82f6, #06b6d4); 
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                    background-clip: text; font-size: 1.8rem; margin-bottom: 0.5rem;">
            Welcome to RAGnarok
        </h2>
        <p style="color: #94a3b8; font-size: 1rem; max-width: 500px; margin: 0 auto 1.5rem;">
            Upload documents, audio, video, or images — then ask questions. 
            Your AI assistant uses hybrid search and re-ranking to find the most relevant answers.
        </p>
        <div style="display: flex; justify-content: center; gap: 1.5rem; flex-wrap: wrap; margin-top: 1.5rem;">
            <div class="metric-card" style="min-width: 120px;">
                <div style="font-size: 1.5rem;">📄</div>
                <div class="metric-label">PDFs & Docs</div>
            </div>
            <div class="metric-card" style="min-width: 120px;">
                <div style="font-size: 1.5rem;">🎵</div>
                <div class="metric-label">Audio Files</div>
            </div>
            <div class="metric-card" style="min-width: 120px;">
                <div style="font-size: 1.5rem;">🎬</div>
                <div class="metric-label">Video Files</div>
            </div>
            <div class="metric-card" style="min-width: 120px;">
                <div style="font-size: 1.5rem;">🖼️</div>
                <div class="metric-label">Images</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick start tips
    st.markdown("""
    <div class="glass-card" style="margin-top: 1rem;">
        <h4 style="color: #e2e8f0; margin-bottom: 1rem;">⚡ Quick Start</h4>
        <ol style="color: #94a3b8; line-height: 2;">
            <li><strong style="color: #e2e8f0;">Configure API Keys</strong> — Add your Groq, OpenAI, Gemini, or Claude API key in the sidebar</li>
            <li><strong style="color: #e2e8f0;">Upload Files</strong> — Drag & drop PDFs, audio, video, or images into the sidebar uploader</li>
            <li><strong style="color: #e2e8f0;">Process & Index</strong> — Click "Process & Index" to chunk and embed your files</li>
            <li><strong style="color: #e2e8f0;">Ask Questions</strong> — Type your question below and get grounded, cited answers</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
