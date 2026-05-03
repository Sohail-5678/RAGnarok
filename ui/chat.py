"""
Chat interface: message rendering, streaming, and context display.
"""
import streamlit as st
from typing import List, Dict, Any, Optional

from config import LLM_PROVIDERS, TOP_K_RERANK
from ui.styles import render_source_tag
from core.retrieval import hybrid_search
from core.generation import generate_response
from utils.helpers import format_context_for_llm


def render_chat():
    """Render the main chat interface."""
    # Initialize messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing messages
    for msg in st.session_state.messages:
        _render_message(msg)

    # Chat input
    if prompt := st.chat_input("Ask anything about your uploaded documents...", key="chat_input"):
        # Add user message
        user_msg = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_msg)
        _render_message(user_msg)

        # Generate response
        with st.chat_message("assistant", avatar="🔮"):
            _generate_and_stream(prompt)


def _render_message(msg: Dict[str, Any]):
    """Render a single chat message."""
    role = msg["role"]
    avatar = "👤" if role == "user" else "🔮"

    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["content"])

        # Render sources if available
        if role == "assistant" and "sources" in msg:
            _render_sources(msg["sources"])


def _generate_and_stream(prompt: str):
    """Generate response with RAG retrieval and stream it."""
    api_keys = st.session_state.get("api_keys", {})
    provider = st.session_state.get("selected_provider", "groq")
    top_k = st.session_state.get("top_k", TOP_K_RERANK)
    enable_reranking = st.session_state.get("enable_reranking", True)
    modality_filter = st.session_state.get("modality_filter", "all")
    dense_weight = st.session_state.get("dense_weight", 0.7)
    bm25_weight = st.session_state.get("bm25_weight", 0.3)

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
    }
    if sources:
        assistant_msg["sources"] = sources

    st.session_state.messages.append(assistant_msg)


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
