"""
RAGnarok — Production-Grade Multimodal RAG Application
======================================================

A Streamlit-powered Retrieval-Augmented Generation system supporting
PDFs, audio, video, and images with hybrid search, cross-encoder
re-ranking, and multi-LLM routing.

Author: Ameer Sohail
License: MIT
"""
import os
import sys
import streamlit as st

# ── Page Config (must be first Streamlit call) ────────────────────────
st.set_page_config(
    page_title="RAGnarok — Multimodal RAG",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/yourusername/ragnarok",
        "Report a Bug": "https://github.com/yourusername/ragnarok/issues",
        "About": "RAGnarok — Production-Grade Multimodal RAG Application",
    },
)

# ── Apply Glass UI Theme ─────────────────────────────────────────────
from ui.styles import get_glass_css, render_header, render_metric_card
st.markdown(get_glass_css(), unsafe_allow_html=True)

# ── App Header ────────────────────────────────────────────────────────
st.markdown(render_header(), unsafe_allow_html=True)

# ── Render Sidebar ────────────────────────────────────────────────────
from ui.sidebar import render_sidebar
render_sidebar()

# ── Main Content Area ─────────────────────────────────────────────────
from ui.chat import render_chat, render_welcome
from core.vector_store import get_collection_stats


def main():
    """Main application logic."""
    # Get KB stats for metrics bar
    try:
        stats = get_collection_stats()
        has_data = stats["total_chunks"] > 0
    except Exception:
        stats = {"total_chunks": 0, "num_sources": 0, "modalities": []}
        has_data = False

    # ── Metrics Row ───────────────────────────────────────
    if has_data:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                render_metric_card(str(stats["total_chunks"]), "Total Chunks", "📦"),
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                render_metric_card(str(stats["num_sources"]), "Sources", "📂"),
                unsafe_allow_html=True,
            )
        with col3:
            modalities = len(stats.get("modalities", []))
            st.markdown(
                render_metric_card(str(modalities), "Modalities", "🎯"),
                unsafe_allow_html=True,
            )
        with col4:
            provider = st.session_state.get("selected_provider", "groq")
            from config import LLM_PROVIDERS
            pinfo = LLM_PROVIDERS.get(provider, {})
            st.markdown(
                render_metric_card(
                    provider.upper(),
                    "Active LLM",
                    pinfo.get("icon", "🤖"),
                ),
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # ── Chat or Welcome ───────────────────────────────────
    if not st.session_state.get("messages"):
        render_welcome()

    render_chat()


if __name__ == "__main__":
    main()