"""
Sidebar UI: API key management, file upload, and settings.
"""
import os
import asyncio
import streamlit as st
from typing import Dict, List, Optional

from config import (
    LLM_PROVIDERS, ALL_SUPPORTED, SUPPORTED_DOCUMENTS,
    SUPPORTED_AUDIO, SUPPORTED_VIDEO, SUPPORTED_IMAGES,
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP,
    TOP_K_RERANK, DENSE_WEIGHT, BM25_WEIGHT,
)
from ui.styles import render_status_badge
from utils.helpers import (
    get_file_extension, format_file_size, save_uploaded_file,
    cleanup_temp_file, validate_api_key,
)
from core.ingestion import ingest_files_parallel
from core.embeddings import generate_embeddings
from core.vector_store import (
    add_chunks_to_store, get_collection_stats, clear_collection,
    get_or_create_collection,
)


def render_sidebar():
    """Render the complete sidebar with API keys, upload, and settings."""
    with st.sidebar:
        # ── Logo & Title ──────────────────────────────────
        st.markdown("## 🔮 RAGnarok")
        st.markdown("---")

        # ── API Key Management ────────────────────────────
        _render_api_keys_section()

        st.markdown("---")

        # ── File Upload ───────────────────────────────────
        _render_upload_section()

        st.markdown("---")

        # ── Knowledge Base Stats ──────────────────────────
        _render_kb_stats()

        st.markdown("---")

        # ── Advanced Settings ─────────────────────────────
        _render_advanced_settings()

        st.markdown("---")

        # ── Actions ───────────────────────────────────────
        _render_actions()


def _render_api_keys_section():
    """Render API key input section."""
    st.markdown("### 🔑 API Keys")
    st.caption("Enter your API keys. Groq is the default if no key is provided.")

    # Initialize API keys in session state
    if "api_keys" not in st.session_state:
        st.session_state.api_keys = {
            "groq": os.environ.get("GROQ_API_KEY", ""),
            "openai": "",
            "gemini": "",
            "claude": "",
        }

    for provider_id, info in LLM_PROVIDERS.items():
        key = st.text_input(
            f"{info['icon']} {info['name']}",
            value=st.session_state.api_keys.get(provider_id, ""),
            type="password",
            key=f"api_key_{provider_id}",
            placeholder=f"Enter {info['name'].split('(')[0].strip()} API key...",
        )
        st.session_state.api_keys[provider_id] = key

    # Show which providers are active
    active_providers = []
    for pid, info in LLM_PROVIDERS.items():
        key = st.session_state.api_keys.get(pid, "").strip()
        if not key:
            key = os.environ.get(info["env_key"], "").strip()
        if key:
            active_providers.append(f"{info['icon']} {pid.capitalize()}")

    if active_providers:
        st.markdown(
            render_status_badge(f"Active: {', '.join(active_providers)}", "active"),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            render_status_badge("No API keys configured", "warning"),
            unsafe_allow_html=True,
        )

    # LLM Selection
    st.markdown("#### 🤖 Active Model")
    provider_options = {
        pid: f"{info['icon']} {info['name']}" for pid, info in LLM_PROVIDERS.items()
    }

    if "selected_provider" not in st.session_state:
        st.session_state.selected_provider = "groq"

    selected = st.selectbox(
        "Select LLM Provider",
        options=list(provider_options.keys()),
        format_func=lambda x: provider_options[x],
        index=list(provider_options.keys()).index(st.session_state.selected_provider),
        key="provider_select",
        label_visibility="collapsed",
    )
    st.session_state.selected_provider = selected


def _render_upload_section():
    """Render file upload section."""
    st.markdown("### 📁 Upload Files")

    # Format supported types for display
    st.caption(
        "Supports: PDF, DOCX, TXT, MD, CSV, MP3, WAV, MP4, AVI, PNG, JPG"
    )

    uploaded_files = st.file_uploader(
        "Drop files here",
        type=[ext.lstrip('.') for ext in ALL_SUPPORTED],
        accept_multiple_files=True,
        key="file_uploader",
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected**")

        for uf in uploaded_files:
            ext = get_file_extension(uf.name)
            icon = "📄"
            if ext in SUPPORTED_AUDIO:
                icon = "🎵"
            elif ext in SUPPORTED_VIDEO:
                icon = "🎬"
            elif ext in SUPPORTED_IMAGES:
                icon = "🖼️"
            
            st.caption(f"{icon} {uf.name} ({format_file_size(uf.size)})")

        # Process button
        if st.button("⚡ Process & Index", use_container_width=True, key="btn_process"):
            _process_uploaded_files(uploaded_files)


def _process_uploaded_files(uploaded_files):
    """Process and index uploaded files."""
    progress_bar = st.progress(0, text="Preparing files...")

    # Save files to temp
    file_infos = []
    for uf in uploaded_files:
        temp_path = save_uploaded_file(uf)
        file_infos.append({
            "path": temp_path,
            "name": uf.name,
            "type": get_file_extension(uf.name),
        })

    try:
        # ── Step 1: Ingest and chunk ──────────────────────
        progress_bar.progress(0.1, text="📝 Parsing & chunking files...")

        def update_progress(pct):
            progress_bar.progress(0.1 + pct * 0.4, text=f"📝 Processing... {int(pct*100)}%")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        chunks = loop.run_until_complete(
            ingest_files_parallel(
                files=file_infos,
                api_keys=st.session_state.get("api_keys", {}),
                progress_callback=update_progress,
            )
        )
        loop.close()

        if not chunks:
            st.warning("No content could be extracted from the uploaded files.")
            return

        # ── Step 2: Generate embeddings ───────────────────
        progress_bar.progress(0.55, text="🧬 Generating embeddings...")

        texts = [c.content for c in chunks]
        embeddings = generate_embeddings(texts, show_progress=False)

        # ── Step 3: Store in vector DB ────────────────────
        progress_bar.progress(0.8, text="💾 Storing in vector database...")

        collection = get_or_create_collection()
        added = add_chunks_to_store(chunks, embeddings, collection)

        # ── Step 4: Update BM25 index ─────────────────────
        progress_bar.progress(0.95, text="🔍 Building search index...")

        # Store chunks in session state for BM25
        if "all_chunks" not in st.session_state:
            st.session_state.all_chunks = []
        st.session_state.all_chunks.extend([c.to_dict() for c in chunks])

        progress_bar.progress(1.0, text="✅ Complete!")

        # Track processed files
        if "processed_files" not in st.session_state:
            st.session_state.processed_files = []
        for fi in file_infos:
            st.session_state.processed_files.append(fi["name"])

        st.success(f"✨ Indexed **{added}** chunks from **{len(uploaded_files)}** file(s)")

    except Exception as e:
        st.error(f"❌ Processing failed: {str(e)[:300]}")

    finally:
        # Cleanup temp files
        for fi in file_infos:
            cleanup_temp_file(fi["path"])


def _render_kb_stats():
    """Render knowledge base statistics."""
    st.markdown("### 📊 Knowledge Base")

    try:
        stats = get_collection_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chunks", stats["total_chunks"])
        with col2:
            st.metric("Sources", stats["num_sources"])

        if stats["modalities"]:
            modality_icons = {"text": "📄", "audio": "🎵", "video": "🎬", "image": "🖼️"}
            mod_display = " ".join(
                modality_icons.get(m, "📎") for m in stats["modalities"]
            )
            st.caption(f"Modalities: {mod_display}")

        if stats["sources"]:
            with st.expander("📂 Indexed Files", expanded=False):
                for src in stats["sources"]:
                    st.caption(f"• {src}")
    except Exception:
        st.caption("No data indexed yet.")


def _render_advanced_settings():
    """Render advanced retrieval settings."""
    with st.expander("⚙️ Advanced Settings", expanded=False):
        st.markdown("**Retrieval**")

        if "top_k" not in st.session_state:
            st.session_state.top_k = TOP_K_RERANK

        st.session_state.top_k = st.slider(
            "Top-K Results",
            min_value=1, max_value=20,
            value=st.session_state.top_k,
            key="slider_top_k",
        )

        st.markdown("**Search Weights**")
        dense_w = st.slider(
            "Dense (Semantic) Weight",
            min_value=0.0, max_value=1.0,
            value=DENSE_WEIGHT, step=0.1,
            key="slider_dense",
        )
        st.session_state.dense_weight = dense_w
        st.session_state.bm25_weight = 1.0 - dense_w

        st.markdown("**Re-ranking**")
        st.session_state.enable_reranking = st.checkbox(
            "Enable Cross-Encoder Re-ranking",
            value=True,
            key="cb_reranking",
        )

        st.markdown("**Modality Filter**")
        st.session_state.modality_filter = st.selectbox(
            "Filter by modality",
            options=["all", "text", "audio", "video", "image"],
            key="select_modality",
            label_visibility="collapsed",
        )


def _render_actions():
    """Render action buttons."""
    st.markdown("### 🛠️ Actions")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🗑️ Clear KB", use_container_width=True, key="btn_clear"):
            clear_collection()
            st.session_state.all_chunks = []
            st.session_state.processed_files = []
            if "messages" in st.session_state:
                st.session_state.messages = []
            st.success("Knowledge base cleared!")
            st.rerun()

    with col2:
        if st.button("🔄 New Chat", use_container_width=True, key="btn_new_chat"):
            st.session_state.messages = []
            st.rerun()

    # Download project ZIP
    st.markdown("---")
    st.caption("📦 Download this project")
    
    from utils.helpers import create_project_zip
    project_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(project_dir)  # Go up from ui/ to project root
    
    if st.button("📥 Generate Project ZIP", use_container_width=True, key="btn_zip"):
        try:
            zip_bytes = create_project_zip(project_dir)
            st.download_button(
                label="⬇️ Download ZIP",
                data=zip_bytes,
                file_name="ragnarok_project.zip",
                mime="application/zip",
                use_container_width=True,
                key="btn_download_zip",
            )
        except Exception as e:
            st.error(f"ZIP generation failed: {e}")
