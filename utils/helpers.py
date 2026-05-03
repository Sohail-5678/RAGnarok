"""
Utility helper functions for the Multimodal RAG application.
"""
import os
import io
import zipfile
import tempfile
from typing import List, Dict, Any, Optional
from pathlib import Path


def get_file_extension(filename: str) -> str:
    """Get lowercase file extension."""
    return Path(filename).suffix.lower()


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def save_uploaded_file(uploaded_file) -> str:
    """Save a Streamlit uploaded file to a temporary location."""
    suffix = Path(uploaded_file.name).suffix
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(uploaded_file.getbuffer())
    temp_file.close()
    return temp_file.name


def cleanup_temp_file(file_path: str):
    """Remove a temporary file safely."""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except Exception:
        pass


def create_project_zip(project_dir: str) -> bytes:
    """
    Create a ZIP file of the entire project directory.
    Returns the ZIP file as bytes.
    """
    buffer = io.BytesIO()
    
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(project_dir):
            # Skip common non-essential directories
            dirs[:] = [d for d in dirs if d not in {
                '__pycache__', '.git', 'venv', 'env', '.venv',
                'node_modules', '.pytest_cache', '.mypy_cache',
            }]
            
            for file in files:
                if file.endswith(('.pyc', '.pyo', '.DS_Store')):
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, project_dir)
                zf.write(file_path, arcname)
    
    buffer.seek(0)
    return buffer.getvalue()


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to max length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def format_context_for_llm(chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a structured context string for the LLM.
    """
    if not chunks:
        return "No relevant context found."

    context_parts: List[str] = []

    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", "Unknown")
        modality = metadata.get("modality", "text")
        content = chunk.get("content", "")

        # Build source citation
        citation_parts = [f"Source: {source}"]
        
        if modality == "text":
            page = metadata.get("page_number")
            if page:
                citation_parts.append(f"Page {page}")
        elif modality == "audio":
            timestamp = metadata.get("timestamp")
            if timestamp:
                citation_parts.append(f"Timestamp: {timestamp}")
        elif modality == "video":
            timestamp = metadata.get("timestamp")
            if timestamp:
                citation_parts.append(f"Frames: {timestamp}")
        elif modality == "image":
            citation_parts.append("Image Description")

        citation = " | ".join(citation_parts)
        modality_icon = {"text": "📄", "audio": "🎵", "video": "🎬", "image": "🖼️"}.get(modality, "📎")

        context_parts.append(
            f"--- Chunk {i} [{modality_icon} {modality.upper()}] ---\n"
            f"[{citation}]\n"
            f"{content}\n"
        )

    return "\n".join(context_parts)


def validate_api_key(key: str, min_length: int = 10) -> bool:
    """Basic validation for API key format."""
    if not key or not isinstance(key, str):
        return False
    key = key.strip()
    return len(key) >= min_length
