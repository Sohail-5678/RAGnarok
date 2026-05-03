"""
Configuration constants and environment settings for the Multimodal RAG application.
"""
import os
from dataclasses import dataclass, field
from typing import Optional


# ─── Chunking Defaults ────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE = 600        # tokens
DEFAULT_CHUNK_OVERLAP = 60      # ~10% overlap
MAX_CHUNK_SIZE = 800
MIN_CHUNK_SIZE = 100

# ─── Video Processing ─────────────────────────────────────────────────
FRAMES_PER_SECOND = 1           # Extract 1 frame per second
MAX_VIDEO_DURATION = 600        # 10 minutes max
FRAME_RESIZE = (640, 480)       # Resize frames for processing

# ─── Audio Processing ─────────────────────────────────────────────────
MAX_AUDIO_DURATION = 1800       # 30 minutes max
AUDIO_CHUNK_DURATION = 30       # Seconds per audio chunk for transcription
SUPPORTED_AUDIO_FORMATS = [".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"]

# ─── Retrieval Settings ───────────────────────────────────────────────
TOP_K_RETRIEVAL = 20            # Initial retrieval count
TOP_K_RERANK = 5                # After re-ranking
BM25_WEIGHT = 0.3               # Sparse search weight
DENSE_WEIGHT = 0.7              # Dense search weight
SIMILARITY_THRESHOLD = 0.25     # Minimum similarity score

# ─── Embedding Settings ───────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBEDDING_BATCH_SIZE = 64

# ─── Reranker Settings ────────────────────────────────────────────────
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ─── Supported File Types ─────────────────────────────────────────────
SUPPORTED_DOCUMENTS = [".pdf", ".docx", ".txt", ".md", ".csv"]
SUPPORTED_AUDIO = [".mp3", ".wav", ".m4a", ".ogg", ".flac"]
SUPPORTED_VIDEO = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
SUPPORTED_IMAGES = [".png", ".jpg", ".jpeg", ".gif", ".webp"]

ALL_SUPPORTED = SUPPORTED_DOCUMENTS + SUPPORTED_AUDIO + SUPPORTED_VIDEO + SUPPORTED_IMAGES


@dataclass
class LLMConfig:
    """Configuration for a specific LLM provider."""
    provider: str
    model_name: str
    api_key: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 4096
    base_url: Optional[str] = None


# ─── LLM Provider Configs ─────────────────────────────────────────────
LLM_PROVIDERS = {
    "groq": {
        "name": "Groq (Llama 3.3 70B)",
        "model": "llama-3.3-70b-versatile",
        "env_key": "GROQ_API_KEY",
        "icon": "⚡",
    },
    "openai": {
        "name": "OpenAI (GPT-4o)",
        "model": "gpt-4o",
        "env_key": "OPENAI_API_KEY",
        "icon": "🧠",
    },
    "gemini": {
        "name": "Google Gemini (2.0 Flash)",
        "model": "gemini-2.0-flash",
        "env_key": "GOOGLE_API_KEY",
        "icon": "✨",
    },
    "claude": {
        "name": "Anthropic Claude (Sonnet 4)",
        "model": "claude-sonnet-4-20250514",
        "env_key": "ANTHROPIC_API_KEY",
        "icon": "🎭",
    },
}

# ─── System Prompt ─────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a precise, knowledgeable AI assistant powered by a Multimodal Retrieval-Augmented Generation (RAG) system. Your responses MUST follow these strict rules:

1. **Ground All Answers in Retrieved Context**: Only use information from the provided context chunks. If the context doesn't contain relevant information, explicitly state: "The uploaded documents do not contain information about this topic."

2. **Cite Sources Precisely**: For every claim, cite the exact source using this format:
   - For documents: [Source: filename.pdf, Page X]
   - For audio: [Source: filename.mp3, Timestamp MM:SS]
   - For video: [Source: filename.mp4, Frame at MM:SS]
   - For images: [Source: filename.jpg, Description]

3. **Handle Multimodal Context**: When context includes image descriptions or audio transcripts, integrate them naturally into your response while maintaining citations.

4. **Never Hallucinate**: Do not invent information, statistics, or claims not present in the context. If uncertain, say so.

5. **Be Concise and Structured**: Use bullet points, headers, and formatting to make responses scannable and clear.

6. **Acknowledge Limitations**: If the retrieved context is insufficient or ambiguous, explicitly note the limitation rather than guessing.

RETRIEVED CONTEXT:
{context}

USER QUESTION: {question}
"""

NO_CONTEXT_PROMPT = """You are a helpful AI assistant for a Multimodal RAG application. No documents have been uploaded yet.

Respond helpfully to the user's message. If they ask about document content, remind them to upload files first using the sidebar.

You can help with:
- Explaining how the RAG system works
- Guiding them on what types of files to upload (PDFs, audio, video, images)
- General knowledge questions (but note you won't have RAG context)

USER MESSAGE: {question}
"""
