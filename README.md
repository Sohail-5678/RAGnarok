<div align="center">
  <h1>🔮 RAGnarok</h1>
  <p><strong>Production-Grade Multimodal Retrieval-Augmented Generation Application</strong></p>
  <p>
    PDFs · Audio · Video · Images · Documents&nbsp;&nbsp;|&nbsp;&nbsp;
    Hybrid Search · Cross-Encoder Re-ranking · LLM-as-Judge Evaluation
  </p>
</div>

## 📌 Overview

RAGnarok is a state-of-the-art, multimodal AI application that ingests, processes, and queries complex data sources — **PDFs, DOCX, TXT, MD, CSV, Audio, Video, and Images**. Built with Streamlit and wrapped in a custom **Glassmorphism** UI, it offers dynamic multi-LLM routing (Groq, OpenAI, Gemini, Claude), hybrid search retrieval, and a full **offline + inline evaluation suite**.

The default deployment runs entirely on the **free Groq API** — vision, transcription, generation, and LLM-as-judge evaluation all use Groq models out of the box. No paid keys required.

## 🏗️ Architecture & Pipeline

### 1. Multimodal Ingestion

| Modality | Pipeline |
|---|---|
| **Documents** (PDF/DOCX/TXT/MD/CSV) | Parallel parsing → sentence-aware semantic chunking with token limits + overlap → per-page metadata |
| **Audio** (.mp3/.wav/.m4a/…) | Groq Whisper `whisper-large-v3-turbo` → timestamped segment chunks (auto-falls back to OpenAI Whisper if available) |
| **Video** (.mp4/.mov/.mkv/.webm/.avi) | OpenCV frame extraction (1 fps, smart-sampled to ≤24 frames) → Groq `meta-llama/llama-4-scout-17b-16e-instruct` vision per frame **+** Groq Whisper on the audio track → unified `[Visual]` + `[Audio]` chunks |
| **Images** (.png/.jpg/.gif/.webp) | Groq Llama-4 Scout vision (auto-falls back to Gemini 2.0 Flash or GPT-4o if keys present) |

> **Provider priority** is **Groq → Gemini → OpenAI** for vision, and **Groq → OpenAI** for transcription. Keys are resolved from session state first, then environment variables — so a single `GROQ_API_KEY` env var enables every modality.

### 2. Embeddings & Storage

- **Model**: `all-MiniLM-L6-v2` via `sentence-transformers` (384-dim, normalized, batched).
- **Vector DB**: Serverless `ChromaDB` (`hnsw:space=cosine`) with strict modality, source, page/timestamp metadata.

### 3. Smart Retrieval Layer

- **Hybrid Search**: Dense (ChromaDB) + Sparse (BM25 Okapi) fused via **Reciprocal Rank Fusion (RRF)** with configurable weights.
- **Cross-Encoder Re-ranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2` re-scores top results for maximum precision (optional toggle).
- **Modality filtering** and per-config A/B switches exposed in the sidebar.

### 4. Generation Layer (Multi-LLM)

- **Dynamic Routing**: Groq Llama 3.3 70B (default) ↔ OpenAI GPT-4o ↔ Gemini 2.0 Flash ↔ Anthropic Claude Sonnet 4.
- **Strict Citation Prompting**: enforces `[Source: file.pdf, Page X]`, `[Source: vid.mp4, Frame at MM:SS]`, etc.
- **Token streaming** for ultra-low perceived latency.

### 5. Evaluation Suite ⭐ NEW

A first-class evaluation layer lives in `evaluation/`:

| Concern | Implementation |
|---|---|
| **Retrieval quality** | Hit@K, Recall@K, Precision@K, MRR, nDCG@K |
| **Reference-based generation** | BLEU, ROUGE-1/2/L, optional BERTScore |
| **Reference-free generation** | Citation coverage, fake-citation rate, **LLM-as-judge** faithfulness & answer relevancy (via Groq) |
| **Optional frameworks** | RAGAS (Groq-wired), DeepEval, TruLens — all lazy-imported |
| **Persistent logging** | SQLite + JSONL of every live query (opt-in toggle) |
| **A/B harness + CLI** | `python -m evaluation.cli --ab rerank_on rerank_off` |
| **Inline per-answer scoring** | "📈 Score this answer" button under every chat reply — no gold set needed |
| **Unit tests** | `tests/` — 65 pytest cases, ~3s, zero network |

## 🎨 The "Glass UI"

Fully custom CSS theme — translucent backgrounds with `backdrop-filter: blur`, animated gradients, subtle glowing hover states, smooth transitions, responsive sidebar with secure API-key management.

## 🚀 Deployment (Streamlit Community Cloud)

This project is tailored for **one-click Streamlit Cloud deployment** with the free Groq API.

1. **Fork/Clone** this repository to your GitHub account.
2. Visit [share.streamlit.io](https://share.streamlit.io) and log in.
3. Click **"New App"** and select the repository.
4. Set the **Main file path** to `app.py`.
5. Under **Advanced Settings → Secrets**, add (TOML):
   ```toml
   GROQ_API_KEY = "gsk_your_groq_key_here"
   ```
   Optional, only if you want paid providers active:
   ```toml
   OPENAI_API_KEY    = "sk-..."
   GOOGLE_API_KEY    = "..."
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
6. Click **Deploy**. Streamlit will:
   - install system packages from `packages.txt` (gives you `ffmpeg` + `ffprobe`),
   - install Python deps from `requirements.txt`,
   - bootstrap the bundled `imageio-ffmpeg` binary as a safety net if anything is missing,
   - launch `app.py`.

> ✅ With just `GROQ_API_KEY`, every modality (PDF, audio, video, image) and the LLM-as-judge evaluator are fully functional. No paid keys required.

## 💻 Local Installation

```bash
# Clone
git clone https://github.com/Sohail-5678/RAGnarok.git
cd RAGnarok

# Virtual env
python3 -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

# Install
pip install -r requirements.txt

# (Optional) install system ffmpeg for fastest audio/video processing
brew install ffmpeg          # macOS — Linux users: sudo apt install ffmpeg

# Provide a Groq key (free tier is enough)
export GROQ_API_KEY="gsk_..."

# Run
streamlit run app.py
```

If you skip the `brew install ffmpeg` step, RAGnarok automatically falls back to the static `ffmpeg` binary bundled by the `imageio-ffmpeg` wheel — no manual setup needed.

## 🧪 Running the Test Suite

```bash
pytest tests/ -v
```

Covers chunking, RRF math, BM25, key/provider resolution, evaluation metrics, and persistent logging. **65 tests, no API keys, no network.**

## 📊 Using the Evaluation Suite

### Option 1 — Inline scoring (zero setup)

After every assistant answer in the live chat, click **📈 Score this answer**. RAGnarok runs:

- **Faithfulness** — LLM-as-judge: is every claim supported by the retrieved chunks?
- **Answer Relevancy** — LLM-as-judge: does the answer address the question?
- **Citation coverage** — deterministic: are cited sources real?

Results render as colored badges right under the message, with a judge-reasoning expander for transparency.

### Option 2 — Universal probe set (works on any upload, default in sidebar)

`docs/examples/universal_probe_set.jsonl` contains **12 generic probe questions** that exercise the pipeline against any document. Because it has no `expected_*` fields, **only the reference-free metrics are meaningful** (faithfulness, answer relevancy, citation coverage). Retrieval metrics will be 0 — that's intentional.

```bash
python -m evaluation.cli --dataset docs/examples/universal_probe_set.jsonl
```

Use this as a quick health-check after uploading any PDF/audio/video.

### Option 3 — Corpus-specific gold set (most accurate)

For real accuracy numbers tailored to *your* documents, write a small JSONL gold set:

```json
{"id": "q1", "query": "...", "expected_sources": ["file.pdf"],
 "expected_substrings": ["key phrase"], "gold_answer": "..."}
```

Then:

```bash
python -m evaluation.cli --dataset path/to/your_gold.jsonl
python -m evaluation.cli --dataset path/to/your_gold.jsonl --ab rerank_on rerank_off
```

Reports land in `evaluation/reports/` as JSONL + JSON + Markdown. See `evaluation/README.md` for the full schema.

### Option 4 — Persistent live logging

Enable the **"Persist queries to SQLite + JSONL"** toggle in the sidebar's 📊 Evaluation & Logging expander. Every chat turn is logged to `evaluation/logs/ragnarok_queries.db` + `…queries.jsonl` for offline analysis (these paths are git-ignored).

See **`evaluation/README.md`** for full documentation of metrics, dataset schema, and programmatic API.

## 📁 Project Structure

```
RAGnarok/
├── app.py                       # Streamlit entrypoint
├── config.py                    # Models, weights, prompts, providers
├── packages.txt                 # apt packages for Streamlit Cloud (ffmpeg)
├── requirements.txt             # Python deps
├── core/
│   ├── ingestion.py             # Parallel multimodal ingestion + key resolution
│   ├── embeddings.py            # all-MiniLM-L6-v2 wrapper
│   ├── vector_store.py          # ChromaDB persistent client
│   ├── retrieval.py             # Hybrid search + RRF + cross-encoder rerank
│   └── generation.py            # Streaming Groq / OpenAI / Gemini / Claude
├── utils/
│   ├── __init__.py              # ffmpeg bootstrap (system → imageio-ffmpeg)
│   ├── chunking.py              # Semantic / audio / video chunkers
│   ├── audio_processor.py       # Whisper transcription
│   ├── video_processor.py       # OpenCV frames + Groq vision + Whisper
│   └── helpers.py               # File/IO/context utilities
├── ui/
│   ├── styles.py                # Glassmorphism CSS
│   ├── sidebar.py               # API keys, uploads, settings, evaluation
│   └── chat.py                  # Streaming chat + inline "Score this answer"
├── evaluation/
│   ├── dataset.py               # JSONL gold-set loader
│   ├── retrieval_metrics.py     # Hit@K, Recall@K, MRR, nDCG@K, …
│   ├── generation_metrics.py    # BLEU, ROUGE, citation coverage, LLM judges
│   ├── ragas_eval.py            # Optional RAGAS / DeepEval / TruLens bridges
│   ├── logger.py                # SQLite + JSONL persistent query logger
│   ├── harness.py               # End-to-end + A/B harness
│   ├── cli.py                   # python -m evaluation.cli
│   └── README.md                # Evaluation usage docs
├── tests/
│   ├── conftest.py              # Pytest fixtures
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   ├── test_ingestion.py
│   ├── test_generation.py
│   └── test_evaluation.py
└── docs/
    └── examples/
        ├── universal_probe_set.jsonl  # 12 generic questions, works on any PDF
        └── sample_dataset.jsonl       # RAGnarok-specific gold set (schema demo)
```

## 🔐 Security & Privacy

- API keys entered in the sidebar live only in `st.session_state` (never written to disk).
- Default keys are read from environment variables / Streamlit secrets (never committed).
- Persistent query logs are **opt-in** and git-ignored (`evaluation/logs/`).
- ChromaDB persistence on Streamlit Cloud is **ephemeral** — the knowledge base resets on container restart by design.

## 📜 License

MIT. See `LICENSE` (or add one) for details.

## 🙏 Acknowledgements

Built by **Ameer Sohail**. Powered by Groq, Streamlit, ChromaDB, sentence-transformers, and the open-source RAG community.
