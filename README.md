<div align="center">
  <h1>🔮 RAGnarok</h1>
  <p><strong>Production-Grade Multimodal Retrieval-Augmented Generation Application</strong></p>
</div>

## 📌 Overview

RAGnarok is a state-of-the-art, multimodal AI application designed to ingest, process, and query complex data sources—including **PDFs, Documents, Audio, Video, and Images**. Built with Streamlit and wrapped in a stunning custom "Glassmorphism" UI, it offers dynamic multi-LLM routing (Groq, OpenAI, Gemini, Claude) and advanced hybrid search retrieval.

## 🏗️ Architecture & Pipeline

### 1. Multimodal Ingestion
- **Text & Documents**: Parallel parsing and semantic-aware chunking (sentence boundaries, token limits) for PDFs, DOCX, TXT, MD, and CSVs.
- **Audio**: Automatic transcription via Whisper API (Groq/OpenAI), chunked by timestamp metadata.
- **Video**: Frame extraction (1 FPS) via OpenCV combined with vision model frame descriptions (Gemini/GPT-4o) to form unified temporal chunks.
- **Images**: Rich contextual descriptions generated via Vision APIs.

### 2. Embeddings & Storage
- **Model**: Fast, local embedding generation using `all-MiniLM-L6-v2` via `sentence-transformers` with batch processing.
- **Vector Database**: Serverless `ChromaDB` storing semantic vectors alongside strict modality, source, and timestamp metadata.

### 3. Smart Retrieval Layer
- **Hybrid Search**: Fuses Dense Vector Search (ChromaDB) with Sparse Keyword Search (BM25 Okapi) using Reciprocal Rank Fusion (RRF).
- **Cross-Encoder Re-ranking**: Passes top-K results through `ms-marco-MiniLM-L-6-v2` to mathematically re-score relevance, ensuring maximum precision.

### 4. Generation Layer (Multi-LLM)
- **Dynamic Routing**: Switch seamlessly between Groq (Llama 3), OpenAI (GPT-4o), Google (Gemini 2.0), and Anthropic (Claude 3.5).
- **Strict Prompting**: Enforces citation generation (e.g., *[Source: video.mp4, Frame at 02:15]*) to eliminate hallucinations. Async streaming capabilities ensure ultra-low perceived latency.

## 🎨 The "Glass UI"
Designed to break the mold of standard Streamlit apps, RAGnarok features a fully custom CSS theme:
- Translucent backgrounds with `backdrop-filter: blur`.
- Animated gradients, subtle glowing hover states, and smooth transitions.
- Fully responsive sidebar for secure API Key management and configuration.

## 🚀 Deployment Instructions (Streamlit Community Cloud)

This project is perfectly tailored for deployment on Streamlit Cloud.

1. **Fork/Clone** this repository to your GitHub account.
2. Visit [share.streamlit.io](https://share.streamlit.io) and log in.
3. Click **"New App"** and select the repository.
4. Set the Main file path to `app.py`.
5. **Add Environment Variables**: Under "Advanced Settings", add:
   ```env
   GROQ_API_KEY=your_groq_key_here
   ```
6. Click **Deploy**. Streamlit will automatically install dependencies from `requirements.txt`.

## 💻 Local Installation

```bash
# Clone the project
git clone https://github.com/yourusername/ragnarok.git
cd ragnarok

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```
