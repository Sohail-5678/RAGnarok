"""
Document ingestion: parallel parsing of PDFs, DOCX, TXT, MD, CSV, and images.
"""
import os
import asyncio
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from config import (
    SUPPORTED_DOCUMENTS, SUPPORTED_AUDIO, SUPPORTED_VIDEO, SUPPORTED_IMAGES,
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP,
)
from utils.chunking import semantic_chunk_text, Chunk, clean_text
from utils.helpers import get_file_extension, save_uploaded_file, cleanup_temp_file


# ─── Text Extraction Functions ────────────────────────────────────────

def extract_text_from_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Extract text from PDF with page-level metadata."""
    import PyPDF2
    
    pages: List[Dict[str, Any]] = []
    
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append({
                        "text": text,
                        "page_number": page_num,
                        "total_pages": len(reader.pages),
                    })
    except Exception as e:
        raise RuntimeError(f"PDF extraction failed: {e}")
    
    return pages


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX files."""
    import docx
    
    try:
        doc = docx.Document(file_path)
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        return '\n\n'.join(paragraphs)
    except Exception as e:
        raise RuntimeError(f"DOCX extraction failed: {e}")


def extract_text_from_txt(file_path: str) -> str:
    """Extract text from plain text files."""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    raise RuntimeError("Could not decode text file with any supported encoding")


def extract_text_from_csv(file_path: str) -> str:
    """Extract and format text from CSV files."""
    import csv
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        if not rows:
            return ""
        
        # Format as readable table
        header = rows[0] if rows else []
        text_parts = [f"CSV Data with columns: {', '.join(header)}"]
        
        for i, row in enumerate(rows[1:], 1):
            row_text = ' | '.join(f"{header[j] if j < len(header) else f'col{j}'}: {cell}" 
                                   for j, cell in enumerate(row) if cell.strip())
            if row_text:
                text_parts.append(f"Row {i}: {row_text}")
        
        return '\n'.join(text_parts)
    except Exception as e:
        raise RuntimeError(f"CSV extraction failed: {e}")


def describe_image_basic(file_path: str) -> str:
    """Generate a basic description of an image using metadata."""
    from PIL import Image
    
    try:
        img = Image.open(file_path)
        width, height = img.size
        mode = img.mode
        format_name = img.format or "Unknown"
        
        description = (
            f"Image file: {Path(file_path).name}. "
            f"Format: {format_name}, Size: {width}x{height}px, "
            f"Color mode: {mode}."
        )
        
        # Check for EXIF data
        exif_data = img.getexif()
        if exif_data:
            description += " Contains EXIF metadata."
        
        return description
    except Exception as e:
        return f"Image file: {Path(file_path).name}. [Unable to analyze: {str(e)[:100]}]"


async def describe_image_with_vision(
    file_path: str,
    api_key: str,
    provider: str = "gemini",
) -> str:
    """Describe an image using a vision model."""
    import base64
    
    if provider == "gemini" and api_key:
        try:
            import google.generativeai as genai
            from PIL import Image
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            
            img = Image.open(file_path)
            response = model.generate_content([
                "Describe this image in detail. Include: main subjects, colors, "
                "text visible, setting/background, and any notable details. "
                "Be factual and thorough.",
                img,
            ])
            return response.text.strip()
        except Exception as e:
            return describe_image_basic(file_path) + f" [Vision failed: {str(e)[:100]}]"
    
    elif provider == "openai" and api_key:
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=api_key)
            with open(file_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode('utf-8')
            
            ext = get_file_extension(file_path).lstrip('.')
            mime = f"image/{ext}" if ext != "jpg" else "image/jpeg"
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in detail."},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    ],
                }],
                max_tokens=500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return describe_image_basic(file_path) + f" [Vision failed: {str(e)[:100]}]"
    
    return describe_image_basic(file_path)


# ─── Main Ingestion Pipeline ──────────────────────────────────────────

async def ingest_document(
    file_path: str,
    source_name: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Chunk]:
    """
    Ingest a text document (PDF, DOCX, TXT, MD, CSV) and return chunks.
    """
    ext = get_file_extension(file_path)
    chunks: List[Chunk] = []

    if ext == ".pdf":
        pages = extract_text_from_pdf(file_path)
        for page_data in pages:
            page_chunks = semantic_chunk_text(
                text=page_data["text"],
                source_file=source_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                modality="text",
                extra_metadata={
                    "page_number": page_data["page_number"],
                    "total_pages": page_data["total_pages"],
                },
            )
            chunks.extend(page_chunks)

    elif ext == ".docx":
        text = extract_text_from_docx(file_path)
        chunks = semantic_chunk_text(
            text=text,
            source_file=source_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            modality="text",
        )

    elif ext in (".txt", ".md"):
        text = extract_text_from_txt(file_path)
        chunks = semantic_chunk_text(
            text=text,
            source_file=source_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            modality="text",
        )

    elif ext == ".csv":
        text = extract_text_from_csv(file_path)
        chunks = semantic_chunk_text(
            text=text,
            source_file=source_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            modality="text",
        )

    return chunks


async def ingest_image(
    file_path: str,
    source_name: str,
    vision_api_key: Optional[str] = None,
    vision_provider: str = "gemini",
) -> List[Chunk]:
    """Ingest an image file and return chunks with description."""
    if vision_api_key:
        description = await describe_image_with_vision(
            file_path, vision_api_key, vision_provider
        )
    else:
        description = describe_image_basic(file_path)

    from utils.chunking import Chunk, generate_chunk_id

    chunk_id = generate_chunk_id(description, source_name, 0)
    return [Chunk(
        content=description,
        chunk_id=chunk_id,
        metadata={
            "source": source_name,
            "modality": "image",
            "chunk_index": 0,
            "token_estimate": len(description) // 4,
        },
    )]


async def ingest_files_parallel(
    files: List[Dict[str, Any]],
    api_keys: Dict[str, str],
    progress_callback=None,
) -> List[Chunk]:
    """
    Parallel ingestion of multiple files.
    
    Each file dict should have: {'path': str, 'name': str, 'type': str}
    """
    all_chunks: List[Chunk] = []
    total = len(files)

    tasks = []

    for i, file_info in enumerate(files):
        path = file_info["path"]
        name = file_info["name"]
        ext = get_file_extension(name)

        if ext in SUPPORTED_DOCUMENTS:
            tasks.append(("document", ingest_document(path, name)))
        elif ext in SUPPORTED_IMAGES:
            vision_key = api_keys.get("gemini") or api_keys.get("openai")
            vision_provider = "gemini" if api_keys.get("gemini") else "openai"
            tasks.append(("image", ingest_image(
                path, name, vision_key, vision_provider
            )))
        elif ext in SUPPORTED_AUDIO:
            from utils.audio_processor import process_audio_file
            audio_key = api_keys.get("groq") or api_keys.get("openai")
            audio_provider = "groq" if api_keys.get("groq") else "openai"
            tasks.append(("audio", process_audio_file(
                path, name, audio_key, audio_provider
            )))
        elif ext in SUPPORTED_VIDEO:
            from utils.video_processor import process_video_file
            vision_key = api_keys.get("gemini") or api_keys.get("openai")
            vision_provider = "gemini" if api_keys.get("gemini") else "openai"
            # For video, we'd need audio transcript too, but process_video_file handles it
            tasks.append(("video", process_video_file(
                path, name, vision_provider, vision_key
            )))

    # Execute tasks with progress tracking
    for i, (task_type, coro) in enumerate(tasks):
        try:
            result = await coro
            if task_type in ("audio", "video"):
                chunks = result[0]  # Tuple returns (chunks, extra_data)
            else:
                chunks = result
            all_chunks.extend(chunks)
        except Exception as e:
            # Log error but continue with other files
            error_chunk = Chunk(
                content=f"[Error processing file: {str(e)[:200]}]",
                chunk_id=f"error_{i}",
                metadata={"source": files[i]["name"], "modality": "error", "error": str(e)},
            )
            all_chunks.append(error_chunk)

        if progress_callback:
            progress_callback((i + 1) / total)

    return all_chunks
