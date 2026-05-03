"""
Semantic chunking utilities for text, audio transcripts, and video descriptions.
"""
import re
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """Represents a single chunk of content with metadata."""
    content: str
    chunk_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "chunk_id": self.chunk_id,
            "metadata": self.metadata,
        }


def generate_chunk_id(content: str, source: str, index: int) -> str:
    """Generate a deterministic chunk ID."""
    hash_input = f"{source}:{index}:{content[:100]}"
    return hashlib.md5(hash_input.encode()).hexdigest()[:12]


def estimate_tokens(text: str) -> int:
    """Rough token estimation (~4 chars per token for English)."""
    return len(text) // 4


def clean_text(text: str) -> str:
    """Clean and normalize extracted text."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    # Remove page numbers patterns
    text = re.sub(r'\b(Page|page)\s*\d+\s*(of\s*\d+)?\b', '', text)
    return text.strip()


def semantic_chunk_text(
    text: str,
    source_file: str,
    chunk_size: int = 600,
    chunk_overlap: int = 60,
    modality: str = "text",
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> List[Chunk]:
    """
    Perform semantic-aware chunking on text content.
    
    Splits on sentence boundaries while respecting token limits.
    Applies overlap for context continuity.
    """
    if not text or not text.strip():
        return []

    text = clean_text(text)
    
    # Split into sentences using regex
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return []

    chunks: List[Chunk] = []
    current_chunk_sentences: List[str] = []
    current_tokens = 0
    chunk_index = 0

    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        
        # If single sentence exceeds chunk_size, split it further
        if sentence_tokens > chunk_size:
            # Flush current chunk first
            if current_chunk_sentences:
                chunk_text = ' '.join(current_chunk_sentences)
                chunk_id = generate_chunk_id(chunk_text, source_file, chunk_index)
                metadata = {
                    "source": source_file,
                    "modality": modality,
                    "chunk_index": chunk_index,
                    "token_estimate": estimate_tokens(chunk_text),
                }
                if extra_metadata:
                    metadata.update(extra_metadata)
                chunks.append(Chunk(content=chunk_text, chunk_id=chunk_id, metadata=metadata))
                chunk_index += 1
                current_chunk_sentences = []
                current_tokens = 0

            # Split the long sentence by words
            words = sentence.split()
            word_chunk: List[str] = []
            word_tokens = 0
            for word in words:
                wt = estimate_tokens(word + ' ')
                if word_tokens + wt > chunk_size and word_chunk:
                    chunk_text = ' '.join(word_chunk)
                    chunk_id = generate_chunk_id(chunk_text, source_file, chunk_index)
                    metadata = {
                        "source": source_file,
                        "modality": modality,
                        "chunk_index": chunk_index,
                        "token_estimate": estimate_tokens(chunk_text),
                    }
                    if extra_metadata:
                        metadata.update(extra_metadata)
                    chunks.append(Chunk(content=chunk_text, chunk_id=chunk_id, metadata=metadata))
                    chunk_index += 1
                    # Overlap: keep last few words
                    overlap_words = max(1, len(word_chunk) // 10)
                    word_chunk = word_chunk[-overlap_words:]
                    word_tokens = sum(estimate_tokens(w + ' ') for w in word_chunk)
                word_chunk.append(word)
                word_tokens += wt
            if word_chunk:
                current_chunk_sentences = [' '.join(word_chunk)]
                current_tokens = word_tokens
            continue

        # Check if adding this sentence would exceed the limit
        if current_tokens + sentence_tokens > chunk_size and current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            chunk_id = generate_chunk_id(chunk_text, source_file, chunk_index)
            metadata = {
                "source": source_file,
                "modality": modality,
                "chunk_index": chunk_index,
                "token_estimate": estimate_tokens(chunk_text),
            }
            if extra_metadata:
                metadata.update(extra_metadata)
            chunks.append(Chunk(content=chunk_text, chunk_id=chunk_id, metadata=metadata))
            chunk_index += 1

            # Apply overlap: keep last N tokens worth of sentences
            overlap_sentences: List[str] = []
            overlap_tokens = 0
            for s in reversed(current_chunk_sentences):
                st = estimate_tokens(s)
                if overlap_tokens + st <= chunk_overlap:
                    overlap_sentences.insert(0, s)
                    overlap_tokens += st
                else:
                    break
            current_chunk_sentences = overlap_sentences
            current_tokens = overlap_tokens

        current_chunk_sentences.append(sentence)
        current_tokens += sentence_tokens

    # Flush remaining content
    if current_chunk_sentences:
        chunk_text = ' '.join(current_chunk_sentences)
        chunk_id = generate_chunk_id(chunk_text, source_file, chunk_index)
        metadata = {
            "source": source_file,
            "modality": modality,
            "chunk_index": chunk_index,
            "token_estimate": estimate_tokens(chunk_text),
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        chunks.append(Chunk(content=chunk_text, chunk_id=chunk_id, metadata=metadata))

    return chunks


def chunk_audio_transcript(
    transcript_segments: List[Dict[str, Any]],
    source_file: str,
    chunk_duration: float = 30.0,
) -> List[Chunk]:
    """
    Chunk audio transcript segments with timestamp metadata.
    
    Each segment should have: {'text': str, 'start': float, 'end': float}
    """
    if not transcript_segments:
        return []

    chunks: List[Chunk] = []
    current_text_parts: List[str] = []
    current_start = transcript_segments[0].get("start", 0.0)
    current_end = 0.0
    chunk_index = 0

    for segment in transcript_segments:
        seg_text = segment.get("text", "").strip()
        seg_start = segment.get("start", 0.0)
        seg_end = segment.get("end", 0.0)

        if not seg_text:
            continue

        # Check if adding this segment exceeds the duration threshold
        if seg_end - current_start > chunk_duration and current_text_parts:
            chunk_text = ' '.join(current_text_parts)
            chunk_text = clean_text(chunk_text)
            chunk_id = generate_chunk_id(chunk_text, source_file, chunk_index)

            start_ts = format_timestamp(current_start)
            end_ts = format_timestamp(current_end)

            chunks.append(Chunk(
                content=chunk_text,
                chunk_id=chunk_id,
                metadata={
                    "source": source_file,
                    "modality": "audio",
                    "chunk_index": chunk_index,
                    "start_time": current_start,
                    "end_time": current_end,
                    "timestamp": f"{start_ts} - {end_ts}",
                    "token_estimate": estimate_tokens(chunk_text),
                },
            ))
            chunk_index += 1
            current_text_parts = []
            current_start = seg_start

        current_text_parts.append(seg_text)
        current_end = seg_end

    # Flush remaining
    if current_text_parts:
        chunk_text = ' '.join(current_text_parts)
        chunk_text = clean_text(chunk_text)
        chunk_id = generate_chunk_id(chunk_text, source_file, chunk_index)
        start_ts = format_timestamp(current_start)
        end_ts = format_timestamp(current_end)

        chunks.append(Chunk(
            content=chunk_text,
            chunk_id=chunk_id,
            metadata={
                "source": source_file,
                "modality": "audio",
                "chunk_index": chunk_index,
                "start_time": current_start,
                "end_time": current_end,
                "timestamp": f"{start_ts} - {end_ts}",
                "token_estimate": estimate_tokens(chunk_text),
            },
        ))

    return chunks


def chunk_video_descriptions(
    frame_descriptions: List[Dict[str, Any]],
    audio_transcript: str,
    source_file: str,
    chunk_duration: float = 10.0,
) -> List[Chunk]:
    """
    Create unified multimodal chunks from video frame descriptions + audio transcript.
    
    Each frame_description should have: {'timestamp': float, 'description': str, 'frame_index': int}
    """
    if not frame_descriptions:
        return []

    chunks: List[Chunk] = []
    current_descriptions: List[str] = []
    current_start = frame_descriptions[0].get("timestamp", 0.0)
    chunk_index = 0

    for frame in frame_descriptions:
        ts = frame.get("timestamp", 0.0)
        desc = frame.get("description", "").strip()

        if not desc:
            continue

        if ts - current_start > chunk_duration and current_descriptions:
            # Build multimodal chunk content
            visual_content = "\n".join(current_descriptions)
            chunk_text = f"[Visual Content]\n{visual_content}"
            
            # Add relevant audio transcript portion if available
            if audio_transcript:
                chunk_text += f"\n\n[Audio Context]\n{audio_transcript[:500]}"

            chunk_text = clean_text(chunk_text)
            chunk_id = generate_chunk_id(chunk_text, source_file, chunk_index)

            chunks.append(Chunk(
                content=chunk_text,
                chunk_id=chunk_id,
                metadata={
                    "source": source_file,
                    "modality": "video",
                    "chunk_index": chunk_index,
                    "start_time": current_start,
                    "end_time": ts,
                    "timestamp": f"{format_timestamp(current_start)} - {format_timestamp(ts)}",
                    "frame_count": len(current_descriptions),
                    "token_estimate": estimate_tokens(chunk_text),
                },
            ))
            chunk_index += 1
            current_descriptions = []
            current_start = ts

        current_descriptions.append(f"[{format_timestamp(ts)}] {desc}")

    # Flush remaining
    if current_descriptions:
        visual_content = "\n".join(current_descriptions)
        chunk_text = f"[Visual Content]\n{visual_content}"
        if audio_transcript:
            chunk_text += f"\n\n[Audio Context]\n{audio_transcript[:500]}"

        chunk_text = clean_text(chunk_text)
        chunk_id = generate_chunk_id(chunk_text, source_file, chunk_index)
        end_ts = frame_descriptions[-1].get("timestamp", 0.0)

        chunks.append(Chunk(
            content=chunk_text,
            chunk_id=chunk_id,
            metadata={
                "source": source_file,
                "modality": "video",
                "chunk_index": chunk_index,
                "start_time": current_start,
                "end_time": end_ts,
                "timestamp": f"{format_timestamp(current_start)} - {format_timestamp(end_ts)}",
                "frame_count": len(current_descriptions),
                "token_estimate": estimate_tokens(chunk_text),
            },
        ))

    return chunks


def format_timestamp(seconds: float) -> str:
    """Format seconds into MM:SS or HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
