"""
Audio processing utilities: transcription via Whisper API (Groq/OpenAI).
"""
import os
import tempfile
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from utils.chunking import chunk_audio_transcript, Chunk


def get_audio_duration(file_path: str) -> float:
    """Get audio duration using pydub."""
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(file_path)
        return len(audio) / 1000.0  # milliseconds to seconds
    except Exception:
        return 0.0


def convert_to_wav(file_path: str) -> str:
    """Convert audio file to WAV format for compatibility."""
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(file_path)
        wav_path = tempfile.mktemp(suffix=".wav")
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        raise RuntimeError(f"Audio conversion failed: {e}")


async def transcribe_audio_groq(
    file_path: str,
    api_key: str,
) -> List[Dict[str, Any]]:
    """
    Transcribe audio using Groq's Whisper API.
    Returns list of segments with text, start, and end timestamps.
    """
    from groq import Groq

    client = Groq(api_key=api_key)
    
    file_size = os.path.getsize(file_path)
    max_size = 25 * 1024 * 1024  # 25MB limit

    segments = []

    if file_size <= max_size:
        # Single file transcription
        with open(file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(Path(file_path).name, audio_file.read()),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
        
        if hasattr(transcription, 'segments') and transcription.segments:
            for seg in transcription.segments:
                segments.append({
                    "text": seg.get("text", seg.text if hasattr(seg, 'text') else ""),
                    "start": seg.get("start", seg.start if hasattr(seg, 'start') else 0),
                    "end": seg.get("end", seg.end if hasattr(seg, 'end') else 0),
                })
        elif hasattr(transcription, 'text') and transcription.text:
            segments.append({
                "text": transcription.text,
                "start": 0.0,
                "end": get_audio_duration(file_path),
            })
    else:
        # Split large files into chunks
        from pydub import AudioSegment
        audio = AudioSegment.from_file(file_path)
        chunk_ms = 20 * 60 * 1000  # 20 minute chunks
        offset = 0.0

        for i in range(0, len(audio), chunk_ms):
            chunk = audio[i:i + chunk_ms]
            chunk_path = tempfile.mktemp(suffix=".wav")
            chunk.export(chunk_path, format="wav")

            try:
                with open(chunk_path, "rb") as audio_file:
                    transcription = client.audio.transcriptions.create(
                        file=(f"chunk_{i}.wav", audio_file.read()),
                        model="whisper-large-v3-turbo",
                        response_format="verbose_json",
                        timestamp_granularities=["segment"],
                    )

                if hasattr(transcription, 'segments') and transcription.segments:
                    for seg in transcription.segments:
                        segments.append({
                            "text": seg.get("text", seg.text if hasattr(seg, 'text') else ""),
                            "start": (seg.get("start", seg.start if hasattr(seg, 'start') else 0)) + offset,
                            "end": (seg.get("end", seg.end if hasattr(seg, 'end') else 0)) + offset,
                        })
                elif hasattr(transcription, 'text') and transcription.text:
                    chunk_duration = len(chunk) / 1000.0
                    segments.append({
                        "text": transcription.text,
                        "start": offset,
                        "end": offset + chunk_duration,
                    })
            finally:
                os.unlink(chunk_path)

            offset += chunk_ms / 1000.0

    return segments


async def transcribe_audio_openai(
    file_path: str,
    api_key: str,
) -> List[Dict[str, Any]]:
    """
    Transcribe audio using OpenAI's Whisper API.
    Returns list of segments with text, start, and end timestamps.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    segments = []

    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    if hasattr(transcription, 'segments') and transcription.segments:
        for seg in transcription.segments:
            segments.append({
                "text": seg.get("text", ""),
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
            })
    elif hasattr(transcription, 'text'):
        segments.append({
            "text": transcription.text,
            "start": 0.0,
            "end": get_audio_duration(file_path),
        })

    return segments


async def process_audio_file(
    file_path: str,
    source_name: str,
    api_key: str,
    provider: str = "groq",
    chunk_duration: float = 30.0,
) -> Tuple[List[Chunk], str]:
    """
    Full audio processing pipeline: transcribe → chunk.
    
    Returns:
        Tuple of (chunks, full_transcript_text)
    """
    # Transcribe
    if provider == "openai":
        segments = await transcribe_audio_openai(file_path, api_key)
    else:
        segments = await transcribe_audio_groq(file_path, api_key)

    if not segments:
        return [], ""

    # Get full transcript
    full_transcript = ' '.join(seg["text"] for seg in segments)

    # Chunk the transcript
    chunks = chunk_audio_transcript(
        transcript_segments=segments,
        source_file=source_name,
        chunk_duration=chunk_duration,
    )

    return chunks, full_transcript
