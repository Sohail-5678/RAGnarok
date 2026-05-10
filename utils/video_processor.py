"""
Video processing: frame extraction with OpenCV + vision model descriptions.
"""
import os
import cv2
import base64
import tempfile
import asyncio
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

from config import (
    FRAMES_PER_SECOND, MAX_VIDEO_DURATION, FRAME_RESIZE,
    GROQ_VISION_MODEL, GEMINI_VISION_MODEL, OPENAI_VISION_MODEL,
    MAX_VISION_FRAMES_PER_VIDEO,
)
from utils.chunking import chunk_video_descriptions, Chunk


def extract_frames(
    video_path: str,
    fps: int = FRAMES_PER_SECOND,
    max_duration: int = MAX_VIDEO_DURATION,
    resize: tuple = FRAME_RESIZE,
) -> List[Dict[str, Any]]:
    """
    Extract frames from video at specified FPS rate.
    
    Returns list of: {'frame': np.ndarray, 'timestamp': float, 'frame_index': int}
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    # Cap duration
    effective_duration = min(duration, max_duration)
    frame_interval = int(video_fps / fps) if fps > 0 else int(video_fps)

    frames: List[Dict[str, Any]] = []
    frame_count = 0
    extracted_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_count / video_fps if video_fps > 0 else 0
        if timestamp > effective_duration:
            break

        if frame_count % frame_interval == 0:
            # Resize frame
            if resize:
                frame = cv2.resize(frame, resize)

            frames.append({
                "frame": frame,
                "timestamp": timestamp,
                "frame_index": extracted_index,
            })
            extracted_index += 1

        frame_count += 1

    cap.release()
    return frames


def frame_to_base64(frame: np.ndarray) -> str:
    """Convert a frame (numpy array) to base64-encoded JPEG."""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buffer).decode('utf-8')


def get_video_metadata(video_path: str) -> Dict[str, Any]:
    """Extract video metadata."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    metadata = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        if cap.get(cv2.CAP_PROP_FPS) > 0 else 0,
    }
    cap.release()
    return metadata


def smart_sample_frames(
    frames: List[Dict[str, Any]],
    max_frames: int = MAX_VISION_FRAMES_PER_VIDEO,
) -> List[Dict[str, Any]]:
    """
    Uniformly down-sample a list of frames to at most ``max_frames``.
    Preserves chronological order and original timestamps so the
    resulting descriptions still span the full video timeline.
    """
    if not frames or len(frames) <= max_frames:
        return frames
    step = len(frames) / float(max_frames)
    sampled = [frames[min(int(i * step), len(frames) - 1)] for i in range(max_frames)]
    return sampled


def extract_audio_from_video(video_path: str) -> Optional[str]:
    """
    Extract the audio track from a video file to a temporary WAV file.

    Returns the path to the WAV file, or ``None`` if extraction fails
    (e.g. the video has no audio track, or ffmpeg is unavailable).

    Passes an explicit ``format=`` hint derived from the file extension
    so pydub can decode using only the ``ffmpeg`` binary, without
    requiring ``ffprobe`` to be installed separately. This matters
    when the static binary from the ``imageio-ffmpeg`` wheel is used
    as a fallback (it ships ffmpeg but not ffprobe).
    """
    try:
        from pydub import AudioSegment

        ext = Path(video_path).suffix.lower().lstrip(".") or None
        # pydub maps "mp4"/"mov"/"m4v" to the same demuxer family.
        fmt_alias = {
            "mp4": "mp4", "m4v": "mp4", "mov": "mov",
            "mkv": "matroska", "webm": "webm", "avi": "avi",
        }
        fmt = fmt_alias.get(ext, ext)

        try:
            audio = AudioSegment.from_file(video_path, format=fmt) if fmt \
                else AudioSegment.from_file(video_path)
        except Exception:
            # Last-resort: let pydub auto-detect (needs ffprobe).
            audio = AudioSegment.from_file(video_path)

        if len(audio) == 0:
            return None

        wav_path = tempfile.mktemp(suffix=".wav")
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception:
        return None


async def describe_frames_with_groq(
    frames: List[Dict[str, Any]],
    api_key: str,
    model: str = GROQ_VISION_MODEL,
    batch_size: int = 4,
) -> List[Dict[str, Any]]:
    """
    Use Groq's multimodal Llama 4 Scout vision model to describe video frames.
    Groq's free tier has stricter rate limits, so we keep batches small and
    insert a short sleep between batches.
    """
    from groq import Groq

    client = Groq(api_key=api_key)
    descriptions: List[Dict[str, Any]] = []

    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]

        for frame_data in batch:
            frame = frame_data["frame"]
            timestamp = frame_data["timestamp"]
            frame_index = frame_data["frame_index"]

            b64_image = frame_to_base64(frame)

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "Describe this video frame in 2-3 concise, factual sentences. "
                                        "Focus on: key visual elements, any visible text, people and "
                                        "their actions, setting/background, and notable details. "
                                        "Do not speculate beyond what is visible."
                                    ),
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{b64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                    temperature=0.1,
                    max_tokens=250,
                )
                description = response.choices[0].message.content.strip()
            except Exception as e:
                description = f"[Frame description unavailable: {str(e)[:120]}]"

            descriptions.append({
                "timestamp": timestamp,
                "description": description,
                "frame_index": frame_index,
            })

        # Throttle to stay within Groq free-tier rate limits
        if i + batch_size < len(frames):
            await asyncio.sleep(1.2)

    return descriptions


async def describe_frames_with_gemini(
    frames: List[Dict[str, Any]],
    api_key: str,
    batch_size: int = 5,
) -> List[Dict[str, Any]]:
    """
    Use Google Gemini Vision to describe video frames.
    Processes frames in batches for efficiency.
    """
    import google.generativeai as genai
    from PIL import Image

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_VISION_MODEL)

    descriptions: List[Dict[str, Any]] = []

    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        
        for frame_data in batch:
            frame = frame_data["frame"]
            timestamp = frame_data["timestamp"]
            frame_index = frame_data["frame_index"]

            # Convert BGR to RGB for PIL
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            try:
                response = model.generate_content([
                    "Describe this video frame in 2-3 concise sentences. "
                    "Focus on: key visual elements, text visible on screen, "
                    "people and their actions, and any important details. "
                    "Be factual and specific.",
                    pil_image,
                ])
                description = response.text.strip()
            except Exception as e:
                description = f"[Frame description unavailable: {str(e)[:100]}]"

            descriptions.append({
                "timestamp": timestamp,
                "description": description,
                "frame_index": frame_index,
            })

        # Small delay between batches to respect rate limits
        if i + batch_size < len(frames):
            await asyncio.sleep(0.5)

    return descriptions


async def describe_frames_with_openai(
    frames: List[Dict[str, Any]],
    api_key: str,
    batch_size: int = 5,
) -> List[Dict[str, Any]]:
    """
    Use OpenAI GPT-4o Vision to describe video frames.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    descriptions: List[Dict[str, Any]] = []

    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]

        for frame_data in batch:
            frame = frame_data["frame"]
            timestamp = frame_data["timestamp"]
            frame_index = frame_data["frame_index"]

            b64_image = frame_to_base64(frame)

            try:
                response = client.chat.completions.create(
                    model=OPENAI_VISION_MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Describe this video frame in 2-3 concise sentences. "
                                            "Focus on key visual elements, any text, people and actions.",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                                },
                            ],
                        }
                    ],
                    max_tokens=200,
                )
                description = response.choices[0].message.content.strip()
            except Exception as e:
                description = f"[Frame description unavailable: {str(e)[:100]}]"

            descriptions.append({
                "timestamp": timestamp,
                "description": description,
                "frame_index": frame_index,
            })

        if i + batch_size < len(frames):
            await asyncio.sleep(0.5)

    return descriptions


async def describe_frames_simple(
    frames: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Generate simple frame descriptions without a vision model.
    Uses basic image analysis (color histograms, brightness, etc.)
    """
    descriptions: List[Dict[str, Any]] = []

    for frame_data in frames:
        frame = frame_data["frame"]
        timestamp = frame_data["timestamp"]
        frame_index = frame_data["frame_index"]

        # Basic analysis
        height, width = frame.shape[:2]
        mean_brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        
        # Detect dominant colors
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mean_hue = np.mean(hsv[:, :, 0])
        mean_saturation = np.mean(hsv[:, :, 1])

        brightness_desc = "dark" if mean_brightness < 85 else "bright" if mean_brightness > 170 else "moderately lit"
        color_desc = "colorful" if mean_saturation > 100 else "muted"

        description = (
            f"Video frame at {timestamp:.1f}s: {width}x{height} resolution, "
            f"{brightness_desc} scene with {color_desc} tones. "
            f"Average brightness: {mean_brightness:.0f}/255."
        )

        descriptions.append({
            "timestamp": timestamp,
            "description": description,
            "frame_index": frame_index,
        })

    return descriptions


async def process_video_file(
    video_path: str,
    source_name: str,
    vision_provider: Optional[str] = None,
    vision_api_key: Optional[str] = None,
    audio_provider: Optional[str] = None,
    audio_api_key: Optional[str] = None,
    audio_transcript: str = "",
    fps: int = FRAMES_PER_SECOND,
    max_vision_frames: int = MAX_VISION_FRAMES_PER_VIDEO,
) -> Tuple[List[Chunk], List[Dict[str, Any]]]:
    """
    Full video processing pipeline:
       1. Extract frames via OpenCV.
       2. Down-sample frames for the vision model (rate-limit friendly).
       3. Describe frames using the best available vision model
          (Groq Llama-4 Scout → Gemini → OpenAI → OpenCV heuristic).
       4. Optionally extract & transcribe the audio track via Whisper
          (Groq or OpenAI) to enrich each video chunk with spoken context.
       5. Build unified multimodal chunks (visual + audio).

    Returns:
        Tuple of (chunks, frame_descriptions)
    """
    # ── Step 1: Frame extraction ──────────────────────────────────────
    frames = extract_frames(video_path, fps=fps)
    if not frames:
        return [], []

    # ── Step 2: Smart sampling for the vision model ───────────────────
    vision_frames = smart_sample_frames(frames, max_frames=max_vision_frames)

    # ── Step 3: Frame description ─────────────────────────────────────
    if vision_provider == "groq" and vision_api_key:
        frame_descriptions = await describe_frames_with_groq(vision_frames, vision_api_key)
    elif vision_provider == "gemini" and vision_api_key:
        frame_descriptions = await describe_frames_with_gemini(vision_frames, vision_api_key)
    elif vision_provider == "openai" and vision_api_key:
        frame_descriptions = await describe_frames_with_openai(vision_frames, vision_api_key)
    else:
        # Fallback: simple analysis without vision model
        frame_descriptions = await describe_frames_simple(vision_frames)

    # ── Step 4: Audio transcription (best-effort) ─────────────────────
    if not audio_transcript and audio_api_key:
        audio_wav = extract_audio_from_video(video_path)
        if audio_wav:
            try:
                # Lazy import to avoid circular deps
                from utils.audio_processor import (
                    transcribe_audio_groq, transcribe_audio_openai,
                )
                if audio_provider == "openai":
                    segments = await transcribe_audio_openai(audio_wav, audio_api_key)
                else:
                    segments = await transcribe_audio_groq(audio_wav, audio_api_key)
                audio_transcript = " ".join(
                    (s.get("text") or "").strip() for s in segments
                ).strip()
            except Exception:
                audio_transcript = ""
            finally:
                try:
                    os.unlink(audio_wav)
                except Exception:
                    pass

    # ── Step 5: Build unified multimodal chunks ───────────────────────
    chunks = chunk_video_descriptions(
        frame_descriptions=frame_descriptions,
        audio_transcript=audio_transcript,
        source_file=source_name,
    )

    return chunks, frame_descriptions
