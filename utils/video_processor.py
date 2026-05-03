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

from config import FRAMES_PER_SECOND, MAX_VIDEO_DURATION, FRAME_RESIZE
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
    model = genai.GenerativeModel("gemini-2.0-flash")

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
                    model="gpt-4o",
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
    audio_transcript: str = "",
    fps: int = FRAMES_PER_SECOND,
) -> Tuple[List[Chunk], List[Dict[str, Any]]]:
    """
    Full video processing pipeline: extract frames → describe → chunk.
    
    Returns:
        Tuple of (chunks, frame_descriptions)
    """
    # Extract frames
    frames = extract_frames(video_path, fps=fps)

    if not frames:
        return [], []

    # Describe frames using available vision model
    if vision_provider == "gemini" and vision_api_key:
        frame_descriptions = await describe_frames_with_gemini(frames, vision_api_key)
    elif vision_provider == "openai" and vision_api_key:
        frame_descriptions = await describe_frames_with_openai(frames, vision_api_key)
    else:
        # Fallback: simple analysis without vision model
        frame_descriptions = await describe_frames_simple(frames)

    # Create unified multimodal chunks
    chunks = chunk_video_descriptions(
        frame_descriptions=frame_descriptions,
        audio_transcript=audio_transcript,
        source_file=source_name,
    )

    return chunks, frame_descriptions
