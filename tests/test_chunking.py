"""Unit tests for ``utils.chunking``."""
from __future__ import annotations

from utils.chunking import (
    Chunk,
    generate_chunk_id,
    estimate_tokens,
    clean_text,
    semantic_chunk_text,
    chunk_audio_transcript,
    chunk_video_descriptions,
    format_timestamp,
)


class TestHelpers:
    def test_estimate_tokens_rough(self):
        assert estimate_tokens("hello world") > 0
        assert estimate_tokens("") == 0

    def test_clean_text_collapses_whitespace(self):
        assert clean_text("hello   world\n\n\nfoo") == "hello world foo"

    def test_clean_text_strips_control_chars(self):
        assert "\x01" not in clean_text("hi\x01there")

    def test_clean_text_removes_page_numbers(self):
        out = clean_text("intro Page 5 of 12 continuing")
        assert "Page" not in out
        assert "intro" in out and "continuing" in out

    def test_generate_chunk_id_deterministic(self):
        a = generate_chunk_id("some content", "file.pdf", 0)
        b = generate_chunk_id("some content", "file.pdf", 0)
        c = generate_chunk_id("some content", "file.pdf", 1)
        assert a == b
        assert a != c
        assert len(a) == 12

    def test_format_timestamp(self):
        assert format_timestamp(0) == "00:00"
        assert format_timestamp(65) == "01:05"
        assert format_timestamp(3725) == "01:02:05"


class TestSemanticChunking:
    def test_empty_input_returns_empty(self):
        assert semantic_chunk_text("", "src") == []
        assert semantic_chunk_text("   ", "src") == []

    def test_single_short_text_one_chunk(self):
        text = "This is a single sentence. It is short enough to fit in one chunk."
        chunks = semantic_chunk_text(text, "src", chunk_size=600)
        assert len(chunks) == 1
        assert chunks[0].metadata["source"] == "src"
        assert chunks[0].metadata["modality"] == "text"
        assert chunks[0].metadata["chunk_index"] == 0

    def test_long_text_splits_into_multiple_chunks(self):
        # ~2000 tokens worth of text, well over the default 600-token limit.
        sentences = ["This is sentence number {i} with some filler content."
                     .format(i=i) * 5 for i in range(60)]
        text = " ".join(sentences)
        chunks = semantic_chunk_text(text, "doc.pdf", chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1
        # chunk_index must be monotonically increasing
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == sorted(indices)
        # every chunk knows its source
        assert all(c.metadata["source"] == "doc.pdf" for c in chunks)

    def test_extra_metadata_propagated(self):
        text = "Hello world. How are you today?"
        chunks = semantic_chunk_text(
            text, "x.pdf", extra_metadata={"page_number": 7, "total_pages": 10}
        )
        assert chunks[0].metadata["page_number"] == 7
        assert chunks[0].metadata["total_pages"] == 10


class TestAudioTranscriptChunking:
    def test_empty_segments(self):
        assert chunk_audio_transcript([], "audio.mp3") == []

    def test_segments_chunked_by_duration(self):
        segments = [
            {"text": f"segment {i}", "start": i * 10.0, "end": (i + 1) * 10.0}
            for i in range(6)
        ]
        chunks = chunk_audio_transcript(segments, "talk.mp3", chunk_duration=20.0)
        assert chunks, "should produce at least one chunk"
        for c in chunks:
            assert c.metadata["modality"] == "audio"
            assert "timestamp" in c.metadata
            assert c.metadata["source"] == "talk.mp3"

    def test_chunks_have_monotonic_times(self):
        segments = [
            {"text": "a", "start": 0.0, "end": 5.0},
            {"text": "b", "start": 5.0, "end": 25.0},
            {"text": "c", "start": 25.0, "end": 40.0},
        ]
        chunks = chunk_audio_transcript(segments, "x.mp3", chunk_duration=15.0)
        starts = [c.metadata["start_time"] for c in chunks]
        assert starts == sorted(starts)


class TestVideoChunking:
    def test_empty_descriptions(self):
        assert chunk_video_descriptions([], "audio txt", "vid.mp4") == []

    def test_video_chunks_include_audio_context(self):
        descs = [
            {"timestamp": float(i), "description": f"frame {i}", "frame_index": i}
            for i in range(20)
        ]
        chunks = chunk_video_descriptions(
            descs, audio_transcript="spoken context", source_file="v.mp4",
            chunk_duration=5.0,
        )
        assert chunks
        assert any("[Visual Content]" in c.content for c in chunks)
        assert any("[Audio Context]" in c.content for c in chunks)
        assert all(c.metadata["modality"] == "video" for c in chunks)
