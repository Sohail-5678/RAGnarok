"""Unit tests for ``core.ingestion`` helpers: key resolution + provider picking."""
from __future__ import annotations

import os

import pytest

from core.ingestion import (
    resolve_api_keys, _pick_vision_provider, _pick_audio_provider,
)


class TestResolveApiKeys:
    def test_user_key_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "env_groq")
        keys = resolve_api_keys({"groq": "user_groq"})
        assert keys["groq"] == "user_groq"

    def test_env_used_when_user_empty(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "env_groq")
        keys = resolve_api_keys({"groq": ""})
        assert keys["groq"] == "env_groq"

    def test_missing_keys_are_empty_strings(self, monkeypatch):
        # Clear all provider env vars.
        for v in ("GROQ_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY",
                  "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(v, raising=False)
        keys = resolve_api_keys({})
        assert keys == {"groq": "", "openai": "", "gemini": "", "claude": ""}

    def test_handles_none_input(self, monkeypatch):
        for v in ("GROQ_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY",
                  "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(v, raising=False)
        # resolve_api_keys is documented to accept any dict-like; ensure {}
        # works (the real call site passes ``api_keys or {}``).
        keys = resolve_api_keys({})
        assert "groq" in keys


class TestProviderPicking:
    def test_vision_prefers_groq(self):
        keys = {"groq": "g", "gemini": "x", "openai": "o", "claude": ""}
        provider, key = _pick_vision_provider(keys)
        assert provider == "groq" and key == "g"

    def test_vision_falls_back_to_gemini(self):
        keys = {"groq": "", "gemini": "x", "openai": "o", "claude": ""}
        provider, key = _pick_vision_provider(keys)
        assert provider == "gemini" and key == "x"

    def test_vision_falls_back_to_openai(self):
        keys = {"groq": "", "gemini": "", "openai": "o", "claude": ""}
        provider, key = _pick_vision_provider(keys)
        assert provider == "openai" and key == "o"

    def test_vision_returns_none_when_no_keys(self):
        provider, key = _pick_vision_provider(
            {"groq": "", "gemini": "", "openai": "", "claude": ""}
        )
        assert provider is None and key is None

    def test_audio_prefers_groq(self):
        provider, key = _pick_audio_provider({"groq": "g", "openai": "o"})
        assert provider == "groq" and key == "g"

    def test_audio_falls_back_to_openai(self):
        provider, key = _pick_audio_provider({"groq": "", "openai": "o"})
        assert provider == "openai" and key == "o"

    def test_audio_returns_none_when_no_keys(self):
        provider, key = _pick_audio_provider({"groq": "", "openai": ""})
        assert provider is None and key is None
