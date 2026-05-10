"""Unit tests for ``core.generation`` — provider config + fallback routing."""
from __future__ import annotations

import pytest

from core.generation import (
    get_llm_config, get_available_providers, generate_response,
)


class TestGetLlmConfig:
    def test_returns_none_for_unknown_provider(self, monkeypatch):
        for v in ("GROQ_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY",
                  "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(v, raising=False)
        assert get_llm_config("nonexistent", {}) is None

    def test_returns_none_when_no_key_available(self, monkeypatch):
        for v in ("GROQ_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY",
                  "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(v, raising=False)
        assert get_llm_config("groq", {}) is None
        assert get_llm_config("groq", {"groq": ""}) is None

    def test_uses_user_key_when_provided(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        cfg = get_llm_config("groq", {"groq": "user_key"})
        assert cfg is not None
        assert cfg.api_key == "user_key"
        assert cfg.provider == "groq"

    def test_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "env_key")
        cfg = get_llm_config("groq", {"groq": ""})
        assert cfg is not None
        assert cfg.api_key == "env_key"

    def test_user_key_overrides_env(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "env_key")
        cfg = get_llm_config("groq", {"groq": "user_key"})
        assert cfg.api_key == "user_key"


class TestGetAvailableProviders:
    def test_empty_when_nothing_configured(self, monkeypatch):
        for v in ("GROQ_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY",
                  "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(v, raising=False)
        assert get_available_providers({}) == []

    def test_detects_user_keys(self, monkeypatch):
        for v in ("GROQ_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY",
                  "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(v, raising=False)
        assert set(get_available_providers({"groq": "x", "openai": "y"})) == {
            "groq", "openai",
        }


class TestGenerateResponseFallback:
    def test_yields_friendly_message_when_no_keys(self, monkeypatch):
        for v in ("GROQ_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY",
                  "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(v, raising=False)
        out = "".join(generate_response(
            question="hi",
            context="",
            provider="groq",
            api_keys={},
        ))
        assert "No API key configured" in out or "⚠️" in out
