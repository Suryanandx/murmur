"""Unit tests for config loading and validation."""

import os
import tempfile
from pathlib import Path

import pytest

from murmur.config import MurmurConfig


def test_defaults() -> None:
    cfg = MurmurConfig()
    assert cfg.stt.provider == "faster-whisper"
    assert cfg.llm.provider == "openrouter"
    assert cfg.tts.provider == "kokoro"
    assert cfg.pipeline.mode == "push-to-talk"


def test_load_from_toml(tmp_path: Path) -> None:
    toml = tmp_path / "murmur.toml"
    toml.write_text("""
[stt]
provider = "groq"
language = "es"

[llm]
provider = "openrouter"
model = "mistralai/mistral-7b-instruct:free"

[tts]
provider = "edge-tts"
""")
    cfg = MurmurConfig.load(toml)
    assert cfg.stt.provider == "groq"
    assert cfg.stt.language == "es"
    assert cfg.tts.provider == "edge-tts"


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MURMUR_STT_PROVIDER", "deepgram")
    monkeypatch.setenv("MURMUR_LLM_MODEL", "openai/gpt-4o-mini")
    cfg = MurmurConfig()
    assert cfg.stt.provider == "deepgram"
    assert cfg.llm.model == "openai/gpt-4o-mini"


def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    cfg = MurmurConfig()
    assert cfg.api_key("openrouter") == "sk-or-test-key"


def test_api_key_missing() -> None:
    cfg = MurmurConfig()
    assert cfg.api_key("nonexistent") is None


def test_load_falls_back_to_defaults_when_no_file() -> None:
    cfg = MurmurConfig.load("/nonexistent/path/murmur.toml")
    assert cfg.stt.provider == "faster-whisper"
