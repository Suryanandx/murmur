"""Unit tests for mock providers and provider registries."""

import pytest

from murmur.config import MurmurConfig
from murmur.stt.registry import get_stt_provider, list_stt_providers
from murmur.llm.registry import get_llm_provider, list_llm_providers
from murmur.tts.registry import get_tts_provider, list_tts_providers


def test_list_stt_providers() -> None:
    providers = list_stt_providers()
    assert "faster-whisper" in providers
    assert "groq" in providers
    assert "deepgram" in providers
    assert "openai" in providers
    assert "assemblyai" in providers
    assert "mock" in providers


def test_list_llm_providers() -> None:
    providers = list_llm_providers()
    assert "openrouter" in providers
    assert "openai" in providers
    assert "anthropic" in providers
    assert "ollama" in providers
    assert "mock" in providers


def test_list_tts_providers() -> None:
    providers = list_tts_providers()
    assert "kokoro" in providers
    assert "piper" in providers
    assert "elevenlabs" in providers
    assert "openai" in providers
    assert "cartesia" in providers
    assert "edge-tts" in providers
    assert "mock" in providers


@pytest.mark.asyncio
async def test_mock_stt_provider() -> None:
    cfg = MurmurConfig()
    cfg.stt.provider = "mock"  # type: ignore[assignment]
    cfg.stt.config["response"] = "test transcript"
    provider = get_stt_provider(cfg.stt, cfg)
    result = await provider.transcribe(b"\x00" * 1000)
    assert result.text == "test transcript"
    assert result.provider == "mock"


@pytest.mark.asyncio
async def test_mock_llm_provider() -> None:
    cfg = MurmurConfig()
    cfg.llm.provider = "mock"  # type: ignore[assignment]
    cfg.llm.config["response"] = "mock LLM response"
    provider = get_llm_provider(cfg.llm, cfg)
    result = await provider.complete([{"role": "user", "content": "hi"}])
    assert result.text == "mock LLM response"
    assert result.provider == "mock"


@pytest.mark.asyncio
async def test_mock_llm_stream() -> None:
    cfg = MurmurConfig()
    cfg.llm.provider = "mock"  # type: ignore[assignment]
    cfg.llm.config["response"] = "hello world"
    provider = get_llm_provider(cfg.llm, cfg)
    tokens = []
    async for token in provider.stream([{"role": "user", "content": "hi"}]):
        tokens.append(token)
    assert "".join(tokens).strip() == "hello world"


@pytest.mark.asyncio
async def test_mock_tts_provider() -> None:
    cfg = MurmurConfig()
    cfg.tts.provider = "mock"  # type: ignore[assignment]
    provider = get_tts_provider(cfg.tts, cfg)
    chunks = []
    async for chunk in provider.synthesize("Hello world"):
        chunks.append(chunk)
    assert len(chunks) > 0
    assert all(c.is_pcm for c in chunks)


def test_unknown_stt_provider_raises() -> None:
    cfg = MurmurConfig()
    cfg.stt.provider = "nonexistent"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Unknown STT provider"):
        get_stt_provider(cfg.stt, cfg)


def test_openrouter_missing_key_raises() -> None:
    import os
    cfg = MurmurConfig()
    cfg.llm.provider = "openrouter"  # type: ignore[assignment]
    # Ensure no key is set
    env_backup = os.environ.pop("OPENROUTER_API_KEY", None)
    try:
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            get_llm_provider(cfg.llm, cfg)
    finally:
        if env_backup:
            os.environ["OPENROUTER_API_KEY"] = env_backup
