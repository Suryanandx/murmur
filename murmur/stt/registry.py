"""STT provider registry — maps provider names to classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from murmur.config import MurmurConfig, STTConfig
from murmur.stt.base import STTProvider

if TYPE_CHECKING:
    pass

PROVIDER_INFO = {
    "faster-whisper": {
        "description": "Local Whisper via CTranslate2. No API key. Best for offline use.",
        "local": True,
        "requires_api_key": False,
        "install": "pip install murmur-voice[faster-whisper]",
        "models": ["tiny.en", "base.en", "small.en", "medium", "large-v3"],
    },
    "openai": {
        "description": "OpenAI Whisper API. Accurate, cloud-based.",
        "local": False,
        "requires_api_key": True,
        "env_key": "OPENAI_API_KEY",
        "install": "pip install openai",
        "models": ["whisper-1"],
    },
    "groq": {
        "description": "Groq Whisper API. Fastest cloud STT, free tier available.",
        "local": False,
        "requires_api_key": True,
        "env_key": "GROQ_API_KEY",
        "install": "pip install groq",
        "models": ["whisper-large-v3", "whisper-large-v3-turbo", "distil-whisper-large-v3-en"],
    },
    "deepgram": {
        "description": "Deepgram Nova-2. Best streaming latency, speaker diarization.",
        "local": False,
        "requires_api_key": True,
        "env_key": "DEEPGRAM_API_KEY",
        "install": "pip install murmur-voice[deepgram]",
        "models": ["nova-2", "nova-2-general", "nova-2-meeting", "nova-2-phonecall"],
    },
    "assemblyai": {
        "description": "AssemblyAI. Best punctuation and formatting.",
        "local": False,
        "requires_api_key": True,
        "env_key": "ASSEMBLYAI_API_KEY",
        "install": "pip install murmur-voice[assemblyai]",
        "models": ["default"],
    },
    "mock": {
        "description": "Mock provider for testing. Returns fixed text.",
        "local": True,
        "requires_api_key": False,
        "install": None,
        "models": ["n/a"],
    },
}


def list_stt_providers() -> dict[str, dict]:
    return PROVIDER_INFO


def get_stt_provider(cfg: STTConfig, murmur_cfg: MurmurConfig) -> STTProvider:
    """Instantiate the configured STT provider with resolved API keys."""
    name = cfg.provider
    api_key = murmur_cfg.api_key(name) if PROVIDER_INFO.get(name, {}).get("requires_api_key") else None
    extra = cfg.config

    if name == "faster-whisper":
        from murmur.stt.faster_whisper_provider import FasterWhisperProvider
        return FasterWhisperProvider(
            model=extra.get("model", "base.en"),
            device=extra.get("device", "auto"),
            compute_type=extra.get("compute_type", "default"),
            language=extra.get("language", cfg.language) or None,
            beam_size=extra.get("beam_size", 5),
        )

    if name == "openai":
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set. Add it to env or ~/.config/murmur/credentials")
        from murmur.stt.openai_provider import OpenAISTTProvider
        return OpenAISTTProvider(
            api_key=api_key,
            model=extra.get("model", "whisper-1"),
            language=cfg.language,
        )

    if name == "groq":
        if not api_key:
            raise ValueError("GROQ_API_KEY not set.")
        from murmur.stt.openai_provider import GroqSTTProvider
        return GroqSTTProvider(
            api_key=api_key,
            model=extra.get("model", "whisper-large-v3-turbo"),
            language=cfg.language,
        )

    if name == "deepgram":
        if not api_key:
            raise ValueError("DEEPGRAM_API_KEY not set.")
        from murmur.stt.deepgram_provider import DeepgramProvider
        return DeepgramProvider(
            api_key=api_key,
            model=extra.get("model", "nova-2"),
            language=extra.get("language", cfg.language),
        )

    if name == "assemblyai":
        if not api_key:
            raise ValueError("ASSEMBLYAI_API_KEY not set.")
        from murmur.stt.deepgram_provider import AssemblyAIProvider
        return AssemblyAIProvider(api_key=api_key, language=cfg.language)

    if name == "mock":
        from murmur.stt.mock_provider import MockSTTProvider
        return MockSTTProvider(response=extra.get("response", "Hello, this is a test."))

    raise ValueError(f"Unknown STT provider: {name!r}. Valid: {list(PROVIDER_INFO)}")
