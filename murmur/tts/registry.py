"""TTS provider registry."""

from __future__ import annotations

from murmur.config import MurmurConfig, TTSConfig
from murmur.tts.base import TTSProvider

PROVIDER_INFO = {
    "kokoro": {
        "description": "Local high-quality TTS. Apache 2.0. No API key.",
        "local": True,
        "requires_api_key": False,
        "install": "pip install murmur-voice[kokoro]",
        "voices": ["af_sarah", "af_bella", "af_nicole", "am_adam", "am_michael",
                   "bf_emma", "bf_isabella", "bm_george", "bm_lewis"],
        "platform": "Linux/macOS only",
    },
    "piper": {
        "description": "Extremely fast local TTS. Great for edge/Raspberry Pi.",
        "local": True,
        "requires_api_key": False,
        "install": "pip install murmur-voice[piper]",
        "platform": "All platforms",
    },
    "elevenlabs": {
        "description": "Best voice quality. Ultra-realistic. 10K free chars/month.",
        "local": False,
        "requires_api_key": True,
        "env_key": "ELEVENLABS_API_KEY",
        "signup": "https://elevenlabs.io",
        "install": "pip install murmur-voice[elevenlabs]",
        "models": ["eleven_turbo_v2_5", "eleven_multilingual_v2", "eleven_monolingual_v1"],
    },
    "openai": {
        "description": "OpenAI TTS (tts-1 / tts-1-hd). Good quality, reliable.",
        "local": False,
        "requires_api_key": True,
        "env_key": "OPENAI_API_KEY",
        "voices": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
        "models": ["tts-1", "tts-1-hd"],
    },
    "cartesia": {
        "description": "Ultra-low latency (<80ms). Best for real-time conversation.",
        "local": False,
        "requires_api_key": True,
        "env_key": "CARTESIA_API_KEY",
        "signup": "https://cartesia.ai",
        "install": "pip install murmur-voice[cartesia]",
    },
    "edge-tts": {
        "description": "Microsoft Edge TTS. Free, no API key. 300+ voices.",
        "local": False,
        "requires_api_key": False,
        "install": "pip install murmur-voice[edge-tts]",
        "note": "Uses Microsoft servers. No key, but requires internet.",
    },
    "mock": {
        "description": "Mock provider. Outputs silence. For testing.",
        "local": True,
        "requires_api_key": False,
    },
}


def list_tts_providers() -> dict[str, dict]:
    return PROVIDER_INFO


def get_tts_provider(cfg: TTSConfig, murmur_cfg: MurmurConfig) -> TTSProvider:
    name = cfg.provider
    api_key = murmur_cfg.api_key(name) if PROVIDER_INFO.get(name, {}).get("requires_api_key") else None
    extra = cfg.config

    if name == "kokoro":
        from murmur.tts.kokoro_provider import KokoroProvider
        return KokoroProvider(
            voice=extra.get("voice", cfg.voice),
            speed=extra.get("speed", cfg.speed),
            lang=extra.get("lang", "en-us"),
        )

    if name == "piper":
        from murmur.tts.piper_provider import PiperProvider
        return PiperProvider(
            model=extra.get("model", "en_US-lessac-medium"),
            speed=cfg.speed,
        )

    if name == "elevenlabs":
        if not api_key:
            raise ValueError(
                "ELEVENLABS_API_KEY not set. Sign up at https://elevenlabs.io (free tier available)."
            )
        from murmur.tts.cloud_providers import ElevenLabsProvider
        return ElevenLabsProvider(
            api_key=api_key,
            voice_id=extra.get("voice_id", "JBFqnCBsd6RMkjVDRZzb"),
            model=extra.get("model", "eleven_turbo_v2_5"),
            stability=extra.get("stability", 0.5),
            similarity_boost=extra.get("similarity_boost", 0.75),
            speed=cfg.speed,
        )

    if name == "openai":
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set.")
        from murmur.tts.cloud_providers import OpenAITTSProvider
        return OpenAITTSProvider(
            api_key=api_key,
            voice=extra.get("voice", cfg.voice if cfg.voice in ["alloy","echo","fable","onyx","nova","shimmer"] else "nova"),
            model=extra.get("model", "tts-1"),
            speed=cfg.speed,
        )

    if name == "cartesia":
        if not api_key:
            raise ValueError("CARTESIA_API_KEY not set. Sign up at https://cartesia.ai")
        from murmur.tts.cloud_providers import CartesiaProvider
        return CartesiaProvider(
            api_key=api_key,
            voice_id=extra.get("voice_id", "a0e99841-438c-4a64-b679-ae501e7d6091"),
            model=extra.get("model", "sonic-english"),
            speed=extra.get("speed", "normal"),
        )

    if name == "edge-tts":
        from murmur.tts.cloud_providers import EdgeTTSProvider
        return EdgeTTSProvider(
            voice=extra.get("voice", cfg.voice if cfg.voice else "en-US-AriaNeural"),
            rate=extra.get("rate", "+0%"),
            volume=extra.get("volume", "+0%"),
        )

    if name == "mock":
        from murmur.tts.cloud_providers import MockTTSProvider
        return MockTTSProvider()

    raise ValueError(f"Unknown TTS provider: {name!r}. Valid: {list(PROVIDER_INFO)}")
