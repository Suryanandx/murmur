"""Config loading: TOML file → env var overrides → validated Pydantic model."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class AudioConfig(BaseModel):
    input_device: int | str | None = None
    output_device: int | str | None = None
    sample_rate: int = 16000
    channels: int = 1
    chunk_ms: int = 30


class VADConfig(BaseModel):
    provider: Literal["silero", "webrtcvad", "none"] = "silero"
    threshold: float = 0.5
    min_silence_ms: int = 700
    min_speech_ms: int = 250
    padding_ms: int = 300


class STTConfig(BaseModel):
    provider: Literal[
        "faster-whisper", "openai", "groq", "deepgram", "assemblyai", "mock"
    ] = "faster-whisper"
    language: str = "en"
    # provider-specific
    config: dict[str, Any] = Field(default_factory=dict)


class LLMConfig(BaseModel):
    provider: Literal["openrouter", "openai", "anthropic", "ollama", "mock"] = "openrouter"
    model: str = "mistralai/mistral-7b-instruct"
    system_prompt: str = (
        "You are a helpful voice assistant. Be concise — "
        "your responses will be spoken aloud, so avoid markdown, lists, or long paragraphs."
    )
    max_history_turns: int = 10
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = True
    # provider-specific
    config: dict[str, Any] = Field(default_factory=dict)


class TTSConfig(BaseModel):
    provider: Literal[
        "kokoro", "piper", "elevenlabs", "openai", "cartesia", "edge-tts", "mock"
    ] = "kokoro"
    voice: str = "af_sarah"
    speed: float = 1.0
    # provider-specific
    config: dict[str, Any] = Field(default_factory=dict)


class PipelineConfig(BaseModel):
    mode: Literal["push-to-talk", "vad", "continuous"] = "push-to-talk"
    max_recording_s: float = 30.0
    sentence_min_chars: int = 20  # buffer TTS chunks until this many chars
    enable_ui: bool = False
    ui_port: int = 8765
    record_session: bool = False
    session_dir: Path = Path("~/.murmur/sessions")


class MurmurConfig(BaseModel):
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)

    @model_validator(mode="after")
    def apply_env_overrides(self) -> "MurmurConfig":
        env_map = {
            "MURMUR_STT_PROVIDER": ("stt", "provider"),
            "MURMUR_TTS_PROVIDER": ("tts", "provider"),
            "MURMUR_LLM_PROVIDER": ("llm", "provider"),
            "MURMUR_LLM_MODEL": ("llm", "model"),
            "MURMUR_PIPELINE_MODE": ("pipeline", "mode"),
        }
        for env_key, (section, field) in env_map.items():
            val = os.getenv(env_key)
            if val:
                setattr(getattr(self, section), field, val)
        return self

    @classmethod
    def load(cls, path: Path | str | None = None) -> "MurmurConfig":
        """Load config from TOML file. Falls back to defaults if no file found."""
        search_paths = [
            Path(path) if path else None,
            Path("murmur.toml"),
            Path.home() / ".config" / "murmur" / "config.toml",
        ]

        for p in search_paths:
            if p and p.exists():
                with open(p, "rb") as f:
                    data = tomllib.load(f)
                return cls.model_validate(data)

        return cls()

    def api_key(self, name: str) -> str | None:
        """Resolve an API key by checking env vars then credential file."""
        env_keys = {
            "openrouter": "OPENROUTER_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "deepgram": "DEEPGRAM_API_KEY",
            "assemblyai": "ASSEMBLYAI_API_KEY",
            "groq": "GROQ_API_KEY",
            "elevenlabs": "ELEVENLABS_API_KEY",
            "cartesia": "CARTESIA_API_KEY",
        }
        env_var = env_keys.get(name)
        if env_var:
            val = os.getenv(env_var)
            if val:
                return val

        cred_file = Path.home() / ".config" / "murmur" / "credentials"
        if cred_file.exists():
            for line in cred_file.read_text().splitlines():
                line = line.strip()
                if line.startswith(f"{name.upper()}_API_KEY="):
                    return line.split("=", 1)[1].strip()

        return None
