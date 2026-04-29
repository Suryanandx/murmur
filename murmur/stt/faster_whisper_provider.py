"""STT via faster-whisper (local, no API key required)."""

from __future__ import annotations

import io
import wave
from typing import Any

import numpy as np

from murmur.stt.base import STTProvider, STTResult


class FasterWhisperProvider(STTProvider):
    """Local Whisper inference via CTranslate2-optimized faster-whisper.

    Install: pip install murmur-voice[faster-whisper]
    Models: tiny, tiny.en, base, base.en, small, medium, large-v3
    """

    name = "faster-whisper"
    requires_api_key = False
    local = True

    def __init__(
        self,
        model: str = "base.en",
        device: str = "auto",
        compute_type: str = "default",
        language: str | None = None,
        beam_size: int = 5,
        **kwargs: Any,
    ) -> None:
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self._model: Any = None

    async def setup(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper not installed. Run: pip install murmur-voice[faster-whisper]"
            )
        self._model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )

    async def transcribe(self, audio: bytes, sample_rate: int = 16000) -> STTResult:
        if not self._model:
            await self.setup()

        # Convert raw PCM bytes → float32 numpy array
        audio_np = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        segments, info = self._model.transcribe(
            audio_np,
            beam_size=self.beam_size,
            language=self.language,
            vad_filter=True,
        )

        text = " ".join(seg.text.strip() for seg in segments).strip()
        return STTResult(
            text=text,
            language=info.language,
            confidence=info.language_probability,
            provider=self.name,
        )

    async def teardown(self) -> None:
        self._model = None
