"""STT via OpenAI Whisper API."""

from __future__ import annotations

import io
import wave
from typing import Any

from murmur.stt.base import STTProvider, STTResult


class OpenAISTTProvider(STTProvider):
    """OpenAI Whisper API (whisper-1 model).

    Requires: OPENAI_API_KEY
    Cost: $0.006/minute
    """

    name = "openai"
    requires_api_key = True
    local = False

    def __init__(self, api_key: str, model: str = "whisper-1", language: str = "en", **_: Any) -> None:
        self.api_key = api_key
        self.model = model
        self.language = language
        self._client: Any = None

    async def setup(self) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")
        self._client = AsyncOpenAI(api_key=self.api_key)

    async def transcribe(self, audio: bytes, sample_rate: int = 16000) -> STTResult:
        if not self._client:
            await self.setup()

        wav_buf = _pcm_to_wav(audio, sample_rate)
        wav_buf.name = "audio.wav"

        response = await self._client.audio.transcriptions.create(
            model=self.model,
            file=wav_buf,
            language=self.language,
            response_format="verbose_json",
        )

        return STTResult(
            text=response.text.strip(),
            language=response.language,
            duration_s=response.duration,
            provider=self.name,
        )

    async def teardown(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None


class GroqSTTProvider(STTProvider):
    """Groq Whisper API — faster and cheaper than OpenAI Whisper.

    Requires: GROQ_API_KEY
    Models: whisper-large-v3, whisper-large-v3-turbo, distil-whisper-large-v3-en
    """

    name = "groq"
    requires_api_key = True
    local = False

    def __init__(
        self,
        api_key: str,
        model: str = "whisper-large-v3-turbo",
        language: str = "en",
        **_: Any,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.language = language
        self._client: Any = None

    async def setup(self) -> None:
        try:
            from groq import AsyncGroq
        except ImportError:
            raise ImportError("groq not installed. Run: pip install groq")
        self._client = AsyncGroq(api_key=self.api_key)

    async def transcribe(self, audio: bytes, sample_rate: int = 16000) -> STTResult:
        if not self._client:
            await self.setup()

        wav_buf = _pcm_to_wav(audio, sample_rate)
        wav_buf.name = "audio.wav"

        response = await self._client.audio.transcriptions.create(
            file=wav_buf,
            model=self.model,
            language=self.language,
            response_format="verbose_json",
        )

        return STTResult(
            text=response.text.strip(),
            language=getattr(response, "language", self.language),
            provider=self.name,
        )

    async def teardown(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int) -> Any:
    """Wrap raw PCM in a WAV container for API uploads."""
    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    buf.seek(0)
    return buf
