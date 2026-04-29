"""Cloud TTS providers: ElevenLabs, OpenAI TTS, Cartesia, Edge-TTS (free)."""

from __future__ import annotations

from typing import Any, AsyncIterator

from murmur.tts.base import TTSProvider, TTSChunk


class ElevenLabsProvider(TTSProvider):
    """ElevenLabs TTS — best voice quality, ultra-realistic.

    Requires: ELEVENLABS_API_KEY
    Install: pip install murmur-voice[elevenlabs]
    Free tier: 10,000 chars/month

    Voices: https://elevenlabs.io/voice-library
    Models: eleven_turbo_v2_5 (fast), eleven_multilingual_v2 (quality)
    """

    name = "elevenlabs"
    requires_api_key = True
    supports_streaming = True
    local = False
    sample_rate = 22050

    def __init__(
        self,
        api_key: str,
        voice_id: str = "JBFqnCBsd6RMkjVDRZzb",  # George — calm, natural
        model: str = "eleven_turbo_v2_5",
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        speed: float = 1.0,
        **_: Any,
    ) -> None:
        self.api_key = api_key
        self.voice_id = voice_id
        self.model = model
        self.stability = stability
        self.similarity_boost = similarity_boost
        self.speed = speed
        self._client: Any = None

    async def setup(self) -> None:
        try:
            from elevenlabs.client import AsyncElevenLabs
            self._client = AsyncElevenLabs(api_key=self.api_key)
        except ImportError:
            raise ImportError("elevenlabs not installed. Run: pip install murmur-voice[elevenlabs]")

    async def synthesize(self, text: str) -> AsyncIterator[TTSChunk]:  # type: ignore[override]
        if not self._client:
            await self.setup()

        response = await self._client.text_to_speech.convert(
            voice_id=self.voice_id,
            text=text,
            model_id=self.model,
            voice_settings={
                "stability": self.stability,
                "similarity_boost": self.similarity_boost,
                "speed": self.speed,
            },
            output_format="pcm_22050",
        )

        async for chunk in response:
            if chunk:
                yield TTSChunk(audio=chunk, sample_rate=22050, is_pcm=True)

    async def teardown(self) -> None:
        self._client = None


class OpenAITTSProvider(TTSProvider):
    """OpenAI TTS — tts-1 (fast) and tts-1-hd (high quality).

    Requires: OPENAI_API_KEY
    Voices: alloy, echo, fable, onyx, nova, shimmer
    Cost: ~$15/1M chars (tts-1), ~$30/1M chars (tts-1-hd)
    """

    name = "openai"
    requires_api_key = True
    supports_streaming = True
    local = False
    sample_rate = 24000

    def __init__(
        self,
        api_key: str,
        voice: str = "nova",
        model: str = "tts-1",
        speed: float = 1.0,
        **_: Any,
    ) -> None:
        self.api_key = api_key
        self.voice = voice
        self.model = model
        self.speed = speed
        self._client: Any = None

    async def setup(self) -> None:
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(api_key=self.api_key)

    async def synthesize(self, text: str) -> AsyncIterator[TTSChunk]:  # type: ignore[override]
        if not self._client:
            await self.setup()

        async with self._client.audio.speech.with_streaming_response.create(
            model=self.model,
            voice=self.voice,  # type: ignore[arg-type]
            input=text,
            speed=self.speed,
            response_format="pcm",
        ) as response:
            async for chunk in response.iter_bytes(chunk_size=4096):
                yield TTSChunk(audio=chunk, sample_rate=24000, is_pcm=True)

    async def teardown(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None


class CartesiaProvider(TTSProvider):
    """Cartesia TTS — ultra-low latency streaming, best for real-time use.

    Requires: CARTESIA_API_KEY
    Install: pip install murmur-voice[cartesia]
    Latency: <80ms to first byte
    """

    name = "cartesia"
    requires_api_key = True
    supports_streaming = True
    local = False
    sample_rate = 22050

    def __init__(
        self,
        api_key: str,
        voice_id: str = "a0e99841-438c-4a64-b679-ae501e7d6091",  # Barbershop Man
        model: str = "sonic-english",
        speed: str = "normal",
        **_: Any,
    ) -> None:
        self.api_key = api_key
        self.voice_id = voice_id
        self.model = model
        self.speed = speed
        self._client: Any = None

    async def setup(self) -> None:
        try:
            from cartesia import AsyncCartesia
            self._client = AsyncCartesia(api_key=self.api_key)
        except ImportError:
            raise ImportError("cartesia not installed. Run: pip install murmur-voice[cartesia]")

    async def synthesize(self, text: str) -> AsyncIterator[TTSChunk]:  # type: ignore[override]
        if not self._client:
            await self.setup()

        async for output in self._client.tts.sse(
            model_id=self.model,
            transcript=text,
            voice={"id": self.voice_id},
            output_format={"container": "raw", "encoding": "pcm_s16le", "sample_rate": 22050},
            stream=True,
        ):
            if output.audio:
                yield TTSChunk(audio=output.audio, sample_rate=22050, is_pcm=True)

    async def teardown(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None


class EdgeTTSProvider(TTSProvider):
    """Microsoft Edge TTS — completely free, no API key, 300+ voices.

    Install: pip install murmur-voice[edge-tts]
    Voices: run `edge-tts --list-voices` to see all options
    Popular: en-US-AriaNeural, en-US-GuyNeural, en-GB-SoniaNeural
    """

    name = "edge-tts"
    requires_api_key = False
    supports_streaming = True
    local = False  # uses Microsoft servers but no key required
    sample_rate = 24000

    def __init__(
        self,
        voice: str = "en-US-AriaNeural",
        rate: str = "+0%",
        volume: str = "+0%",
        **_: Any,
    ) -> None:
        self.voice = voice
        self.rate = rate
        self.volume = volume

    async def synthesize(self, text: str) -> AsyncIterator[TTSChunk]:  # type: ignore[override]
        try:
            import edge_tts
        except ImportError:
            raise ImportError("edge-tts not installed. Run: pip install murmur-voice[edge-tts]")

        import io
        import wave
        import tempfile
        import os

        communicate = edge_tts.Communicate(text, self.voice, rate=self.rate, volume=self.volume)

        # Edge-TTS outputs MP3; we collect then yield as single chunk
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])

        buf.seek(0)
        mp3_data = buf.read()
        if mp3_data:
            # Yield as encoded chunk (pipeline handles decoding if needed)
            yield TTSChunk(audio=mp3_data, sample_rate=24000, is_pcm=False)


class MockTTSProvider(TTSProvider):
    """Mock TTS for testing — outputs silence."""

    name = "mock"
    requires_api_key = False
    local = True
    sample_rate = 16000

    async def synthesize(self, text: str) -> AsyncIterator[TTSChunk]:  # type: ignore[override]
        import struct
        # 0.5 seconds of silence at 16kHz, 16-bit
        silence = struct.pack("<" + "h" * 8000, *([0] * 8000))
        yield TTSChunk(audio=silence, sample_rate=16000, is_pcm=True)
