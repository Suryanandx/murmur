"""TTS via Kokoro — high-quality local TTS, Apache 2.0 licensed."""

from __future__ import annotations

from typing import Any, AsyncIterator

from murmur.tts.base import TTSProvider, TTSChunk


KOKORO_VOICES = {
    "af_sarah": "American Female — Sarah (warm, clear)",
    "af_bella": "American Female — Bella (expressive)",
    "af_nicole": "American Female — Nicole (professional)",
    "am_adam": "American Male — Adam (deep)",
    "am_michael": "American Male — Michael (natural)",
    "bf_emma": "British Female — Emma (posh, articulate)",
    "bf_isabella": "British Female — Isabella (elegant)",
    "bm_george": "British Male — George (authoritative)",
    "bm_lewis": "British Male — Lewis (warm)",
}


class KokoroProvider(TTSProvider):
    """Kokoro TTS — best local voice quality, Apache 2.0.

    Install: pip install murmur-voice[kokoro]
    No API key, no internet required.
    """

    name = "kokoro"
    requires_api_key = False
    supports_streaming = True
    local = True
    sample_rate = 24000

    def __init__(
        self,
        voice: str = "af_sarah",
        speed: float = 1.0,
        lang: str = "en-us",
        **_: Any,
    ) -> None:
        self.voice = voice
        self.speed = speed
        self.lang = lang
        self._pipeline: Any = None

    async def setup(self) -> None:
        try:
            from kokoro import KPipeline
        except ImportError:
            raise ImportError(
                "kokoro not installed. Run: pip install murmur-voice[kokoro]\n"
                "Note: kokoro requires Linux or macOS (not Windows)."
            )
        import asyncio
        loop = asyncio.get_event_loop()
        self._pipeline = await loop.run_in_executor(None, KPipeline, self.lang)

    async def synthesize(self, text: str) -> AsyncIterator[TTSChunk]:  # type: ignore[override]
        if not self._pipeline:
            await self.setup()

        import asyncio
        import numpy as np

        loop = asyncio.get_event_loop()

        def _run() -> list[bytes]:
            chunks = []
            generator = self._pipeline(text, voice=self.voice, speed=self.speed, split_pattern=r"\n+")
            for _, _, audio in generator:
                if audio is not None:
                    pcm = (audio * 32767).astype(np.int16).tobytes()
                    chunks.append(pcm)
            return chunks

        audio_chunks = await loop.run_in_executor(None, _run)
        for chunk in audio_chunks:
            yield TTSChunk(audio=chunk, sample_rate=self.sample_rate, is_pcm=True)

    async def teardown(self) -> None:
        self._pipeline = None
