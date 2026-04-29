"""Piper TTS — extremely fast local TTS, great for edge devices."""

from __future__ import annotations

from typing import Any, AsyncIterator

from murmur.tts.base import TTSProvider, TTSChunk


class PiperProvider(TTSProvider):
    """Piper TTS — fast, lightweight, runs on Raspberry Pi.

    Install: pip install murmur-voice[piper]
    Models: https://github.com/rhasspy/piper/blob/master/VOICES.md
    """

    name = "piper"
    requires_api_key = False
    local = True
    sample_rate = 22050

    def __init__(self, model: str = "en_US-lessac-medium", speed: float = 1.0, **_: Any) -> None:
        self.model = model
        self.speed = speed
        self._piper: Any = None

    async def setup(self) -> None:
        try:
            from piper.voice import PiperVoice
            import asyncio
            loop = asyncio.get_event_loop()
            self._piper = await loop.run_in_executor(None, PiperVoice.load, self.model)
        except ImportError:
            raise ImportError("piper-tts not installed. Run: pip install murmur-voice[piper]")

    async def synthesize(self, text: str) -> AsyncIterator[TTSChunk]:  # type: ignore[override]
        if not self._piper:
            await self.setup()

        import asyncio
        import io
        import wave

        loop = asyncio.get_event_loop()

        def _run() -> bytes:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                self._piper.synthesize(text, wf, length_scale=1.0 / self.speed)
            buf.seek(0)
            with wave.open(buf, "rb") as wf:
                return wf.readframes(wf.getnframes())

        pcm = await loop.run_in_executor(None, _run)
        yield TTSChunk(audio=pcm, sample_rate=self.sample_rate, is_pcm=True)

    async def teardown(self) -> None:
        self._piper = None
