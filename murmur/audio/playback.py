"""Audio playback with support for PCM and encoded audio (MP3 via ffmpeg)."""

from __future__ import annotations

import asyncio
import io
from typing import AsyncIterator

from murmur.config import AudioConfig
from murmur.tts.base import TTSChunk


class AudioPlayback:
    """Plays TTS audio chunks as they arrive (streaming playback)."""

    def __init__(self, cfg: AudioConfig) -> None:
        self.cfg = cfg

    async def play_chunks(self, chunks: AsyncIterator[TTSChunk]) -> None:
        """Stream audio chunks to the speaker in real time."""
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError("sounddevice not installed. Run: pip install sounddevice")

        stream: object | None = None
        sample_rate: int | None = None

        async for chunk in chunks:
            audio = chunk.audio
            if not chunk.is_pcm:
                audio = await _decode_audio(audio, chunk.sample_rate)

            if stream is None or sample_rate != chunk.sample_rate:
                if stream is not None:
                    stream.stop()  # type: ignore[attr-defined]
                    stream.close()  # type: ignore[attr-defined]
                sample_rate = chunk.sample_rate
                stream = sd.RawOutputStream(
                    samplerate=sample_rate,
                    channels=chunk.channels,
                    dtype="int16",
                    device=self.cfg.output_device,
                )
                stream.start()  # type: ignore[attr-defined]

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, stream.write, audio)  # type: ignore[attr-defined]

        if stream is not None:
            stream.stop()  # type: ignore[attr-defined]
            stream.close()  # type: ignore[attr-defined]

    async def play_bytes(self, pcm: bytes, sample_rate: int = 24000, channels: int = 1) -> None:
        """Play a single buffer of PCM audio, blocking until complete."""
        try:
            import sounddevice as sd
            import numpy as np
        except ImportError:
            raise ImportError("sounddevice not installed.")

        audio = np.frombuffer(pcm, dtype=np.int16)
        if channels > 1:
            audio = audio.reshape(-1, channels)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: sd.play(audio, samplerate=sample_rate, blocking=True, device=self.cfg.output_device),
        )


async def _decode_audio(encoded: bytes, sample_rate: int) -> bytes:
    """Decode MP3/OGG to PCM using ffmpeg subprocess."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-i", "pipe:0",
        "-f", "s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    pcm, _ = await proc.communicate(input=encoded)
    return pcm
