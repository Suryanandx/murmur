"""Microphone capture with push-to-talk and VAD modes."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import numpy as np

from murmur.config import AudioConfig


class AudioCapture:
    """Captures audio from the microphone in async chunks."""

    def __init__(self, cfg: AudioConfig) -> None:
        self.cfg = cfg
        self._stream: object | None = None
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()

    async def record_until_silence(
        self,
        vad: object | None = None,
        max_seconds: float = 30.0,
    ) -> bytes:
        """Record audio until VAD detects silence or max_seconds elapsed.

        If vad is None, records for max_seconds (use for push-to-talk).
        Returns raw PCM bytes (16-bit, mono).
        """
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError("sounddevice not installed. Run: pip install sounddevice")

        sample_rate = self.cfg.sample_rate
        chunk_frames = int(sample_rate * self.cfg.chunk_ms / 1000)
        max_frames = int(sample_rate * max_seconds)

        loop = asyncio.get_event_loop()
        buffer: list[np.ndarray] = []
        done_event = asyncio.Event()
        total_frames = 0

        def _callback(indata: np.ndarray, frames: int, time: object, status: object) -> None:
            nonlocal total_frames
            chunk = indata[:, 0].copy()
            buffer.append(chunk)
            total_frames += frames
            if total_frames >= max_frames:
                loop.call_soon_threadsafe(done_event.set)

        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=chunk_frames,
            device=self.cfg.input_device,
            callback=_callback,
        ):
            await done_event.wait()

        if not buffer:
            return b""

        audio = np.concatenate(buffer).astype(np.int16)
        return audio.tobytes()

    async def record_push_to_talk(self, stop_event: asyncio.Event, max_seconds: float = 30.0) -> bytes:
        """Record from mic until stop_event is set (push-to-talk mode)."""
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError("sounddevice not installed. Run: pip install sounddevice")

        sample_rate = self.cfg.sample_rate
        chunk_frames = int(sample_rate * self.cfg.chunk_ms / 1000)
        max_frames = int(sample_rate * max_seconds)
        loop = asyncio.get_event_loop()
        buffer: list[np.ndarray] = []
        total_frames = 0

        def _callback(indata: np.ndarray, frames: int, time: object, status: object) -> None:
            nonlocal total_frames
            buffer.append(indata[:, 0].copy())
            total_frames += frames
            if total_frames >= max_frames:
                loop.call_soon_threadsafe(stop_event.set)

        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=chunk_frames,
            device=self.cfg.input_device,
            callback=_callback,
        ):
            await stop_event.wait()

        if not buffer:
            return b""
        return np.concatenate(buffer).astype(np.int16).tobytes()

    @staticmethod
    def list_devices() -> list[dict]:
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            return [
                {"index": i, "name": d["name"], "channels": d["max_input_channels"]}
                for i, d in enumerate(devices)
                if d["max_input_channels"] > 0
            ]
        except ImportError:
            return []
