"""Abstract base class for all TTS providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class TTSChunk:
    audio: bytes           # raw PCM or encoded audio
    sample_rate: int = 24000
    channels: int = 1
    sample_width: int = 2  # bytes (16-bit)
    is_pcm: bool = True    # False if encoded (mp3/ogg)


class TTSProvider(ABC):
    """All TTS providers implement this interface.

    Providers receive text strings and yield TTSChunk objects containing
    raw audio bytes. Streaming providers yield multiple chunks per sentence.
    """

    name: str = "base"
    requires_api_key: bool = False
    supports_streaming: bool = False
    local: bool = False
    sample_rate: int = 24000

    @abstractmethod
    async def synthesize(self, text: str) -> AsyncIterator[TTSChunk]:
        """Synthesize text to audio chunks. Must be an async generator."""
        ...

    async def __aenter__(self) -> "TTSProvider":
        await self.setup()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.teardown()

    async def setup(self) -> None:
        """Load models, open connections."""

    async def teardown(self) -> None:
        """Release resources."""
