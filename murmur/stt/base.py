"""Abstract base class for all STT providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class STTResult:
    text: str
    confidence: float | None = None
    language: str | None = None
    duration_s: float | None = None
    provider: str = ""


class STTProvider(ABC):
    """All STT providers implement this interface.

    Providers receive raw PCM audio bytes and return either a final
    STTResult or a stream of partial results for real-time display.
    """

    name: str = "base"
    requires_api_key: bool = False
    supports_streaming: bool = False
    local: bool = False

    @abstractmethod
    async def transcribe(self, audio: bytes, sample_rate: int = 16000) -> STTResult:
        """Transcribe audio bytes to text. Blocking until full result."""
        ...

    async def transcribe_stream(
        self, audio_chunks: AsyncIterator[bytes], sample_rate: int = 16000
    ) -> AsyncIterator[STTResult]:
        """Stream partial transcripts. Default: buffer and call transcribe once."""
        chunks: list[bytes] = []
        async for chunk in audio_chunks:
            chunks.append(chunk)
        result = await self.transcribe(b"".join(chunks), sample_rate)
        yield result

    async def __aenter__(self) -> "STTProvider":
        await self.setup()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.teardown()

    async def setup(self) -> None:
        """Called once before use. Load models, open connections."""

    async def teardown(self) -> None:
        """Called once after use. Release resources."""
