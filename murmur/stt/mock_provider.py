"""Mock STT provider for testing without microphone or API keys."""

from __future__ import annotations

from murmur.stt.base import STTProvider, STTResult


class MockSTTProvider(STTProvider):
    """Returns a fixed transcript. Used for unit tests and dry-run mode."""

    name = "mock"
    requires_api_key = False
    local = True

    def __init__(self, response: str = "Hello, this is a test transcript.", **_: object) -> None:
        self.response = response

    async def transcribe(self, audio: bytes, sample_rate: int = 16000) -> STTResult:
        return STTResult(
            text=self.response,
            confidence=1.0,
            language="en",
            duration_s=len(audio) / (sample_rate * 2),
            provider=self.name,
        )
