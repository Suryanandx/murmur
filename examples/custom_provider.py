"""How to write and register a custom STT provider.

This example adds a custom STT provider that calls a hypothetical
in-house transcription API. Drop this pattern into your own project.
"""

import asyncio
from murmur.stt.base import STTProvider, STTResult
from murmur.config import MurmurConfig
from murmur.pipeline import PipelineSession


class MyCustomSTTProvider(STTProvider):
    """Example custom STT provider."""

    name = "my-custom-stt"
    requires_api_key = True
    local = False

    def __init__(self, api_key: str, endpoint: str, **_: object) -> None:
        self.api_key = api_key
        self.endpoint = endpoint

    async def setup(self) -> None:
        # Initialize your client, load models, etc.
        print(f"[MyCustomSTTProvider] Connected to {self.endpoint}")

    async def transcribe(self, audio: bytes, sample_rate: int = 16000) -> STTResult:
        # In a real implementation, POST audio to your API
        # and parse the response.
        return STTResult(
            text="custom provider transcript",
            confidence=0.95,
            language="en",
            provider=self.name,
        )

    async def teardown(self) -> None:
        print("[MyCustomSTTProvider] Disconnected")


async def main() -> None:
    cfg = MurmurConfig()
    cfg.llm.provider = "mock"   # type: ignore[assignment]
    cfg.tts.provider = "mock"   # type: ignore[assignment]

    async with PipelineSession(cfg) as session:
        # Swap the STT provider after initialization
        await session._stt.teardown()  # type: ignore[union-attr]
        session._stt = MyCustomSTTProvider(
            api_key="my-secret-key",
            endpoint="https://stt.mycompany.com/v1/transcribe",
        )
        await session._stt.setup()

        fake_audio = b"\x00\x00" * 8000
        response = await session.process_audio(fake_audio)
        print(f"Response: {response}")


if __name__ == "__main__":
    asyncio.run(main())
