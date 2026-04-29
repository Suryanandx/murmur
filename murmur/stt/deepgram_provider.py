"""STT via Deepgram Nova-2 (streaming + batch)."""

from __future__ import annotations

from typing import Any, AsyncIterator

from murmur.stt.base import STTProvider, STTResult


class DeepgramProvider(STTProvider):
    """Deepgram Nova-2 STT — excellent streaming latency and accuracy.

    Requires: DEEPGRAM_API_KEY
    Install: pip install murmur-voice[deepgram]
    Models: nova-2, nova-2-general, nova-2-meeting, nova-2-phonecall
    """

    name = "deepgram"
    requires_api_key = True
    supports_streaming = True
    local = False

    def __init__(
        self,
        api_key: str,
        model: str = "nova-2",
        language: str = "en-US",
        smart_format: bool = True,
        punctuate: bool = True,
        **_: Any,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.language = language
        self.smart_format = smart_format
        self.punctuate = punctuate
        self._client: Any = None

    async def setup(self) -> None:
        try:
            from deepgram import DeepgramClient
        except ImportError:
            raise ImportError(
                "deepgram-sdk not installed. Run: pip install murmur-voice[deepgram]"
            )
        self._client = DeepgramClient(self.api_key)

    async def transcribe(self, audio: bytes, sample_rate: int = 16000) -> STTResult:
        if not self._client:
            await self.setup()

        from deepgram import PrerecordedOptions, FileSource

        options = PrerecordedOptions(
            model=self.model,
            language=self.language,
            smart_format=self.smart_format,
            punctuate=self.punctuate,
        )
        source: FileSource = {"buffer": audio, "mimetype": "audio/wav"}
        response = await self._client.listen.asyncrest.v("1").transcribe_file(source, options)
        result = response.results.channels[0].alternatives[0]

        return STTResult(
            text=result.transcript.strip(),
            confidence=result.confidence,
            language=self.language,
            provider=self.name,
        )


class AssemblyAIProvider(STTProvider):
    """AssemblyAI STT — great punctuation and speaker diarization.

    Requires: ASSEMBLYAI_API_KEY
    Install: pip install murmur-voice[assemblyai]
    """

    name = "assemblyai"
    requires_api_key = True
    local = False

    def __init__(self, api_key: str, language: str = "en", **_: Any) -> None:
        self.api_key = api_key
        self.language = language

    async def setup(self) -> None:
        try:
            import assemblyai as aai
            aai.settings.api_key = self.api_key
        except ImportError:
            raise ImportError(
                "assemblyai not installed. Run: pip install murmur-voice[assemblyai]"
            )

    async def transcribe(self, audio: bytes, sample_rate: int = 16000) -> STTResult:
        import asyncio
        import io
        import wave
        import assemblyai as aai

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio)
        buf.seek(0)

        config = aai.TranscriptionConfig(language_code=self.language)
        transcriber = aai.Transcriber(config=config)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: transcriber.transcribe(buf)
        )

        return STTResult(
            text=(result.text or "").strip(),
            confidence=result.words[0].confidence if result.words else None,
            language=self.language,
            provider=self.name,
        )
