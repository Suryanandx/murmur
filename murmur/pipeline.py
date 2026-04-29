"""PipelineSession — the central orchestrator connecting STT → LLM → TTS."""

from __future__ import annotations

import asyncio
import re
from typing import AsyncIterator

from murmur.audio.capture import AudioCapture
from murmur.audio.playback import AudioPlayback
from murmur.config import MurmurConfig
from murmur.events import EventBus, Events
from murmur.llm.base import LLMProvider
from murmur.llm.registry import get_llm_provider
from murmur.session import PipelineSessionState, PipelineState
from murmur.stt.base import STTProvider
from murmur.stt.registry import get_stt_provider
from murmur.tts.base import TTSProvider
from murmur.tts.registry import get_tts_provider

# Split LLM output into speakable sentence chunks
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|(?<=\n)\n+")


class PipelineSession:
    """End-to-end voice pipeline session.

    Usage:
        cfg = MurmurConfig.load("murmur.toml")
        async with PipelineSession(cfg) as session:
            await session.run()

    Or single-turn:
        result = await session.process_audio(audio_bytes)
    """

    def __init__(self, cfg: MurmurConfig) -> None:
        self.cfg = cfg
        self.state = PipelineSessionState(
            system_prompt=cfg.llm.system_prompt,
            max_history_turns=cfg.llm.max_history_turns,
        )
        self.bus = EventBus()

        self._stt: STTProvider | None = None
        self._llm: LLMProvider | None = None
        self._tts: TTSProvider | None = None
        self._capture = AudioCapture(cfg.audio)
        self._playback = AudioPlayback(cfg.audio)
        self._running = False

    async def __aenter__(self) -> "PipelineSession":
        await self._init_providers()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self._teardown_providers()

    async def _init_providers(self) -> None:
        self._stt = get_stt_provider(self.cfg.stt, self.cfg)
        self._llm = get_llm_provider(self.cfg.llm, self.cfg)
        self._tts = get_tts_provider(self.cfg.tts, self.cfg)
        await self._stt.setup()
        await self._llm.setup()
        await self._tts.setup()

    async def _teardown_providers(self) -> None:
        for provider in [self._stt, self._llm, self._tts]:
            if provider:
                await provider.teardown()

    # ── Single-turn processing ──────────────────────────────────────────────

    async def process_audio(self, audio: bytes) -> str:
        """Full pipeline turn: audio → transcript → LLM → TTS → playback.

        Returns the LLM response text.
        """
        assert self._stt and self._llm and self._tts

        # STT
        self.state.transition(PipelineState.TRANSCRIBING)
        await self.bus.emit(Events.STATE_CHANGE, PipelineState.TRANSCRIBING)

        stt_result = await self._stt.transcribe(audio, self.cfg.audio.sample_rate)
        transcript = stt_result.text.strip()

        if not transcript:
            self.state.transition(PipelineState.IDLE)
            return ""

        await self.bus.emit(Events.TRANSCRIPT, transcript)
        self.state.add_user(transcript)

        # LLM + TTS (streamed, interleaved)
        response_text = await self._stream_llm_to_tts()

        self.state.add_assistant(response_text)
        self.state.transition(PipelineState.IDLE)
        await self.bus.emit(Events.STATE_CHANGE, PipelineState.IDLE)

        return response_text

    async def process_text(self, text: str) -> str:
        """Text-in pipeline turn: text → LLM → TTS → playback."""
        assert self._llm and self._tts

        self.state.add_user(text)
        self.state.transition(PipelineState.THINKING)

        response_text = await self._stream_llm_to_tts()

        self.state.add_assistant(response_text)
        self.state.transition(PipelineState.IDLE)
        return response_text

    async def _stream_llm_to_tts(self) -> str:
        """Stream LLM tokens → buffer sentences → stream TTS in parallel."""
        assert self._llm and self._tts

        self.state.transition(PipelineState.THINKING)
        await self.bus.emit(Events.STATE_CHANGE, PipelineState.THINKING)

        messages = self.state.to_messages()
        full_response = ""
        sentence_buffer = ""
        tts_queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def _tts_worker() -> None:
            """Consume sentence queue and play audio."""
            self.state.transition(PipelineState.SPEAKING)
            while True:
                sentence = await tts_queue.get()
                if sentence is None:
                    break
                await self.bus.emit(Events.LLM_SENTENCE, sentence)
                async def _gen_chunks() -> AsyncIterator[object]:
                    async for chunk in self._tts.synthesize(sentence):  # type: ignore[union-attr]
                        yield chunk
                await self._playback.play_chunks(_gen_chunks())  # type: ignore[arg-type]

        tts_task = asyncio.create_task(_tts_worker())

        try:
            async for token in self._llm.stream(messages):
                full_response += token
                sentence_buffer += token
                await self.bus.emit(Events.LLM_TOKEN, token)

                # Flush complete sentences to TTS queue
                while True:
                    parts = _SENTENCE_SPLIT.split(sentence_buffer, maxsplit=1)
                    if len(parts) < 2:
                        break
                    sentence, sentence_buffer = parts
                    sentence = sentence.strip()
                    if len(sentence) >= self.cfg.pipeline.sentence_min_chars:
                        await tts_queue.put(sentence)
                    else:
                        sentence_buffer = sentence + " " + sentence_buffer

        finally:
            # Flush remaining buffer
            if sentence_buffer.strip():
                await tts_queue.put(sentence_buffer.strip())
            await tts_queue.put(None)  # signal done
            await tts_task

        await self.bus.emit(Events.LLM_DONE, full_response)
        return full_response

    # ── Continuous run loop ─────────────────────────────────────────────────

    async def run(self) -> None:
        """Main loop — push-to-talk or VAD depending on config."""
        self._running = True
        await self.bus.emit(Events.PIPELINE_START)
        self.state.transition(PipelineState.IDLE)

        try:
            if self.cfg.pipeline.mode == "push-to-talk":
                await self._run_push_to_talk()
            elif self.cfg.pipeline.mode == "vad":
                await self._run_vad()
            else:
                raise ValueError(f"Unknown pipeline mode: {self.cfg.pipeline.mode!r}")
        finally:
            self._running = False
            await self.bus.emit(Events.PIPELINE_STOP)

    async def _run_push_to_talk(self) -> None:
        from rich.console import Console
        console = Console()

        console.print("\n[bold green]Murmur[/] — push-to-talk mode")
        console.print("[dim]Press Enter to start recording, Enter again to stop. Ctrl+C to quit.[/]\n")

        while self._running:
            try:
                console.print("[bold cyan]▶ Press Enter to speak...[/]", end=" ")
                await asyncio.get_event_loop().run_in_executor(None, input)

                self.state.transition(PipelineState.LISTENING)
                await self.bus.emit(Events.LISTENING_START)
                console.print("[yellow]🎙  Recording... Press Enter to stop.[/]")

                stop_event = asyncio.Event()
                record_task = asyncio.create_task(
                    self._capture.record_push_to_talk(
                        stop_event, max_seconds=self.cfg.pipeline.max_recording_s
                    )
                )

                await asyncio.get_event_loop().run_in_executor(None, input)
                stop_event.set()
                audio = await record_task

                await self.bus.emit(Events.LISTENING_STOP)

                if not audio:
                    console.print("[dim]No audio captured.[/]")
                    continue

                console.print("[dim]Transcribing...[/]")
                response = await self.process_audio(audio)

                if response:
                    console.print(f"\n[bold]Assistant:[/] {response}\n")

            except KeyboardInterrupt:
                break
            except Exception as e:
                await self.bus.emit(Events.ERROR, e)
                console.print(f"[red]Error:[/] {e}")

    async def _run_vad(self) -> None:
        from rich.console import Console
        console = Console()

        console.print("\n[bold green]Murmur[/] — voice activation mode")
        console.print("[dim]Speak naturally. Say nothing to pause. Ctrl+C to quit.[/]\n")
        console.print("[dim]VAD mode: recording for up to 10s per utterance.[/]\n")

        while self._running:
            try:
                self.state.transition(PipelineState.LISTENING)
                await self.bus.emit(Events.LISTENING_START)

                audio = await self._capture.record_until_silence(
                    max_seconds=min(10.0, self.cfg.pipeline.max_recording_s)
                )

                await self.bus.emit(Events.LISTENING_STOP)

                if not audio or len(audio) < 1600:  # < 50ms at 16kHz
                    await asyncio.sleep(0.1)
                    continue

                response = await self.process_audio(audio)
                if response:
                    console.print(f"\n[bold]Assistant:[/] {response}\n")

            except KeyboardInterrupt:
                break
            except Exception as e:
                await self.bus.emit(Events.ERROR, e)
                console.print(f"[red]Error:[/] {e}")

    def clear_history(self) -> None:
        self.state.clear()
