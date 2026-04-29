"""Unit tests for PipelineSession using mock providers."""

import asyncio
import pytest

from murmur.config import MurmurConfig, STTConfig, LLMConfig, TTSConfig
from murmur.pipeline import PipelineSession
from murmur.session import PipelineState


@pytest.fixture
def mock_config() -> MurmurConfig:
    cfg = MurmurConfig()
    cfg.stt.provider = "mock"  # type: ignore[assignment]
    cfg.llm.provider = "mock"  # type: ignore[assignment]
    cfg.tts.provider = "mock"  # type: ignore[assignment]
    cfg.llm.config["response"] = "The sky is blue."
    cfg.stt.config["response"] = "What color is the sky?"
    return cfg


@pytest.mark.asyncio
async def test_process_text(mock_config: MurmurConfig) -> None:
    async with PipelineSession(mock_config) as session:
        response = await session.process_text("Hello")
        assert isinstance(response, str)
        assert len(response) > 0


@pytest.mark.asyncio
async def test_process_audio(mock_config: MurmurConfig) -> None:
    fake_audio = b"\x00\x00" * 8000  # 0.5s of silence at 16kHz
    async with PipelineSession(mock_config) as session:
        response = await session.process_audio(fake_audio)
        assert isinstance(response, str)


@pytest.mark.asyncio
async def test_conversation_history(mock_config: MurmurConfig) -> None:
    async with PipelineSession(mock_config) as session:
        await session.process_text("First message")
        await session.process_text("Second message")
        assert len(session.state.history) == 4  # 2 user + 2 assistant


@pytest.mark.asyncio
async def test_clear_history(mock_config: MurmurConfig) -> None:
    async with PipelineSession(mock_config) as session:
        await session.process_text("Hello")
        session.clear_history()
        assert len(session.state.history) == 0


@pytest.mark.asyncio
async def test_state_transitions(mock_config: MurmurConfig) -> None:
    states: list[PipelineState] = []

    async with PipelineSession(mock_config) as session:
        from murmur.events import Events

        async def _on_state(event: object) -> None:
            from murmur.events import Event
            assert isinstance(event, Event)
            states.append(event.data)

        session.bus.on(Events.STATE_CHANGE, _on_state)
        await session.process_text("Test")

    assert PipelineState.THINKING in states
    assert PipelineState.IDLE in states


@pytest.mark.asyncio
async def test_event_transcript_emitted(mock_config: MurmurConfig) -> None:
    transcripts: list[str] = []
    fake_audio = b"\x00\x00" * 8000

    async with PipelineSession(mock_config) as session:
        from murmur.events import Events

        async def _on_transcript(event: object) -> None:
            from murmur.events import Event
            assert isinstance(event, Event)
            transcripts.append(event.data)

        session.bus.on(Events.TRANSCRIPT, _on_transcript)
        await session.process_audio(fake_audio)

    assert len(transcripts) == 1
    assert isinstance(transcripts[0], str)
