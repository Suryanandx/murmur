"""Lightweight asyncio event bus for pipeline stage communication."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine


@dataclass
class Event:
    name: str
    data: Any = None


EventHandler = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = {}
        self._queue: asyncio.Queue[Event] = asyncio.Queue()

    def on(self, event_name: str, handler: EventHandler) -> None:
        self._handlers.setdefault(event_name, []).append(handler)

    async def emit(self, name: str, data: Any = None) -> None:
        event = Event(name=name, data=data)
        handlers = self._handlers.get(name, []) + self._handlers.get("*", [])
        for handler in handlers:
            await handler(event)
        await self._queue.put(event)

    async def next(self) -> Event:
        return await self._queue.get()


# Standard event names
class Events:
    PIPELINE_START = "pipeline.start"
    PIPELINE_STOP = "pipeline.stop"
    LISTENING_START = "listening.start"
    LISTENING_STOP = "listening.stop"
    TRANSCRIPT = "transcript"              # data: str (final transcript)
    TRANSCRIPT_PARTIAL = "transcript.partial"  # data: str (streaming partial)
    LLM_TOKEN = "llm.token"               # data: str (streaming token)
    LLM_SENTENCE = "llm.sentence"         # data: str (full sentence for TTS)
    LLM_DONE = "llm.done"                 # data: str (full response)
    TTS_START = "tts.start"
    TTS_CHUNK = "tts.chunk"               # data: bytes (audio)
    TTS_DONE = "tts.done"
    STATE_CHANGE = "state.change"          # data: PipelineState
    ERROR = "error"                        # data: Exception
