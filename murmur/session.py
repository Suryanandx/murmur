"""Pipeline session state — conversation history and pipeline state machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Literal


class PipelineState(Enum):
    IDLE = auto()
    LISTENING = auto()
    TRANSCRIBING = auto()
    THINKING = auto()
    SPEAKING = auto()
    ERROR = auto()


@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class PipelineSessionState:
    state: PipelineState = PipelineState.IDLE
    history: list[Message] = field(default_factory=list)
    system_prompt: str = ""
    max_history_turns: int = 10

    def set_system(self, prompt: str) -> None:
        self.system_prompt = prompt

    def add_user(self, content: str) -> None:
        self.history.append(Message(role="user", content=content))
        self._trim()

    def add_assistant(self, content: str) -> None:
        self.history.append(Message(role="assistant", content=content))
        self._trim()

    def to_messages(self) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.extend({"role": m.role, "content": m.content} for m in self.history)
        return msgs

    def clear(self) -> None:
        self.history.clear()

    def _trim(self) -> None:
        max_msgs = self.max_history_turns * 2
        if len(self.history) > max_msgs:
            self.history = self.history[-max_msgs:]

    def transition(self, new_state: PipelineState) -> None:
        self.state = new_state
