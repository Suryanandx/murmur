"""Abstract base class for all LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class LLMResponse:
    text: str
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    provider: str = ""


class LLMProvider(ABC):
    """All LLM providers must implement this interface.

    Providers receive OpenAI-format message dicts and return either a
    complete LLMResponse or a stream of string token chunks.
    """

    name: str = "base"
    requires_api_key: bool = True

    @abstractmethod
    async def complete(self, messages: list[dict[str, str]]) -> LLMResponse:
        """Return a complete response (non-streaming)."""
        ...

    @abstractmethod
    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """Yield token chunks as they arrive."""
        ...

    async def __aenter__(self) -> "LLMProvider":
        await self.setup()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.teardown()

    async def setup(self) -> None:
        pass

    async def teardown(self) -> None:
        pass
