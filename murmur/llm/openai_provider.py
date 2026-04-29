"""OpenAI and Ollama (local) LLM providers."""

from __future__ import annotations

from typing import Any, AsyncIterator

from murmur.llm.base import LLMProvider, LLMResponse


class OpenAIProvider(LLMProvider):
    """OpenAI GPT models (gpt-4o, gpt-4o-mini, gpt-3.5-turbo, etc.)

    Requires: OPENAI_API_KEY
    """

    name = "openai"
    requires_api_key = True

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 512,
        **_: Any,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: Any = None

    async def setup(self) -> None:
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(api_key=self.api_key)

    async def teardown(self) -> None:
        if self._client:
            await self._client.close()

    async def complete(self, messages: list[dict[str, str]]) -> LLMResponse:
        if not self._client:
            await self.setup()
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=False,
        )
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            model=resp.model,
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
            provider=self.name,
        )

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:  # type: ignore[override]
        if not self._client:
            await self.setup()
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        async for chunk in response:
            token = chunk.choices[0].delta.content
            if token:
                yield token


class AnthropicProvider(LLMProvider):
    """Anthropic Claude models (claude-3-5-sonnet, claude-3-haiku, etc.)

    Requires: ANTHROPIC_API_KEY
    Install: pip install murmur-voice[anthropic]
    """

    name = "anthropic"
    requires_api_key = True

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-haiku-20240307",
        temperature: float = 0.7,
        max_tokens: int = 512,
        **_: Any,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: Any = None

    async def setup(self) -> None:
        try:
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic not installed. Run: pip install murmur-voice[anthropic]")

    async def teardown(self) -> None:
        if self._client:
            await self._client.close()

    def _split_messages(self, messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
        system = ""
        chat_msgs = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                chat_msgs.append(m)
        return system, chat_msgs

    async def complete(self, messages: list[dict[str, str]]) -> LLMResponse:
        if not self._client:
            await self.setup()
        system, chat = self._split_messages(messages)
        resp = await self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=chat,  # type: ignore[arg-type]
        )
        return LLMResponse(
            text=resp.content[0].text,
            model=resp.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            provider=self.name,
        )

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:  # type: ignore[override]
        if not self._client:
            await self.setup()
        system, chat = self._split_messages(messages)
        async with self._client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=chat,  # type: ignore[arg-type]
        ) as s:
            async for token in s.text_stream:
                yield token


class OllamaProvider(LLMProvider):
    """Ollama local LLM — run any GGUF model on your machine.

    Requires: Ollama installed and running (https://ollama.com)
    Models: llama3, mistral, phi3, gemma2, qwen2, etc.
    """

    name = "ollama"
    requires_api_key = False

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 512,
        **_: Any,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: Any = None

    async def setup(self) -> None:
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(
            api_key="ollama",
            base_url=f"{self.base_url}/v1",
        )

    async def teardown(self) -> None:
        if self._client:
            await self._client.close()

    async def complete(self, messages: list[dict[str, str]]) -> LLMResponse:
        if not self._client:
            await self.setup()
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            model=self.model,
            provider=self.name,
        )

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:  # type: ignore[override]
        if not self._client:
            await self.setup()
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        async for chunk in response:
            token = chunk.choices[0].delta.content
            if token:
                yield token


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    name = "mock"
    requires_api_key = False

    def __init__(self, response: str = "This is a mock LLM response for testing.", **_: Any) -> None:
        self.response = response

    async def complete(self, messages: list[dict[str, str]]) -> LLMResponse:
        return LLMResponse(text=self.response, model="mock", provider=self.name)

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:  # type: ignore[override]
        for word in self.response.split():
            yield word + " "
