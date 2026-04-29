"""OpenRouter LLM provider — access 100+ models via one API key.

OpenRouter is the primary and recommended LLM provider for Murmur.
It supports every major model (GPT-4o, Claude, Gemini, Llama, Mistral, etc.)
through a single OpenAI-compatible API endpoint.

Free models (as of 2024): meta-llama/llama-3-8b-instruct:free,
  mistralai/mistral-7b-instruct:free, google/gemma-2-9b-it:free

Get your key at: https://openrouter.ai/keys
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

import httpx

from murmur.llm.base import LLMProvider, LLMResponse

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)

# Popular free / cheap models for quick start
RECOMMENDED_MODELS = {
    "free": [
        "meta-llama/llama-3.1-8b-instruct:free",
        "mistralai/mistral-7b-instruct:free",
        "google/gemma-2-9b-it:free",
        "qwen/qwen-2-7b-instruct:free",
    ],
    "fast": [
        "groq/llama3-8b-8192",          # via Groq on OpenRouter
        "anthropic/claude-3-haiku",
        "openai/gpt-4o-mini",
        "mistralai/mistral-small",
    ],
    "capable": [
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-4o",
        "google/gemini-pro-1.5",
        "meta-llama/llama-3.1-70b-instruct",
    ],
}


class OpenRouterProvider(LLMProvider):
    """OpenRouter — the primary LLM provider for Murmur.

    Supports 100+ models via a single API key. Uses the OpenAI-compatible
    REST API with server-sent events for streaming.

    Requires: OPENROUTER_API_KEY
    Sign up: https://openrouter.ai

    Config options (in murmur.toml [llm.config]):
      site_url: str   — your app URL (shown in OpenRouter dashboard)
      site_name: str  — your app name
      fallback_model: str — fallback if primary model is unavailable
    """

    name = "openrouter"
    requires_api_key = True

    def __init__(
        self,
        api_key: str,
        model: str = "mistralai/mistral-7b-instruct:free",
        temperature: float = 0.7,
        max_tokens: int = 512,
        stream: bool = True,
        site_url: str = "https://github.com/murmur-voice/murmur",
        site_name: str = "Murmur Voice",
        fallback_model: str | None = None,
        timeout: float = 120.0,
        **_: Any,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream
        self.site_url = site_url
        self.site_name = site_name
        self.fallback_model = fallback_model
        self._timeout = httpx.Timeout(connect=10.0, read=timeout, write=10.0, pool=10.0)
        self._client: httpx.AsyncClient | None = None

    async def setup(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=OPENROUTER_BASE_URL,
            headers=self._headers(),
            timeout=self._timeout,
        )

    async def teardown(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name,
            "Content-Type": "application/json",
        }

    def _payload(self, messages: list[dict[str, str]], stream: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
        if self.fallback_model:
            payload["models"] = [self.model, self.fallback_model]
            payload.pop("model")
        return payload

    async def complete(self, messages: list[dict[str, str]]) -> LLMResponse:
        client = self._client or httpx.AsyncClient(
            base_url=OPENROUTER_BASE_URL,
            headers=self._headers(),
            timeout=self._timeout,
        )
        try:
            resp = await client.post("/chat/completions", json=self._payload(messages, stream=False))
            resp.raise_for_status()
            data = resp.json()

            # OpenRouter may return errors in 200 responses
            if "error" in data:
                raise RuntimeError(
                    f"OpenRouter error: {data['error'].get('message', data['error'])}"
                )

            choice = data["choices"][0]
            usage = data.get("usage", {})
            return LLMResponse(
                text=choice["message"]["content"].strip(),
                model=data.get("model", self.model),
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                provider=self.name,
            )
        except httpx.HTTPStatusError as e:
            self._raise_friendly(e)
            raise
        finally:
            if not self._client:
                await client.aclose()

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:  # type: ignore[override]
        client = self._client or httpx.AsyncClient(
            base_url=OPENROUTER_BASE_URL,
            headers=self._headers(),
            timeout=self._timeout,
        )
        try:
            async with client.stream(
                "POST", "/chat/completions", json=self._payload(messages, stream=True)
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    raw = line[6:].strip()
                    if raw == "[DONE]":
                        break
                    try:
                        import json
                        chunk = json.loads(raw)
                    except Exception:
                        continue

                    if "error" in chunk:
                        raise RuntimeError(
                            f"OpenRouter stream error: {chunk['error'].get('message', chunk['error'])}"
                        )

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    token = delta.get("content")
                    if token:
                        yield token
        except httpx.HTTPStatusError as e:
            self._raise_friendly(e)
            raise
        finally:
            if not self._client:
                await client.aclose()

    @staticmethod
    def _raise_friendly(e: httpx.HTTPStatusError) -> None:
        status = e.response.status_code
        if status == 401:
            raise PermissionError(
                "OpenRouter API key is invalid or missing. "
                "Get your key at https://openrouter.ai/keys and set OPENROUTER_API_KEY."
            )
        if status == 402:
            raise RuntimeError(
                "OpenRouter account has no credits. "
                "Add credits at https://openrouter.ai/credits or switch to a free model "
                "(e.g. mistralai/mistral-7b-instruct:free)."
            )
        if status == 429:
            raise RuntimeError("OpenRouter rate limit hit. Wait a moment and retry.")
        if status == 503:
            raise RuntimeError(
                "The requested model is overloaded. "
                "Try a fallback_model in your config or choose a different model."
            )
