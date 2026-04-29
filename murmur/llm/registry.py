"""LLM provider registry."""

from __future__ import annotations

from murmur.config import LLMConfig, MurmurConfig
from murmur.llm.base import LLMProvider

PROVIDER_INFO = {
    "openrouter": {
        "description": "100+ models via one API key. Recommended default.",
        "local": False,
        "requires_api_key": True,
        "env_key": "OPENROUTER_API_KEY",
        "signup": "https://openrouter.ai/keys",
        "free_models": [
            "meta-llama/llama-3.1-8b-instruct:free",
            "mistralai/mistral-7b-instruct:free",
            "google/gemma-2-9b-it:free",
        ],
    },
    "openai": {
        "description": "OpenAI GPT models (gpt-4o, gpt-4o-mini, gpt-3.5-turbo).",
        "local": False,
        "requires_api_key": True,
        "env_key": "OPENAI_API_KEY",
        "signup": "https://platform.openai.com/api-keys",
    },
    "anthropic": {
        "description": "Anthropic Claude models (claude-3-5-sonnet, claude-3-haiku).",
        "local": False,
        "requires_api_key": True,
        "env_key": "ANTHROPIC_API_KEY",
        "signup": "https://console.anthropic.com",
    },
    "ollama": {
        "description": "Run any GGUF model locally via Ollama. No API key.",
        "local": True,
        "requires_api_key": False,
        "install": "https://ollama.com",
    },
    "mock": {
        "description": "Mock provider for testing. Returns fixed text.",
        "local": True,
        "requires_api_key": False,
    },
}


def list_llm_providers() -> dict[str, dict]:
    return PROVIDER_INFO


def get_llm_provider(cfg: LLMConfig, murmur_cfg: MurmurConfig) -> LLMProvider:
    name = cfg.provider
    api_key = murmur_cfg.api_key(name)
    extra = cfg.config

    if name == "openrouter":
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not set.\n"
                "  1. Get your free key at https://openrouter.ai/keys\n"
                "  2. Set it: export OPENROUTER_API_KEY=sk-or-...\n"
                "  3. Free models available — no credit card needed for free tier."
            )
        from murmur.llm.openrouter_provider import OpenRouterProvider
        return OpenRouterProvider(
            api_key=api_key,
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            stream=cfg.stream,
            site_url=extra.get("site_url", "https://github.com/murmur-voice/murmur"),
            site_name=extra.get("site_name", "Murmur Voice"),
            fallback_model=extra.get("fallback_model"),
            timeout=extra.get("timeout", 120.0),
        )

    if name == "openai":
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set.")
        from murmur.llm.openai_provider import OpenAIProvider
        return OpenAIProvider(
            api_key=api_key,
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

    if name == "anthropic":
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set.")
        from murmur.llm.openai_provider import AnthropicProvider
        return AnthropicProvider(
            api_key=api_key,
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

    if name == "ollama":
        from murmur.llm.openai_provider import OllamaProvider
        return OllamaProvider(
            model=cfg.model or "llama3",
            base_url=extra.get("base_url", "http://localhost:11434"),
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

    if name == "mock":
        from murmur.llm.openai_provider import MockLLMProvider
        return MockLLMProvider(response=extra.get("response", "Mock response."))

    raise ValueError(f"Unknown LLM provider: {name!r}. Valid: {list(PROVIDER_INFO)}")
