"""Simplest possible Murmur usage — text chat with OpenRouter LLM."""

import asyncio
import os
from murmur.config import MurmurConfig
from murmur.pipeline import PipelineSession


async def main() -> None:
    cfg = MurmurConfig()
    cfg.llm.provider = "openrouter"  # type: ignore[assignment]
    cfg.llm.model = "mistralai/mistral-7b-instruct:free"
    cfg.tts.provider = "mock"   # type: ignore[assignment]  # skip audio for this example
    cfg.stt.provider = "mock"   # type: ignore[assignment]

    # Requires OPENROUTER_API_KEY in environment
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Set OPENROUTER_API_KEY to run this example.")
        print("Get a free key at https://openrouter.ai/keys")
        return

    async with PipelineSession(cfg) as session:
        print("Murmur basic chat (type 'quit' to exit)\n")
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in {"quit", "exit"}:
                break
            response = await session.process_text(user_input)
            print(f"Assistant: {response}\n")


if __name__ == "__main__":
    asyncio.run(main())
