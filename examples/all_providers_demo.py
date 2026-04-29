"""Demo that lists all providers and their status (installed / missing deps)."""

import importlib
import sys


def check_import(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


PROVIDER_CHECKS = {
    "STT": {
        "faster-whisper": ("faster_whisper", "local"),
        "openai (whisper)": ("openai", "cloud"),
        "groq": ("groq", "cloud"),
        "deepgram": ("deepgram", "cloud"),
        "assemblyai": ("assemblyai", "cloud"),
    },
    "LLM": {
        "openrouter": ("openai", "cloud"),   # uses openai SDK
        "openai": ("openai", "cloud"),
        "anthropic": ("anthropic", "cloud"),
        "ollama": ("openai", "local"),        # uses openai SDK pointed at localhost
    },
    "TTS": {
        "kokoro": ("kokoro", "local"),
        "piper": ("piper", "local"),
        "elevenlabs": ("elevenlabs", "cloud"),
        "openai tts": ("openai", "cloud"),
        "cartesia": ("cartesia", "cloud"),
        "edge-tts (free)": ("edge_tts", "free"),
    },
}

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def main() -> None:
    print(f"\n{BOLD}Murmur Provider Status{RESET}\n{'─' * 44}")

    for category, providers in PROVIDER_CHECKS.items():
        print(f"\n{CYAN}{BOLD}{category}{RESET}")
        for name, (module, tier) in providers.items():
            installed = check_import(module)
            status = f"{GREEN}✓ installed{RESET}" if installed else f"{RED}✗ not installed{RESET}"
            tier_color = GREEN if tier == "local" else (YELLOW if tier == "free" else "\033[94m")
            print(f"  {name:<22} {status:<30} {tier_color}[{tier}]{RESET}")

    print(f"\n{'─' * 44}")
    print(f"Install all:  {CYAN}pip install murmur-voice[all]{RESET}")
    print(f"Local only:   {CYAN}pip install murmur-voice[local]{RESET}")
    print(f"Cloud only:   {CYAN}pip install murmur-voice[cloud]{RESET}")
    print(f"\nFree OpenRouter key: {CYAN}https://openrouter.ai/keys{RESET}\n")


if __name__ == "__main__":
    main()
