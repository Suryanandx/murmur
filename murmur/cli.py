"""Murmur CLI — `murmur` entrypoint."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@click.group()
@click.version_option(package_name="murmur-voice")
def cli() -> None:
    """Murmur — local-first agentic voice pipeline.

    STT → LLM (OpenRouter / OpenAI / Anthropic / Ollama) → TTS

    Get started:\n
        murmur init          Create a config file\n
        murmur run           Start the voice pipeline\n
        murmur providers     List all available providers\n
        murmur chat          Text-based chat (no microphone needed)\n
    """


# ── murmur run ─────────────────────────────────────────────────────────────

@cli.command()
@click.option("--config", "-c", type=click.Path(exists=False), default=None, help="Path to murmur.toml")
@click.option("--stt", default=None, help="STT provider override (e.g. faster-whisper, groq, deepgram)")
@click.option("--llm", default=None, help="LLM provider override (e.g. openrouter, openai, ollama)")
@click.option("--tts", default=None, help="TTS provider override (e.g. kokoro, elevenlabs, edge-tts)")
@click.option("--model", default=None, help="LLM model override (e.g. mistralai/mistral-7b-instruct:free)")
@click.option("--mode", type=click.Choice(["push-to-talk", "vad"]), default=None)
@click.option("--dry-run", is_flag=True, help="Use mock providers (no hardware, no API keys)")
def run(
    config: Optional[str],
    stt: Optional[str],
    llm: Optional[str],
    tts: Optional[str],
    model: Optional[str],
    mode: Optional[str],
    dry_run: bool,
) -> None:
    """Start the voice pipeline."""
    from murmur.config import MurmurConfig
    from murmur.pipeline import PipelineSession

    cfg = MurmurConfig.load(config)

    if dry_run:
        cfg.stt.provider = "mock"  # type: ignore[assignment]
        cfg.llm.provider = "mock"  # type: ignore[assignment]
        cfg.tts.provider = "mock"  # type: ignore[assignment]
        console.print("[yellow]Dry-run mode: using mock providers.[/]")

    if stt:
        cfg.stt.provider = stt  # type: ignore[assignment]
    if llm:
        cfg.llm.provider = llm  # type: ignore[assignment]
    if tts:
        cfg.tts.provider = tts  # type: ignore[assignment]
    if model:
        cfg.llm.model = model
    if mode:
        cfg.pipeline.mode = mode  # type: ignore[assignment]

    _print_startup_banner(cfg)

    async def _run() -> None:
        async with PipelineSession(cfg) as session:
            await session.run()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye.[/]")


# ── murmur chat ────────────────────────────────────────────────────────────

@cli.command()
@click.option("--config", "-c", type=click.Path(), default=None)
@click.option("--llm", default=None, help="LLM provider")
@click.option("--model", default=None, help="Model name")
@click.option("--tts", default=None, help="TTS provider (omit to skip audio)")
@click.option("--no-tts", is_flag=True, help="Text-only, skip TTS")
def chat(
    config: Optional[str],
    llm: Optional[str],
    model: Optional[str],
    tts: Optional[str],
    no_tts: bool,
) -> None:
    """Text chat — no microphone needed. Great for testing LLM + TTS."""
    from murmur.config import MurmurConfig
    from murmur.pipeline import PipelineSession

    cfg = MurmurConfig.load(config)
    if llm:
        cfg.llm.provider = llm  # type: ignore[assignment]
    if model:
        cfg.llm.model = model
    if tts:
        cfg.tts.provider = tts  # type: ignore[assignment]
    if no_tts:
        cfg.tts.provider = "mock"  # type: ignore[assignment]

    console.print(Panel(
        f"[bold green]Murmur Chat[/]\n"
        f"LLM: [cyan]{cfg.llm.provider}[/] / [yellow]{cfg.llm.model}[/]\n"
        f"TTS: [cyan]{cfg.tts.provider}[/]\n"
        f"Type your message and press Enter. [dim]Ctrl+C to quit.[/]",
        title="Murmur"
    ))

    async def _run() -> None:
        async with PipelineSession(cfg) as session:
            while True:
                try:
                    text = await asyncio.get_event_loop().run_in_executor(None, lambda: input("\nYou: "))
                    if not text.strip():
                        continue
                    if text.strip().lower() in {"/quit", "/exit", "quit", "exit"}:
                        break
                    if text.strip() == "/clear":
                        session.clear_history()
                        console.print("[dim]Conversation cleared.[/]")
                        continue

                    console.print("[dim]Thinking...[/]")
                    response = await session.process_text(text)
                    console.print(f"\n[bold]Assistant:[/] {response}")
                except KeyboardInterrupt:
                    break

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass
    console.print("\n[dim]Goodbye.[/]")


# ── murmur providers ───────────────────────────────────────────────────────

@cli.command()
@click.option("--type", "ptype", type=click.Choice(["stt", "llm", "tts", "all"]), default="all")
def providers(ptype: str) -> None:
    """List all available STT, LLM, and TTS providers."""
    from murmur.stt.registry import list_stt_providers
    from murmur.llm.registry import list_llm_providers
    from murmur.tts.registry import list_tts_providers

    if ptype in ("stt", "all"):
        _print_provider_table("STT Providers", list_stt_providers(), "green")
    if ptype in ("llm", "all"):
        _print_provider_table("LLM Providers", list_llm_providers(), "blue")
    if ptype in ("tts", "all"):
        _print_provider_table("TTS Providers", list_tts_providers(), "magenta")


def _print_provider_table(title: str, providers_dict: dict, color: str) -> None:
    table = Table(title=title, border_style=color, show_lines=True)
    table.add_column("Name", style="bold")
    table.add_column("Description")
    table.add_column("Local", justify="center")
    table.add_column("API Key", justify="center")

    for name, info in providers_dict.items():
        local = "✓" if info.get("local") else "✗"
        key = "✗" if info.get("requires_api_key") else "✓"
        table.add_row(name, info.get("description", ""), local, key)

    console.print(table)


# ── murmur init ────────────────────────────────────────────────────────────

@cli.command()
@click.option("--output", "-o", default="murmur.toml", help="Output config file path")
@click.option("--preset", type=click.Choice(["local", "cloud", "minimal"]), default="local")
def init(output: str, preset: str) -> None:
    """Create a murmur.toml config file."""
    out = Path(output)
    if out.exists():
        if not click.confirm(f"{output} already exists. Overwrite?"):
            return

    presets = {
        "local": _LOCAL_CONFIG,
        "cloud": _CLOUD_CONFIG,
        "minimal": _MINIMAL_CONFIG,
    }

    out.write_text(presets[preset])
    console.print(f"[green]✓[/] Created [bold]{output}[/] ({preset} preset)")
    console.print("\nNext steps:")
    console.print(f"  1. Edit [bold]{output}[/] to set your API keys / preferences")
    console.print("  2. [bold]murmur run[/] to start the pipeline")


# ── murmur devices ─────────────────────────────────────────────────────────

@cli.command()
def devices() -> None:
    """List available audio input/output devices."""
    from murmur.audio.capture import AudioCapture

    devs = AudioCapture.list_devices()
    if not devs:
        console.print("[red]No audio devices found or sounddevice not installed.[/]")
        return

    table = Table(title="Audio Input Devices", border_style="cyan")
    table.add_column("Index", justify="right")
    table.add_column("Name")
    table.add_column("Channels", justify="right")
    for d in devs:
        table.add_row(str(d["index"]), d["name"], str(d["channels"]))
    console.print(table)


# ── murmur models ──────────────────────────────────────────────────────────

@cli.command()
@click.option("--tier", type=click.Choice(["free", "fast", "capable"]), default=None)
def models(tier: Optional[str]) -> None:
    """Show recommended OpenRouter models."""
    from murmur.llm.openrouter_provider import RECOMMENDED_MODELS

    tiers = [tier] if tier else list(RECOMMENDED_MODELS.keys())
    for t in tiers:
        table = Table(title=f"OpenRouter — {t.upper()} models", border_style="blue")
        table.add_column("Model ID")
        for m in RECOMMENDED_MODELS[t]:
            table.add_row(m)
        console.print(table)

    console.print(
        "\n[dim]Full model list: https://openrouter.ai/models\n"
        "Use with: murmur run --model <model-id>[/]"
    )


# ── helpers ─────────────────────────────────────────────────────────────────

def _print_startup_banner(cfg: object) -> None:
    from murmur.config import MurmurConfig
    assert isinstance(cfg, MurmurConfig)
    console.print(Panel(
        f"[bold green]Murmur[/] voice pipeline starting\n\n"
        f"  STT: [cyan]{cfg.stt.provider}[/]\n"
        f"  LLM: [cyan]{cfg.llm.provider}[/] / [yellow]{cfg.llm.model}[/]\n"
        f"  TTS: [cyan]{cfg.tts.provider}[/]\n"
        f"  Mode: [cyan]{cfg.pipeline.mode}[/]",
        title="🎙 Murmur"
    ))


# ── config presets ──────────────────────────────────────────────────────────

_MINIMAL_CONFIG = """\
# murmur.toml — minimal config (all defaults)
# Run: murmur run

[llm]
provider = "openrouter"
model = "mistralai/mistral-7b-instruct:free"
# Set your key: export OPENROUTER_API_KEY=sk-or-...
# Free key at: https://openrouter.ai/keys
"""

_LOCAL_CONFIG = """\
# murmur.toml — local-first preset
# STT: faster-whisper (local), LLM: openrouter, TTS: kokoro (local)

[pipeline]
mode = "push-to-talk"   # or "vad"
max_recording_s = 30.0
sentence_min_chars = 20

[audio]
sample_rate = 16000
# input_device = 0    # uncomment to set specific mic
# output_device = 0   # uncomment to set specific speaker

[stt]
provider = "faster-whisper"
language = "en"
[stt.config]
model = "base.en"       # tiny.en | base.en | small | medium | large-v3
device = "auto"         # cpu | cuda | auto

[llm]
provider = "openrouter"
model = "mistralai/mistral-7b-instruct:free"
system_prompt = "You are a helpful voice assistant. Be concise."
max_history_turns = 10
temperature = 0.7
max_tokens = 512
# Set: export OPENROUTER_API_KEY=sk-or-...
# Free key: https://openrouter.ai/keys
[llm.config]
# fallback_model = "google/gemma-2-9b-it:free"  # retry on overload

[tts]
provider = "kokoro"
voice = "af_sarah"      # see: murmur providers --type tts
speed = 1.0
"""

_CLOUD_CONFIG = """\
# murmur.toml — cloud preset
# STT: groq (fast), LLM: openrouter, TTS: elevenlabs (best quality)

[pipeline]
mode = "push-to-talk"
max_recording_s = 30.0

[audio]
sample_rate = 16000

[stt]
provider = "groq"
language = "en"
[stt.config]
model = "whisper-large-v3-turbo"
# Set: export GROQ_API_KEY=...

[llm]
provider = "openrouter"
model = "anthropic/claude-3-haiku"
system_prompt = "You are a helpful voice assistant. Be concise."
max_history_turns = 10
temperature = 0.7
max_tokens = 512
# Set: export OPENROUTER_API_KEY=sk-or-...
[llm.config]
fallback_model = "mistralai/mistral-7b-instruct:free"

[tts]
provider = "elevenlabs"
speed = 1.0
[tts.config]
voice_id = "JBFqnCBsd6RMkjVDRZzb"
model = "eleven_turbo_v2_5"
# Set: export ELEVENLABS_API_KEY=...
"""


if __name__ == "__main__":
    cli()
