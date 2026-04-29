# Changelog

All notable changes to Murmur are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [SemVer](https://semver.org/)

## [Unreleased]

## [0.1.0] - 2026-04-29

### Added
- Core streaming pipeline: STT → LLM → TTS with sentence-boundary buffering
- STT providers: faster-whisper (local), openai, groq, deepgram, assemblyai, mock
- LLM providers: openrouter (primary), openai, anthropic, ollama, mock
- TTS providers: kokoro (local), piper (local), elevenlabs, openai, cartesia, edge-tts (free), mock
- Push-to-talk and VAD pipeline modes
- `murmur run`, `murmur chat`, `murmur providers`, `murmur models`, `murmur devices`, `murmur init` CLI commands
- TOML config with env var overrides
- Secure API key resolution (env → ~/.config/murmur/credentials)
- Docker image with CPU and GPU variants
- Apache 2.0 license
- Full unit test suite for all mock providers
- Custom SVG logo, icon, architecture diagram, and provider badges
