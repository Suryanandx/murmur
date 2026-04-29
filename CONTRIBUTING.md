# Contributing to Murmur

## Ways to contribute

- **Bug reports** — open an issue with the bug report template
- **New providers** — add a new STT, LLM, or TTS integration (most impactful!)
- **Documentation** — improve docs, add examples, fix typos
- **Tests** — increase coverage, add integration tests

## Development setup

```bash
git clone https://github.com/murmur-voice/murmur
cd murmur

# Install uv (fast Python package manager)
pip install uv

# Install all deps including dev tools
uv sync --dev

# Install pre-commit hooks
uv run pre-commit install

# Verify everything works
uv run pytest tests/unit/
uv run murmur --version
```

## Adding a new provider

Providers live in `murmur/stt/`, `murmur/llm/`, or `murmur/tts/`. Each provider:

1. Inherits from the corresponding ABC (`STTProvider`, `LLMProvider`, `TTSProvider`)
2. Is registered in the matching `registry.py`
3. Has a matching entry in `PROVIDER_INFO`
4. Has at least one unit test (using the mock pattern)

### Example — adding a new STT provider

```python
# murmur/stt/my_provider.py
from murmur.stt.base import STTProvider, STTResult

class MyProvider(STTProvider):
    name = "my-provider"
    requires_api_key = True
    local = False

    def __init__(self, api_key: str, **_):
        self.api_key = api_key

    async def transcribe(self, audio: bytes, sample_rate: int = 16000) -> STTResult:
        # call your API here
        return STTResult(text="...", provider=self.name)
```

Then register it in `murmur/stt/registry.py` — follow the existing pattern.

## Code style

```bash
uv run ruff check murmur/       # linting
uv run black murmur/            # formatting
uv run mypy murmur/             # type checking
uv run pytest tests/unit/ -v    # tests
```

Or run all at once:

```bash
make check
```

## Commit style

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add Cartesia TTS provider
fix: handle OpenRouter 503 with friendly error message
docs: add edge-tts example to README
chore: bump httpx to 0.28
```

## Pull request checklist

- [ ] `make check` passes
- [ ] Tests added for new code
- [ ] `CHANGELOG.md` updated
- [ ] Provider docs updated (if adding a provider)
- [ ] No API keys or secrets in the diff

## Questions

Open a [GitHub Discussion](https://github.com/murmur-voice/murmur/discussions) — not an issue — for how-to questions.
