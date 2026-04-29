FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    portaudio19-dev \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for fast dependency resolution
RUN pip install uv

COPY pyproject.toml ./
COPY murmur/ ./murmur/

# Install core + local providers (no API keys needed)
RUN uv pip install --system ".[faster-whisper,kokoro,vad,edge-tts,ui]"

# Pre-download the default Whisper model so first run is instant
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base.en', device='cpu')" || true

COPY murmur.example.toml ./murmur.toml

EXPOSE 8765

ENTRYPOINT ["murmur"]
CMD ["run", "--mode", "push-to-talk"]
