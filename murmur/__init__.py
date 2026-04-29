"""Murmur — local-first agentic voice pipeline."""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["PipelineSession", "MurmurConfig"]


def __getattr__(name: str) -> object:
    if name == "PipelineSession":
        from murmur.pipeline import PipelineSession
        return PipelineSession
    if name == "MurmurConfig":
        from murmur.config import MurmurConfig
        return MurmurConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
