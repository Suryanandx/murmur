"""Audio capture, playback, and VAD."""

__all__ = ["AudioCapture", "AudioPlayback"]


def __getattr__(name: str) -> object:
    if name == "AudioCapture":
        from murmur.audio.capture import AudioCapture
        return AudioCapture
    if name == "AudioPlayback":
        from murmur.audio.playback import AudioPlayback
        return AudioPlayback
    raise AttributeError(f"module __name__ has no attribute {name!r}")
