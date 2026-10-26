"""PCM / WAV / base64 helpers. numpy only, so the API process stays torch-free."""

from __future__ import annotations

import base64
import io
import wave

import numpy as np

PCM_SAMPLE_WIDTH = 2  # s16le


def float_to_pcm16(samples: np.ndarray) -> bytes:
    """Clip float samples to [-1, 1] and encode them as little-endian int16."""
    clipped = np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def pcm16_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap mono s16le PCM in a RIFF/WAVE container."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(PCM_SAMPLE_WIDTH)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


def pcm_duration_seconds(pcm: bytes, sample_rate: int) -> float:
    return len(pcm) / PCM_SAMPLE_WIDTH / sample_rate


def encode_audio(pcm: bytes, sample_rate: int, output_format: str) -> str:
    """Return the base64 payload for ``output_format`` (``pcm`` or ``wav``)."""
    if output_format == "wav":
        return base64.b64encode(pcm16_to_wav(pcm, sample_rate)).decode("ascii")
    if output_format == "pcm":
        return base64.b64encode(pcm).decode("ascii")
    raise ValueError(f"unsupported output_format {output_format!r}")
