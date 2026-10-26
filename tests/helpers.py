"""Test helpers shared by several modules."""

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
TOKEN = "test-token"


def wav_bytes(seconds: float, sample_rate: int = 24000, channels: int = 1,
              fmt: str = "WAV") -> bytes:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    tone = 0.2 * np.sin(2 * np.pi * 330 * t)
    data = np.stack([tone] * channels, axis=1) if channels > 1 else tone
    buffer = io.BytesIO()
    sf.write(buffer, data, sample_rate, format=fmt)
    return buffer.getvalue()


def auth(token: str = TOKEN) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def wait_for(predicate, timeout: float = 30.0, interval: float = 0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError("condition not met in time")
