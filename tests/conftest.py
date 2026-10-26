"""Shared fixtures. Tests never load torch: the pool runs FAKE_ENGINE workers.

Engine tests that need Ming's ``tts_tasks`` read the repo from ``MING_DIR``
and are skipped when it is not set or does not exist.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.helpers import ROOT, TOKEN

# app.main builds a module-level app from the environment at import time.
os.environ.setdefault("API_TOKEN", TOKEN)
os.environ.setdefault("FAKE_ENGINE", "1")


def ming_dir_or_none() -> Path | None:
    raw = os.environ.get("MING_DIR", "").strip()
    if raw and (Path(raw) / "tts_tasks.py").is_file():
        return Path(raw)
    return None


@pytest.fixture
def ming_dir() -> Path:
    path = ming_dir_or_none()
    if path is None:
        pytest.skip("MING_DIR does not point at a Ming-3.0-tts checkout")
    return path


@pytest.fixture
def voices_dir(tmp_path: Path) -> Path:
    return tmp_path / "voices"


@pytest.fixture
def registry(voices_dir: Path):
    from app.voices import VoiceRegistry

    reg = VoiceRegistry(voices_dir, ROOT / "voices")
    reg.seed()
    return reg
