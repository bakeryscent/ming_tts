"""Runtime configuration, read once from the environment (see docs/API.md)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_ID = "inclusionAI/Ming-3.0-tts-0.5B"
SERVICE_NAME = "ming-tts-api"

# Ming's decoder runs at 44.1 kHz; requests may ask for 24 kHz to halve the payload.
NATIVE_SAMPLE_RATE = 44100
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int, minimum: int) -> int:
    raw = os.environ.get(name, "").strip()
    value = int(raw) if raw else default
    if value < minimum:
        raise RuntimeError(f"{name} must be >= {minimum}, got {value}")
    return value


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    value = float(raw) if raw else default
    if value <= 0:
        raise RuntimeError(f"{name} must be > 0, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    api_token: str
    model_id: str = DEFAULT_MODEL_ID
    ming_dir: Path = Path("/app/ming")
    voices_dir: Path = Path("/workspace/voices")
    # Voices shipped with the image; seeds an empty VOICES_DIR and holds builtin_zh.json.
    bundled_voices_dir: Path = PROJECT_DIR / "voices"
    num_workers: int = 3
    max_queue: int = 16
    request_timeout_s: float = 90.0
    max_text_chars: int = 1200
    text_normalization: bool = True
    device: str = "cuda:0"
    # FAKE_ENGINE=1: workers return a synthetic tone instead of loading Ming.
    # Used by the test-suite to exercise the pool on machines without a GPU.
    fake_engine: bool = False

    @property
    def stuck_job_s(self) -> float:
        """Age at which a running job is declared hung and its worker killed.

        A request that times out (504) keeps its worker busy until the
        synthesis finishes, because a CUDA decode loop cannot be interrupted
        cooperatively. Only a job running far past the client deadline is
        treated as a hang worth a model reload.
        """
        return self.request_timeout_s * 2 + 30

    @classmethod
    def from_env(cls) -> "Settings":
        token = os.environ.get("API_TOKEN", "").strip()
        if not token:
            raise RuntimeError("API_TOKEN env var is required")
        return cls(
            api_token=token,
            model_id=os.environ.get("MODEL_ID", "").strip() or DEFAULT_MODEL_ID,
            ming_dir=Path(os.environ.get("MING_DIR", "").strip() or "/app/ming"),
            voices_dir=Path(os.environ.get("VOICES_DIR", "").strip() or "/workspace/voices"),
            num_workers=_env_int("NUM_WORKERS", 3, minimum=1),
            max_queue=_env_int("MAX_QUEUE", 16, minimum=0),
            request_timeout_s=_env_float("REQUEST_TIMEOUT_S", 90.0),
            max_text_chars=_env_int("MAX_TEXT_CHARS", 1200, minimum=1),
            text_normalization=_env_bool("TEXT_NORMALIZATION", True),
            fake_engine=_env_bool("FAKE_ENGINE", False),
        )
