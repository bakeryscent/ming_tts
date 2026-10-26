"""Voice registry: Ming built-in ("IP") voices and reference-audio clones.

Clones live in ``VOICES_DIR`` as ``<id>.wav`` plus a ``<id>.json`` sidecar
(``prompt_text``, ``language``, ``created_at``, ``prompt_seconds``). The
sidecar is written last, so a voice only becomes visible once both files are
complete.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import secrets
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf

from app.errors import BadRequest, Conflict, MethodNotAllowed, NotFound

log = logging.getLogger("ming-tts-api.voices")

LANGUAGES = ("en", "zh", "de", "es", "fr", "ko", "ja")
# Languages beyond Chinese/English; Ming needs an explicit language_id for these.
MULTILINGUAL_LANGUAGES = frozenset({"de", "es", "fr", "ko", "ja"})

CLONE_MIN_SECONDS = 1.0
CLONE_MAX_SECONDS = 20.0
MAX_PROMPT_TEXT_CHARS = 1000
ZH_CATALOG_FILE = "builtin_zh.json"

# English premium voices from the Ming cookbook (instruction_en.md). Ming's
# tts_tasks.EN_IP_VOICES lists four of them; US_Stella is documented in the
# cookbook and README but missing from that tuple.
EN_BUILTIN_VOICES: dict[str, dict[str, str]] = {
    "US_Emma": {"gender": "female", "accent": "American",
                "description": "assistants, narration, conversational product voice"},
    "US_James": {"gender": "male", "accent": "American",
                 "description": "narration, explainer and documentary reads"},
    "US_Stella": {"gender": "female", "accent": "American",
                  "description": "narration, conversational, warm, youthful"},
    "UK_George": {"gender": "male", "accent": "British",
                  "description": "audiobooks, formal announcements"},
    "UK_Charlotte": {"gender": "female", "accent": "British",
                     "description": "audiobooks, measured narration"},
}

_CLONE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_CJK_RE = re.compile(r"[㐀-䶿一-鿿]")


def detect_language(text: str) -> str:
    """Chinese if the text contains any CJK ideograph, English otherwise.

    The same test Ming's frontend uses (``text_segment_cut.is_chinese``).
    """
    return "zh" if _CJK_RE.search(text) else "en"


@dataclass(frozen=True)
class Voice:
    id: str
    kind: Literal["builtin", "clone"]
    language: str
    gender: str | None = None
    accent: str | None = None
    description: str | None = None
    emotion: str | None = None
    prompt_seconds: float | None = None
    created_at: str | None = None
    wav_path: Path | None = None
    prompt_text: str | None = None

    def public(self) -> dict:
        """The ``/v1/voices`` representation (no file paths or transcripts)."""
        fields = {
            "id": self.id, "kind": self.kind, "language": self.language,
            "gender": self.gender, "accent": self.accent,
            "description": self.description, "emotion": self.emotion,
            "prompt_seconds": self.prompt_seconds, "created_at": self.created_at,
        }
        return {key: value for key, value in fields.items() if value is not None}


class VoiceRegistry:
    def __init__(self, voices_dir: Path, bundled_dir: Path) -> None:
        self.voices_dir = Path(voices_dir)
        self.bundled_dir = Path(bundled_dir)
        self._zh_catalog = self._load_zh_catalog()
        self._lock = threading.Lock()

    # -- setup ---------------------------------------------------------------
    def _load_zh_catalog(self) -> dict[str, dict]:
        """Chinese IP voices, generated from the Ming cookbook; optional."""
        path = self.bundled_dir / ZH_CATALOG_FILE
        if not path.is_file():
            return {}
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def seed(self) -> int:
        """Copy the bundled preset clones into an empty ``voices_dir``.

        Returns the number of voices copied. A directory that already holds
        any ``.wav`` is left alone, so deleted presets stay deleted.
        """
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        if self.voices_dir.resolve() == self.bundled_dir.resolve():
            return 0
        if any(self.voices_dir.glob("*.wav")):
            return 0
        copied = 0
        for wav in sorted(self.bundled_dir.glob("*.wav")):
            sidecar = wav.with_suffix(".json")
            if not sidecar.is_file():
                continue
            shutil.copyfile(wav, self.voices_dir / wav.name)
            shutil.copyfile(sidecar, self.voices_dir / sidecar.name)
            copied += 1
        return copied

    # -- lookup --------------------------------------------------------------
    def _paths(self, voice_id: str) -> tuple[Path, Path]:
        return self.voices_dir / f"{voice_id}.wav", self.voices_dir / f"{voice_id}.json"

    def _load_clone(self, voice_id: str) -> Voice | None:
        if not _CLONE_ID_RE.match(voice_id):
            return None
        wav_path, json_path = self._paths(voice_id)
        if not (wav_path.is_file() and json_path.is_file()):
            return None
        try:
            with json_path.open(encoding="utf-8") as handle:
                meta = json.load(handle)
            return Voice(
                id=voice_id, kind="clone", language=str(meta.get("language", "en")),
                prompt_seconds=meta.get("prompt_seconds"), created_at=meta.get("created_at"),
                wav_path=wav_path, prompt_text=str(meta["prompt_text"]),
            )
        except (OSError, ValueError, KeyError) as error:
            log.warning("voice_sidecar_unreadable", extra={"voice": voice_id, "error": repr(error)})
            return None

    @staticmethod
    def _builtin(voice_id: str, language: str, meta: dict) -> Voice:
        return Voice(id=voice_id, kind="builtin", language=language,
                     gender=meta.get("gender"), accent=meta.get("accent"),
                     description=meta.get("description"), emotion=meta.get("emotion"))

    def is_builtin(self, voice_id: str) -> bool:
        return voice_id in EN_BUILTIN_VOICES or voice_id in self._zh_catalog

    def get(self, voice_id: str) -> Voice:
        if voice_id in EN_BUILTIN_VOICES:
            return self._builtin(voice_id, "en", EN_BUILTIN_VOICES[voice_id])
        clone = self._load_clone(voice_id)
        if clone is not None:
            return clone
        if voice_id in self._zh_catalog:
            return self._builtin(voice_id, "zh", self._zh_catalog[voice_id])
        raise NotFound(f"voice {voice_id!r} not found")

    def list(self) -> list[Voice]:
        voices = [self._builtin(vid, "en", meta) for vid, meta in EN_BUILTIN_VOICES.items()]
        voices += [self._builtin(vid, "zh", meta) for vid, meta in self._zh_catalog.items()]
        if self.voices_dir.is_dir():
            for sidecar in sorted(self.voices_dir.glob("*.json")):
                clone = self._load_clone(sidecar.stem)
                if clone is not None:
                    voices.append(clone)
        return voices

    def clone_count(self) -> int:
        return sum(1 for voice in self.list() if voice.kind == "clone")

    # -- mutation ------------------------------------------------------------
    def create_clone(
        self,
        audio: bytes,
        prompt_text: str,
        language: str = "auto",
        voice_id: str | None = None,
    ) -> Voice:
        """Validate an uploaded reference and store it as a clone voice."""
        prompt_text = (prompt_text or "").strip()
        if not prompt_text:
            raise BadRequest("prompt_text is required (exact transcript of the audio)")
        if len(prompt_text) > MAX_PROMPT_TEXT_CHARS:
            raise BadRequest(f"prompt_text must be at most {MAX_PROMPT_TEXT_CHARS} characters")
        language = (language or "auto").strip().lower()
        if language == "auto":
            language = detect_language(prompt_text)
        if language not in LANGUAGES:
            raise BadRequest(f"language must be one of {', '.join(('auto',) + LANGUAGES)}")
        if voice_id is not None and not _CLONE_ID_RE.match(voice_id):
            raise BadRequest("id must be 1-64 characters of letters, digits, '_' or '-'")

        try:
            samples, sample_rate = sf.read(io.BytesIO(audio), dtype="float32", always_2d=True)
        except (RuntimeError, TypeError, ValueError) as error:
            raise BadRequest(f"unreadable audio (wav, flac or mp3 expected): {error}") from error
        duration = samples.shape[0] / sample_rate if sample_rate else 0.0
        if not CLONE_MIN_SECONDS <= duration <= CLONE_MAX_SECONDS:
            raise BadRequest(
                f"audio must be {CLONE_MIN_SECONDS:g}-{CLONE_MAX_SECONDS:g} s long, got {duration:.2f} s")
        mono = samples.mean(axis=1) if samples.shape[1] > 1 else samples[:, 0]
        mono = np.clip(mono, -1.0, 1.0)

        with self._lock:
            if voice_id is None:
                voice_id = self._new_clone_id()
            elif self.is_builtin(voice_id) or any(p.exists() for p in self._paths(voice_id)):
                raise Conflict(f"voice {voice_id!r} already exists")
            self.voices_dir.mkdir(parents=True, exist_ok=True)
            wav_path, json_path = self._paths(voice_id)
            meta = {
                "prompt_text": prompt_text,
                "language": language,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "prompt_seconds": round(duration, 2),
            }
            # Original sample rate is kept: Ming resamples prompts itself (to
            # 16 kHz for the speaker embedding, 44.1 kHz for the latent).
            _atomic_write(wav_path, lambda handle: sf.write(
                handle, mono, sample_rate, subtype="PCM_16", format="WAV"))
            _atomic_write(json_path, lambda handle: handle.write(
                json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8")))
        return self.get(voice_id)

    def _new_clone_id(self) -> str:
        while True:
            voice_id = f"clone_{secrets.token_hex(4)}"
            if not any(p.exists() for p in self._paths(voice_id)):
                return voice_id

    def delete(self, voice_id: str) -> None:
        with self._lock:
            if self._load_clone(voice_id) is None:
                if self.is_builtin(voice_id):
                    raise MethodNotAllowed(f"voice {voice_id!r} is built in and cannot be deleted")
                raise NotFound(f"voice {voice_id!r} not found")
            # Sidecar first: the voice disappears from listings before its audio does.
            for path in reversed(self._paths(voice_id)):
                path.unlink(missing_ok=True)


def _atomic_write(path: Path, write) -> None:
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    try:
        with tmp.open("wb") as handle:
            write(handle)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
