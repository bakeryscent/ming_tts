"""Request → MingTTS call mapping.

``plan_request`` turns an API request into a ``SynthesisPlan`` (task routing
and validation, docs/API.md "Task routing"). It runs in the API process and
needs no torch: it only imports Ming's ``tts_tasks`` (pure Python) to reuse
its validation. ``MingBackend`` runs a plan inside a worker process; torch
and the Ming model are imported there, inside its constructor.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
import warnings
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.audio import float_to_pcm16
from app.config import NATIVE_SAMPLE_RATE, Settings
from app.errors import BadRequest, PayloadTooLarge
from app.leveling import level
from app.voices import MULTILINGUAL_LANGUAGES, VoiceRegistry, detect_language

log = logging.getLogger("ming-tts-api.engine")

# infer.py's default; a fixed seed makes repeated requests reproducible.
DEFAULT_SEED = 1895

# BASIC is documented and demonstrated with all three attributes set; a
# missing one is filled with the neutral middle value instead of being sent
# to the model as null.
BASIC_DEFAULTS = {"speed": "中速", "pitch": "中", "volume": "中"}

WARMUP_TEXT = "The voice model is warming up."

# Ming's default is 10; below that the voice turns robotic.
FLOW_STEPS = 16


def _keep_lowercase_a_out_of_units() -> None:
    """Stop Ming reading "9:30 a.m." and "5 a day" as amperes.

    Its unit pattern ignores case, so a number followed by the word "a" matches
    the unit "A". Only the capital letter means amperes.
    """
    from frontend import number_en

    pattern = number_en._unit_re.pattern.replace("|A|", "|(?-i:A)|")
    number_en._unit_re = re.compile(pattern, re.IGNORECASE)


def add_ming_to_path(ming_dir: Path) -> None:
    """Make the vendored Ming repo importable (it is not a package)."""
    path = str(ming_dir)
    if path not in sys.path:
        sys.path.insert(0, path)


def load_tts_tasks(ming_dir: Path, *, required: bool) -> ModuleType | None:
    """Import Ming's ``tts_tasks``; ``None`` when absent and not ``required``."""
    if not (Path(ming_dir) / "tts_tasks.py").is_file():
        if required:
            raise RuntimeError(f"Ming repo not found at MING_DIR={ming_dir}")
        log.warning("ming_repo_missing", extra={"ming_dir": str(ming_dir)})
        return None
    add_ming_to_path(ming_dir)
    import tts_tasks

    return tts_tasks


class TTSRequest(BaseModel):
    """Body of ``POST /v1/tts``. Unknown fields are ignored."""

    model_config = ConfigDict(extra="ignore")

    text: str
    voice: str | None = None
    style: str | None = None
    instruction: str | None = None
    language: Literal["auto", "en", "zh", "de", "es", "fr", "ko", "ja"] = "auto"
    controls: dict[str, str] | None = None
    sample_rate: Literal[24000, 44100] = 24000
    output_format: Literal["pcm", "wav"] = "pcm"
    return_timestamps: bool = True
    seed: int | None = None
    cfg: float = Field(2.0, ge=0.0, le=10.0)
    sigma: float = Field(0.25, gt=0.0, le=2.0)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_length: int = Field(24, ge=4, le=200)
    flow_steps: Literal[5, 6, 7, 10, 12, 16] = 16
    leveling: bool = True

    @field_validator("voice", "style", "instruction", mode="before")
    @classmethod
    def _blank_is_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value.strip() if isinstance(value, str) else value

    @field_validator("language", mode="before")
    @classmethod
    def _lower_language(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value


@dataclass(frozen=True)
class SynthesisPlan:
    """Everything a worker needs for one ``MingTTS.synthesize`` call.

    Sent to the worker as a plain dict (``to_payload``) so the pipe carries
    only builtins.
    """

    task: str
    text: str
    language: str  # resolved language code: en, zh, de, es, fr, ko or ja
    language_id: str | None = None
    voice: str | None = None
    instruction: str | None = None
    controls: dict[str, str] | None = None
    prompt_wav_path: str | None = None
    prompt_text: str | None = None
    spk_emb_path: str | None = None
    use_spk_emb: bool = False
    use_zero_spk_emb: bool = False
    cfg: float = 2.0
    sigma: float = 0.25
    temperature: float = 0.0
    max_length: int = 24
    seed: int = DEFAULT_SEED
    sample_rate: int = 24000
    flow_steps: int = 16
    leveling: bool = True

    def synthesize_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``MingTTS.synthesize``."""
        return {
            "task": self.task, "text": self.text, "language_id": self.language_id,
            "instruction": self.instruction, "controls": self.controls,
            "prompt_wav_path": self.prompt_wav_path, "prompt_text": self.prompt_text,
            "spk_emb_path": self.spk_emb_path,
            "use_spk_emb": self.use_spk_emb, "use_zero_spk_emb": self.use_zero_spk_emb,
            "cfg": self.cfg, "sigma": self.sigma, "temperature": self.temperature,
            "max_length": self.max_length,
        }

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SynthesisPlan":
        names = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in payload.items() if key in names})


def plan_request(
    req: TTSRequest,
    registry: VoiceRegistry,
    tts_tasks: ModuleType | None,
    max_text_chars: int,
) -> SynthesisPlan:
    """Route a request to a Ming task (docs/API.md) and validate it.

    Raises ``BadRequest`` (400), ``PayloadTooLarge`` (413) or the registry's
    ``NotFound`` (404). ``tts_tasks`` is ``None`` only under FAKE_ENGINE
    without a Ming checkout; the Ming-specific value checks are skipped then.
    """
    text = req.text.strip()
    if not text:
        raise BadRequest("text must not be empty")
    if len(text) > max_text_chars:
        raise PayloadTooLarge(f"text is {len(text)} characters; the limit is {max_text_chars}")

    language = detect_language(text) if req.language == "auto" else req.language
    multilingual = language in MULTILINGUAL_LANGUAGES
    # Ming only needs language_id for the five non-zh/en languages, where it
    # also switches off zh/en text normalization; the README passes it for
    # zh/en only on plain cloning (task TTS), so that is the one place it is
    # set for zh/en below.
    extra_language_id = language if multilingual else None
    controls = dict(req.controls or {})
    common: dict[str, Any] = {
        "text": text, "language": language, "cfg": req.cfg, "sigma": req.sigma,
        "temperature": req.temperature, "max_length": req.max_length,
        "seed": DEFAULT_SEED if req.seed is None else req.seed,
        "sample_rate": req.sample_rate, "flow_steps": req.flow_steps, "leveling": req.leveling,
    }

    if req.style is not None and req.voice is not None:
        raise BadRequest("style and voice are mutually exclusive")
    if req.instruction is not None and req.style is not None:
        raise BadRequest("instruction cannot be combined with style")
    if controls and req.voice is None:
        raise BadRequest("controls need a clone voice (they condition on its reference audio)")

    if req.instruction is not None:
        if req.voice is None:
            plan = SynthesisPlan(task="FREEFORM", instruction=req.instruction,
                                 use_zero_spk_emb=True, language_id=extra_language_id, **common)
        else:
            # Without a speaker the model picks any voice. Half of the voice's
            # fingerprint keeps it recognizable; the full one flattens the delivery.
            fingerprint = registry.fingerprint(registry.get(req.voice).id, direction=True)
            if fingerprint is None:
                raise BadRequest("instruction with a voice needs a voice with a fingerprint")
            plan = SynthesisPlan(task="FREEFORM", voice=req.voice, instruction=req.instruction,
                                 spk_emb_path=fingerprint, use_spk_emb=True,
                                 language_id=extra_language_id, **common)
    elif req.style is not None:
        plan = SynthesisPlan(task="VOICE_DESIGN", controls={"style": req.style},
                             use_zero_spk_emb=True, language_id=extra_language_id, **common)
    elif req.voice is None:
        raise BadRequest("one of voice, style or instruction is required")
    else:
        voice = registry.get(req.voice)
        if voice.kind == "builtin":
            if controls:
                raise BadRequest("controls need a clone voice, not a built-in one")
            # The fingerprint keeps the voice steady between fragments and requests.
            fingerprint = registry.fingerprint(voice.id)
            plan = SynthesisPlan(task="IP", voice=voice.id, controls={"ip": voice.id},
                                 spk_emb_path=fingerprint, use_spk_emb=fingerprint is not None,
                                 use_zero_spk_emb=fingerprint is None,
                                 language_id=extra_language_id, **common)
        elif controls:
            task, controls = _control_task(controls, tts_tasks)
            # The control tasks condition on the speaker embedding only, as in
            # Ming's examples: without prompt_text the acoustic prompt is unused.
            plan = SynthesisPlan(
                task=task, voice=voice.id, controls=controls,
                prompt_wav_path=str(voice.wav_path), use_spk_emb=True,
                # DIALECT takes no language id (tts_tasks drops it with a warning).
                language_id=None if task == "DIALECT" else extra_language_id, **common)
        else:
            plan = SynthesisPlan(
                task="MULTILINGUAL" if multilingual else "TTS", voice=voice.id,
                prompt_wav_path=str(voice.wav_path), prompt_text=voice.prompt_text,
                use_spk_emb=True, language_id=language, **common)

    if tts_tasks is not None:
        _validate_with_ming(plan, tts_tasks)
    return plan


def _control_task(
    controls: dict[str, str], tts_tasks: ModuleType | None,
) -> tuple[str, dict[str, str]]:
    keys = set(controls)
    if keys == {"emotion"}:
        task, allowed = "EMOTION", {"emotion": getattr(tts_tasks, "EMOTION_VALUES", None)}
    elif keys == {"dialect"}:
        task, allowed = "DIALECT", {"dialect": getattr(tts_tasks, "DIALECT_VALUES", None)}
    elif keys and keys <= set(BASIC_DEFAULTS):
        task = "BASIC"
        controls = {**BASIC_DEFAULTS, **controls}
        values = getattr(tts_tasks, "BASIC_ATTRIBUTE_VALUES", {})
        allowed = {key: values.get(key) for key in BASIC_DEFAULTS}
    else:
        raise BadRequest(
            "controls must be {emotion}, {dialect} or any of {speed, pitch, volume}")
    for key, choices in allowed.items():
        if choices is not None and controls[key] not in choices:
            raise BadRequest(f"controls.{key} must be one of {', '.join(choices)}")
    return task, controls


def _validate_with_ming(plan: SynthesisPlan, tts_tasks: ModuleType) -> None:
    """Run Ming's own request builder so its validation applies here too."""
    kwargs = plan.synthesize_kwargs()
    kwargs.pop("max_length")
    task = kwargs.pop("task")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            tts_tasks.build_task_generation_kwargs(task, **kwargs)
        except ValueError as error:
            raise BadRequest(str(error)) from error


def display_fragments(text: str, language_id: str | None, max_length: int) -> list[str]:
    """Ming's fragment split of ``text`` before text normalization.

    Mirrors ``MingTTS.prepare_text`` minus the normalization step, so the
    fragments hold the caller's words ("42", not "forty two") and word
    timestamps can be labelled with them. Needs the Ming repo on ``sys.path``.
    """
    import tts_tasks
    from frontend.text_segment_cut import cut_text_by_semantic_length
    from frontend.toolkit import add_period, preprosess_text

    if tts_tasks.preserve_multilingual_text(language_id):
        prepared = preprosess_text(text).strip()
    else:
        prepared = add_period(preprosess_text(text))
    fragments = cut_text_by_semantic_length(prepared, max_length)["fragments"]
    return [fragment for fragment in fragments if fragment.strip()]


class GpuOutOfMemory(RuntimeError):
    """CUDA OOM during synthesis; the worker freed its cache and keeps serving."""


class MingBackend:
    """One MingTTS replica on the GPU. Only constructed inside a worker."""

    def __init__(self, settings: Settings) -> None:
        add_ming_to_path(settings.ming_dir)
        import torch
        import torchaudio
        from infer import seed_everything
        from ming_tts import MingTTS, trim_trailing_silence

        self._torch = torch
        self._resample = torchaudio.functional.resample
        self._seed_everything = seed_everything
        self._trim = trim_trailing_silence
        self._device = settings.device

        started = time.perf_counter()
        self.model = MingTTS(
            settings.model_id, device=settings.device,
            text_normalization=settings.text_normalization, cache_prompts=True)
        if self.model.sample_rate != NATIVE_SAMPLE_RATE:
            raise RuntimeError(f"unexpected model sample rate {self.model.sample_rate}")
        self.load_s = round(time.perf_counter() - started, 2)
        _keep_lowercase_a_out_of_units()
        if settings.compile_dit:
            head = self.model.model.flowloss.cfm.model
            head.forward = torch.compile(head.forward)
        self._flow_steps = FLOW_STEPS
        flow = self.model.model.flowloss
        flow_sample = flow.sample
        flow.sample = lambda *args, **kwargs: flow_sample(*args, **{**kwargs, "steps": self._flow_steps})
        # MingTTS caches encoded prompts by path. A clone deleted and re-created
        # under the same id has the same path, so the file's mtime is tracked
        # to drop stale entries.
        self._prompt_mtimes: dict[str, int] = {}

    def warmup(self) -> int:
        """One short synthesis (also loads the text normalizer); returns ms."""
        started = time.perf_counter()
        self.synthesize(SynthesisPlan(
            task="IP", text=WARMUP_TEXT, language="en", voice="US_James",
            controls={"ip": "US_James"}, use_zero_spk_emb=True))
        return int((time.perf_counter() - started) * 1000)

    def info(self) -> dict[str, Any]:
        torch = self._torch
        free, total = torch.cuda.mem_get_info(self._device)
        return {
            "gpu_name": torch.cuda.get_device_name(self._device),
            "vram_total_mb": total // 2**20,
            "vram_used_mb": (total - free) // 2**20,
            "prompt_cache_size": len(self.model._prompt_cache),
            "load_s": self.load_s,
        }

    def _refresh_prompt(self, path: str) -> None:
        try:
            mtime = os.stat(path).st_mtime_ns
        except FileNotFoundError as error:
            raise ValueError("the voice's reference audio no longer exists") from error
        if self._prompt_mtimes.get(path) != mtime:
            self.model._prompt_cache.pop(path, None)
            self._prompt_mtimes[path] = mtime

    def synthesize(self, plan: SynthesisPlan) -> dict[str, Any]:
        """Run one plan; returns PCM at ``plan.sample_rate`` and fragment timings.

        Raises ``ValueError`` for input Ming rejects and ``GpuOutOfMemory``.
        """
        torch = self._torch
        if plan.prompt_wav_path:
            self._refresh_prompt(plan.prompt_wav_path)
        self._flow_steps = plan.flow_steps
        self._seed_everything(plan.seed)

        # MingTTS.synthesize returns the joined waveform only. Wrapping the
        # model's per-fragment generate() records each fragment's text and its
        # length after the same trailing-silence trim synthesize() applies,
        # which is what the word-timestamp estimate is anchored on.
        recorded: list[tuple[str, int]] = []
        inner = self.model.model
        generate = inner.generate

        def recording_generate(*args: Any, **kwargs: Any) -> Any:
            chunk = generate(*args, **kwargs)
            trimmed = self._trim(chunk, self.model.sample_rate)
            recorded.append((kwargs["text"], int(trimmed.shape[-1])))
            return chunk

        inner.generate = recording_generate
        started = time.perf_counter()
        try:
            waveform = self.model.synthesize(**plan.synthesize_kwargs())
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            raise GpuOutOfMemory("GPU out of memory") from error
        finally:
            del inner.generate  # drop the instance attribute; the class method is back

        if plan.leveling:
            waveform = torch.from_numpy(level(waveform.squeeze(0).numpy(), NATIVE_SAMPLE_RATE)).unsqueeze(0)
        if plan.sample_rate != NATIVE_SAMPLE_RATE:
            waveform = self._resample(waveform, NATIVE_SAMPLE_RATE, plan.sample_rate)
        pcm = float_to_pcm16(waveform.numpy())
        runtime_ms = int((time.perf_counter() - started) * 1000)

        texts = display_fragments(plan.text, plan.language_id, plan.max_length)
        if len(texts) != len(recorded):
            # Normalization emptied a fragment; fall back to the normalized texts.
            texts = [text for text, _ in recorded]
        free, total = torch.cuda.mem_get_info(self._device)
        return {
            "pcm": pcm,
            "sample_rate": plan.sample_rate,
            "fragments": [(text, samples / NATIVE_SAMPLE_RATE)
                          for text, (_, samples) in zip(texts, recorded)],
            "runtime_ms": runtime_ms,
            "prompt_cache_size": len(self.model._prompt_cache),
            "vram_used_mb": (total - free) // 2**20,
        }
