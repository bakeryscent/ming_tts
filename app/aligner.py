"""Forced word alignment on the generated audio (``ALIGNER=mms_fa``).

Ming has no alignment head, so by default word timings are estimated
(app.timestamps). With an aligner the audio is aligned against the words
that were spoken, and the response says ``words_source: "aligned"``.

Backend: torchaudio's ``MMS_FA`` pipeline, a wav2vec2 CTC model trained for
forced alignment on 1,100+ languages over romanized text. The text is
romanized with ``uroman`` (Chinese to pinyin, kana to romaji, hangul to
Latin, accents stripped), lowercased and reduced to the model's alphabet
(``a-z`` and ``'``). The audio is resampled to the model's 16 kHz; it does
not matter whether the request asked for 24 or 44.1 kHz. Each fragment is
aligned on its own slice of audio, anchored at the fragment's real offset
like the estimate: Ming generates fragments independently, so a hard
fragment cannot throw off the next one, and a fragment that fails to align
falls back to its estimate alone.

Words that leave no letters after romanization (``"42"``, ``"—"``: Ming
speaks them as words the aligner never sees) take the gap between their
aligned neighbours. A fragment with no alignable word at all is estimated.

Nothing here imports torch at module level: ``MmsAligner`` is only built
inside the aligner process (app.aligner_service). ``FakeAligner``
(``ALIGNER=fake``) lets the tests run the whole path without torch.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Protocol, Sequence

import numpy as np

from app.audio import PCM_SAMPLE_WIDTH
from app.timestamps import (SILENCE_TAIL_SECONDS, Fragment, Span, finish_words,
                            fragment_tokens, speech_span, spread_fragment)

log = logging.getLogger("ming-tts-api.aligner")

MMS_SAMPLE_RATE = 16000
# uroman wants ISO 639-3 codes; Ming's request languages are ISO 639-1.
UROMAN_LANGUAGE = {"en": "eng", "zh": "zho", "de": "deu", "es": "spa", "fr": "fra",
                   "ko": "kor", "ja": "jpn"}
# MMS_FA was trained on romanized text for every language Ming speaks.
SUPPORTED_LANGUAGES = frozenset(UROMAN_LANGUAGE)
WARMUP_WORDS = ["the", "voice", "model", "is", "warming", "up"]

# One (start, end) in seconds from the start of the aligned audio per word,
# or None where the word had nothing the aligner could place.
WordSpan = Span | None


class AlignerBackend(Protocol):
    def info(self) -> dict[str, Any]: ...

    def align(self, pcm: bytes, sample_rate: int, words: Sequence[str],
              language: str) -> list[WordSpan]: ...


def supports(language: str) -> bool:
    return language in SUPPORTED_LANGUAGES


def clean_romanized(romanized: str, alphabet: frozenset[str]) -> str:
    """Lowercase ``romanized`` and keep only the aligner's alphabet."""
    return "".join(char for char in romanized.lower() if char in alphabet)


def has_letters(word: str) -> bool:
    return any(char.isalpha() for char in word)


def slice_pcm(pcm: bytes, sample_rate: int, offset: float, duration: float) -> bytes:
    """The s16le samples of one fragment, aligned to whole samples."""
    start = int(round(offset * sample_rate)) * PCM_SAMPLE_WIDTH
    end = int(round((offset + duration) * sample_rate)) * PCM_SAMPLE_WIDTH
    return pcm[max(0, start):max(0, end)]


def pcm_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


class FakeAligner:
    """ALIGNER=fake: no torch; spreads each fragment evenly over its words.

    Mirrors the real backend's contract: words without letters get ``None``.
    ``FAKE_ALIGN_S`` (seconds) simulates alignment time, so timeouts can be
    tested; ``FAKE_ALIGN_FAIL=1`` makes every call raise.
    """

    def __init__(self) -> None:
        self._align_s = float(os.environ.get("FAKE_ALIGN_S", "0"))
        self._fail = os.environ.get("FAKE_ALIGN_FAIL", "") == "1"

    def info(self) -> dict[str, Any]:
        return {"backend": "fake", "device": "cpu", "vram_used_mb": 0, "load_s": 0.0}

    def align(self, pcm: bytes, sample_rate: int, words: Sequence[str],
              language: str) -> list[WordSpan]:
        if self._align_s:
            time.sleep(self._align_s)
        if self._fail:
            raise RuntimeError("fake alignment failure")
        duration = len(pcm) / PCM_SAMPLE_WIDTH / sample_rate
        alignable = [index for index, word in enumerate(words) if has_letters(word)]
        spans: list[WordSpan] = [None] * len(words)
        if alignable:
            step = duration / len(alignable)
            for position, index in enumerate(alignable):
                spans[index] = (position * step, (position + 1) * step)
        return spans


class MmsAligner:
    """torchaudio ``MMS_FA`` on one device. Only constructed inside the aligner process."""

    def __init__(self, device: str) -> None:
        import torch
        import torchaudio
        from uroman import Uroman

        self._torch = torch
        self._resample = torchaudio.functional.resample
        self._device = device
        started = time.perf_counter()
        bundle = torchaudio.pipelines.MMS_FA
        self.model = bundle.get_model().to(device).eval()
        self.tokenizer = bundle.get_tokenizer()
        self.aligner = bundle.get_aligner()
        # The dictionary also holds the CTC blank ("-") and the star token.
        self.alphabet = frozenset(label for label in bundle.get_dict() if label not in "-*")
        self.uroman = Uroman()
        self.load_s = round(time.perf_counter() - started, 2)

    def warmup(self) -> int:
        """Align one second of noise, so CUDA kernels are loaded; returns ms."""
        started = time.perf_counter()
        noise = (np.random.default_rng(0).standard_normal(MMS_SAMPLE_RATE) * 1000).astype("<i2")
        self.align(noise.tobytes(), MMS_SAMPLE_RATE, WARMUP_WORDS, "en")
        return int((time.perf_counter() - started) * 1000)

    def info(self) -> dict[str, Any]:
        info: dict[str, Any] = {"backend": "mms_fa", "device": self._device,
                                "load_s": self.load_s, "vram_used_mb": 0}
        if self._device.startswith("cuda"):
            free, total = self._torch.cuda.mem_get_info(self._device)
            info["vram_used_mb"] = (total - free) // 2**20
        return info

    def romanize(self, word: str, language: str) -> str:
        romanized = self.uroman.romanize_string(word, lcode=UROMAN_LANGUAGE.get(language))
        return clean_romanized(romanized, self.alphabet)

    def align(self, pcm: bytes, sample_rate: int, words: Sequence[str],
              language: str) -> list[WordSpan]:
        torch = self._torch
        transcript = [self.romanize(word, language) for word in words]
        keep = [index for index, text in enumerate(transcript) if text]
        spans: list[WordSpan] = [None] * len(words)
        if not keep or not pcm:
            return spans

        waveform = torch.from_numpy(pcm_to_float(pcm)).unsqueeze(0).to(self._device)
        if sample_rate != MMS_SAMPLE_RATE:
            waveform = self._resample(waveform, sample_rate, MMS_SAMPLE_RATE)
        with torch.inference_mode():
            emission, _ = self.model(waveform)
            token_spans = self.aligner(emission[0], self.tokenizer([transcript[i] for i in keep]))
        # Seconds per emission frame (20 ms for wav2vec2), from the real ratio.
        frame_s = waveform.size(1) / emission.size(1) / MMS_SAMPLE_RATE
        for index, token_span in zip(keep, token_spans):
            spans[index] = (token_span[0].start * frame_s, token_span[-1].end * frame_s)
        return spans


def load_backend(name: str, device: str) -> tuple[Any, dict[str, Any]]:
    """Build the backend for ``ALIGNER=name`` (not ``off``), warmed up."""
    if name == "fake":
        backend: Any = FakeAligner()
        return backend, backend.info()
    if name == "mms_fa":
        mms = MmsAligner(device)
        warmup_ms = mms.warmup()
        return mms, {**mms.info(), "warmup_ms": warmup_ms}
    raise RuntimeError(f"unknown ALIGNER backend {name!r}")


def align_fragments(
    backend: AlignerBackend,
    pcm: bytes,
    sample_rate: int,
    language: str,
    fragments: Sequence[tuple[float, float, Sequence[str]]],
) -> list[list[WordSpan] | None]:
    """Align every ``(offset, duration, words)`` fragment on its slice of ``pcm``.

    Spans are relative to the fragment. A fragment whose alignment raises
    yields ``None`` (it will be estimated); the others still count.
    """
    results: list[list[WordSpan] | None] = []
    for offset, duration, words in fragments:
        try:
            results.append(backend.align(slice_pcm(pcm, sample_rate, offset, duration),
                                         sample_rate, words, language))
        except Exception as error:
            log.warning("fragment_alignment_failed", extra={
                "words": len(words), "duration": round(duration, 3), "error": repr(error)})
            results.append(None)
    return results


def _fill_gaps(fragment: Fragment, spans: Sequence[WordSpan],
               tail_seconds: float) -> list[Span]:
    """Absolute spans for a fragment, unaligned words taking the gap between neighbours."""
    base = float(fragment.offset)
    end_of_speech = base + speech_span(fragment, tail_seconds)
    absolute: list[Span | None] = [
        None if span is None else (base + span[0], base + span[1]) for span in spans]
    filled: list[Span] = []
    index = 0
    while index < len(absolute):
        span = absolute[index]
        if span is not None:
            filled.append(span)
            index += 1
            continue
        run_end = index
        while run_end < len(absolute) and absolute[run_end] is None:
            run_end += 1
        left = filled[-1][1] if filled else base
        right = absolute[run_end][0] if run_end < len(absolute) else max(end_of_speech, left)
        right = max(right, left)
        step = (right - left) / (run_end - index)
        for position in range(run_end - index):
            filled.append((left + position * step, left + (position + 1) * step))
        index = run_end
    return filled


def merge_alignment(
    fragments: Sequence[Fragment],
    source_text: str | None,
    aligned: Sequence[Sequence[WordSpan] | None] | None,
    tail_seconds: float = SILENCE_TAIL_SECONDS,
) -> tuple[list[dict], bool]:
    """Combine aligner output with the estimate; returns ``(words, fully_aligned)``.

    ``aligned`` holds one entry per fragment: the backend's spans (relative
    to the fragment) or ``None`` when that fragment could not be aligned.
    ``aligned=None`` means the aligner was unavailable. ``fully_aligned`` is
    True only when every word of every fragment came from the aligner
    (gap-filled words count as aligned: their neighbours were).
    """
    per_fragment = fragment_tokens(fragments, source_text)
    kept_fragments: list[Fragment] = []
    kept_tokens: list[list] = []
    spans: list[list[Span]] = []
    fully_aligned = True
    for index, (fragment, tokens) in enumerate(zip(fragments, per_fragment)):
        if not tokens or fragment.duration <= 0:
            continue
        fragment_spans = aligned[index] if aligned is not None and index < len(aligned) else None
        usable = (fragment_spans is not None and len(fragment_spans) == len(tokens)
                  and any(span is not None for span in fragment_spans))
        if usable:
            spans.append(_fill_gaps(fragment, fragment_spans, tail_seconds))  # type: ignore[arg-type]
        else:
            fully_aligned = False
            spans.append(spread_fragment(fragment, tokens, tail_seconds))
        kept_fragments.append(fragment)
        kept_tokens.append(tokens)
    return finish_words(kept_fragments, kept_tokens, spans), fully_aligned
