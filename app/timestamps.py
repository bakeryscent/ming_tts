"""Estimated word timestamps.

Ming has no alignment head, so word timings are estimated rather than
measured (the response says ``words_source: "estimated"``). Each synthesis
fragment is timed separately: Ming generates fragments independently and
concatenates them, so anchoring every fragment at its real offset stops an
estimation error in one fragment from drifting into the next.

Inside a fragment the speech span is shared out by weight: one unit per
letter/digit, 1.5 per CJK character (each one is its own word), and a pause
after words ending in punctuation (0.6 for commas, 1.2 for sentence ends)
that is left as a gap between words rather than added to the word itself.
Ming keeps a fixed silent tail after every fragment, which is excluded from
the span.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# ming_tts.SILENCE_TAIL_SECONDS: silence kept after each fragment's last sound.
SILENCE_TAIL_SECONDS = 0.3
CJK_WEIGHT = 1.5
COMMA_PAUSE = 0.6
SENTENCE_PAUSE = 1.2

_COMMA_MARKS = frozenset(",，、;；:：-—–")
_SENTENCE_MARKS = frozenset(".。!！?？…")

# CJK ideographs plus kana: scripts written without spaces, timed per character.
_CJK_RANGES = (
    (0x3040, 0x30FF),  # hiragana, katakana
    (0x3400, 0x4DBF),  # CJK extension A
    (0x4E00, 0x9FFF),  # CJK unified ideographs
    (0xF900, 0xFAFF),  # CJK compatibility ideographs
)


@dataclass(frozen=True)
class Fragment:
    text: str
    offset: float  # seconds from the start of the utterance
    duration: float  # seconds of audio for this fragment, trailing tail included


@dataclass
class _Token:
    text: str
    weight: float


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return any(low <= code <= high for low, high in _CJK_RANGES)


def tokenize(text: str) -> list[_Token]:
    """Split ``text`` into timed words.

    Latin-script words are whitespace-delimited and keep attached punctuation
    (``"sunset."``, ``"don't"``); each CJK character is a word of its own and
    absorbs the punctuation that follows it. Punctuation leading a word (an
    opening quote) is prefixed to it; free-standing punctuation (``" - "``)
    is appended to the previous word, where it counts as that word's pause.
    """
    tokens: list[_Token] = []
    prefix = ""
    for chunk in text.split():
        if not any(char.isalnum() or _is_cjk(char) for char in chunk):
            if tokens:
                tokens[-1].text += " " + chunk
            else:
                prefix += chunk + " "
            continue
        in_word = False  # the previous character extended a latin word
        started = False  # this chunk has produced a token
        for char in chunk:
            if _is_cjk(char):
                tokens.append(_Token(prefix + char, CJK_WEIGHT))
                prefix, in_word, started = "", False, True
            elif char.isalnum():
                if in_word:
                    tokens[-1].text += char
                    tokens[-1].weight += 1
                else:
                    tokens.append(_Token(prefix + char, 1.0))
                    prefix, in_word, started = "", True, True
            elif started:
                tokens[-1].text += char
            else:
                prefix += char
    return tokens


def _trailing_pause(text: str) -> float:
    trail = []
    for char in reversed(text):
        if char.isalnum() or _is_cjk(char):
            break
        trail.append(char)
    if any(char in _SENTENCE_MARKS for char in trail):
        return SENTENCE_PAUSE
    if any(char in _COMMA_MARKS for char in trail):
        return COMMA_PAUSE
    return 0.0


Span = tuple[float, float]


def fragment_tokens(
    fragments: Sequence[Fragment], source_text: str | None = None,
) -> list[list[_Token]]:
    """Tokenize every fragment, labelling words from ``source_text`` when possible.

    Ming's fragment splitter rewrites punctuation (``","`` becomes ``"，"``).
    When ``source_text`` (the caller's text) has exactly as many words as the
    fragments together, word texts come from it, so they match what the
    caller sent; otherwise the fragment texts are used as they are.
    """
    per_fragment = [tokenize(fragment.text) for fragment in fragments]
    if source_text is not None:
        source = tokenize(source_text)
        if len(source) == sum(len(tokens) for tokens in per_fragment):
            remaining = iter(source)
            per_fragment = [[next(remaining) for _ in tokens] for tokens in per_fragment]
    return per_fragment


def speech_span(fragment: Fragment, tail_seconds: float = SILENCE_TAIL_SECONDS) -> float:
    """Seconds of a fragment that carry speech (its duration minus the tail).

    Never lets the tail swallow more than half of a very short fragment.
    """
    return max(fragment.duration - tail_seconds, fragment.duration * 0.5)


def spread_fragment(
    fragment: Fragment, tokens: Sequence[_Token], tail_seconds: float = SILENCE_TAIL_SECONDS,
) -> list[Span]:
    """Estimated absolute (start, end) per token: the speech span shared by weight."""
    if not tokens:
        return []
    span = speech_span(fragment, tail_seconds)
    pauses = [_trailing_pause(token.text) for token in tokens]
    # The pause after the fragment's last word is the silent tail itself.
    pauses[-1] = 0.0
    scale = span / (sum(token.weight for token in tokens) + sum(pauses))
    spans: list[Span] = []
    cursor = float(fragment.offset)
    for token, pause in zip(tokens, pauses):
        start = cursor
        end = start + token.weight * scale
        cursor = end + pause * scale
        spans.append((start, end))
    return spans


def finish_words(
    fragments: Sequence[Fragment],
    per_fragment: Sequence[Sequence[_Token]],
    spans: Sequence[Sequence[Span]],
) -> list[dict]:
    """Shape absolute spans into Kokoro-style ``[{id, start, end, text}]``.

    Times are rounded to milliseconds, kept monotonic and non-overlapping,
    and clamped to the end of the word's fragment.
    """
    words: list[dict] = []
    previous_end = 0.0
    for fragment, tokens, fragment_spans in zip(fragments, per_fragment, spans):
        fragment_end = round(fragment.offset + fragment.duration, 3)
        for token, (start, end) in zip(tokens, fragment_spans):
            start_r = max(round(start, 3), previous_end)
            end_r = min(max(round(end, 3), start_r), max(fragment_end, start_r))
            words.append({"id": len(words), "start": start_r, "end": end_r,
                          "text": token.text})
            previous_end = end_r
    return words


def estimate_words(
    fragments: Sequence[Fragment],
    source_text: str | None = None,
    tail_seconds: float = SILENCE_TAIL_SECONDS,
) -> list[dict]:
    """Return Kokoro-shaped ``[{id, start, end, text}]`` for all fragments.

    Word texts follow ``fragment_tokens``; empty and zero-length fragments
    are skipped. Words are monotonic and non-overlapping, times rounded to
    milliseconds.
    """
    per_fragment = fragment_tokens(fragments, source_text)
    kept = [(fragment, tokens) for fragment, tokens in zip(fragments, per_fragment)
            if tokens and fragment.duration > 0]
    spans = [spread_fragment(fragment, tokens, tail_seconds) for fragment, tokens in kept]
    return finish_words([f for f, _ in kept], [t for _, t in kept], spans)
