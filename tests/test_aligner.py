"""Alignment logic without torch: romanization cleanup, PCM slicing, the fake
backend and merging aligner spans with the estimate (app.aligner)."""

from __future__ import annotations

import pytest

from app import aligner
from app.aligner import (FakeAligner, align_fragments, clean_romanized, merge_alignment,
                         slice_pcm, supports)
from app.audio import float_to_pcm16
from app.timestamps import SILENCE_TAIL_SECONDS, Fragment, estimate_words
from tests.test_timestamps import assert_well_formed, texts

ALPHABET = frozenset("abcdefghijklmnopqrstuvwxyz'")


def pcm_of(seconds: float, sample_rate: int = 24000) -> bytes:
    return float_to_pcm16([0.0] * int(seconds * sample_rate))


def test_clean_romanized_keeps_only_the_aligner_alphabet():
    assert clean_romanized("Héllo, Wörld!", ALPHABET) == "hllowrld"  # uroman strips accents first
    assert clean_romanized("Don't", ALPHABET) == "don't"
    assert clean_romanized("42", ALPHABET) == ""
    assert clean_romanized("ni3 hao3", ALPHABET) == "nihao"


def test_every_ming_language_is_supported():
    for language in ("en", "zh", "de", "es", "fr", "ko", "ja"):
        assert supports(language)
    assert not supports("xx")


def test_slice_pcm_cuts_whole_samples():
    pcm = bytes(range(256)) * 10  # 1280 samples
    sliced = slice_pcm(pcm, 1000, 0.5, 0.25)
    assert sliced == pcm[1000:1500]
    assert slice_pcm(pcm, 1000, 1.2, 1.0) == pcm[2400:]
    assert slice_pcm(pcm, 1000, 5.0, 1.0) == b""


def test_fake_aligner_spreads_words_and_skips_numbers():
    spans = FakeAligner().align(pcm_of(2.0), 24000, ["I", "bought", "42", "apples."], "en")
    assert spans[2] is None
    assert spans[0] == pytest.approx((0.0, 2 / 3))
    assert spans[3] == pytest.approx((4 / 3, 2.0))


def test_fake_aligner_failure_is_per_fragment(monkeypatch):
    monkeypatch.setenv("FAKE_ALIGN_FAIL", "1")
    results = align_fragments(FakeAligner(), pcm_of(3.0), 24000, "en",
                              [(0.0, 1.5, ["one"]), (1.5, 1.5, ["two"])])
    assert results == [None, None]


def test_align_fragments_slices_each_fragment():
    results = align_fragments(FakeAligner(), pcm_of(3.0), 24000, "en",
                              [(0.0, 1.0, ["one", "two"]), (1.0, 2.0, ["three"])])
    assert results == [[(0.0, 0.5), (0.5, 1.0)], [(0.0, 2.0)]]


def test_merge_uses_aligned_spans_anchored_at_fragment_offsets():
    fragments = [Fragment("First part here.", 0.0, 1.7), Fragment("Second part.", 1.7, 1.2)]
    aligned = [[(0.05, 0.4), (0.45, 0.8), (0.9, 1.3)], [(0.1, 0.5), (0.6, 0.85)]]
    words, fully = merge_alignment(fragments, None, aligned)
    assert fully
    assert texts(words) == ["First", "part", "here.", "Second", "part."]
    assert_well_formed(words)
    assert [(w["start"], w["end"]) for w in words] == [
        (0.05, 0.4), (0.45, 0.8), (0.9, 1.3), (1.8, 2.2), (2.3, 2.55)]


def test_merge_labels_words_from_the_source_text():
    fragments = [Fragment("I bought 42 apples，", 0.0, 1.5), Fragment("then left。", 1.5, 1.0)]
    aligned = [[(0.0, 0.2), (0.25, 0.6), None, (0.9, 1.3)], [(0.0, 0.4), (0.45, 0.8)]]
    words, fully = merge_alignment(fragments, "I bought 42 apples, then left", aligned)
    assert fully
    assert texts(words) == ["I", "bought", "42", "apples,", "then", "left"]
    # The number takes the gap between its aligned neighbours.
    assert (words[2]["start"], words[2]["end"]) == (0.6, 0.9)


def test_unaligned_run_shares_the_gap_evenly():
    fragments = [Fragment("a 1 2 b", 0.0, 2.0)]
    words, fully = merge_alignment(fragments, None, [[(0.0, 0.2), None, None, (1.0, 1.4)]])
    assert fully
    assert [(w["start"], w["end"]) for w in words] == [
        (0.0, 0.2), (0.2, 0.6), (0.6, 1.0), (1.0, 1.4)]


def test_leading_and_trailing_unaligned_words():
    fragments = [Fragment("1 word 2", 0.0, 2.0 + SILENCE_TAIL_SECONDS)]
    words, _ = merge_alignment(fragments, None, [[None, (0.5, 1.0), None]])
    # Leading: from the fragment start; trailing: up to the end of speech (tail excluded).
    assert [(w["start"], w["end"]) for w in words] == [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0)]


def test_fragment_that_failed_to_align_is_estimated():
    fragments = [Fragment("First part here.", 0.0, 1.7), Fragment("Second part.", 1.7, 1.2)]
    aligned = [[(0.05, 0.4), (0.45, 0.8), (0.9, 1.3)], None]
    words, fully = merge_alignment(fragments, None, aligned)
    assert not fully
    assert_well_formed(words)
    assert (words[0]["start"], words[0]["end"]) == (0.05, 0.4)
    assert words[3:] == estimate_words(fragments)[3:]


def test_fragment_with_nothing_alignable_is_estimated():
    fragments = [Fragment("42 - 7", 0.0, 1.0)]
    words, fully = merge_alignment(fragments, None, [[None, None]])
    assert not fully
    assert words == estimate_words(fragments)


@pytest.mark.parametrize("aligned", [None, [], [[(0.0, 0.5)]]])
def test_missing_or_short_alignment_falls_back_to_the_estimate(aligned):
    fragments = [Fragment("Hello big world.", 0.0, 2.0)]
    words, fully = merge_alignment(fragments, None, aligned)
    assert not fully
    assert words == estimate_words(fragments)


def test_aligned_spans_are_clamped_to_the_fragment_and_monotonic():
    fragments = [Fragment("one two", 0.0, 1.0), Fragment("three", 1.0, 1.0)]
    aligned = [[(0.0, 0.7), (0.6, 1.4)], [(-0.1, 0.5)]]
    words, _ = merge_alignment(fragments, None, aligned)
    assert_well_formed(words)
    assert [(w["start"], w["end"]) for w in words] == [(0.0, 0.7), (0.7, 1.0), (1.0, 1.5)]


def test_empty_and_zero_length_fragments_are_skipped_in_aligned_results():
    fragments = [Fragment("...", 0.0, 0.5), Fragment("Okay", 0.5, 0.0), Fragment("Yes.", 0.5, 1.0)]
    words, fully = merge_alignment(fragments, None, [[], [], [(0.1, 0.6)]])
    assert fully
    assert words == [{"id": 0, "start": 0.6, "end": 1.1, "text": "Yes."}]


def test_load_backend_rejects_unknown_names():
    with pytest.raises(RuntimeError):
        aligner.load_backend("nope", "cpu")
    backend, info = aligner.load_backend("fake", "cpu")
    assert isinstance(backend, FakeAligner) and info["backend"] == "fake"
