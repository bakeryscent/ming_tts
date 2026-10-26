import pytest

from app.timestamps import (COMMA_PAUSE, SENTENCE_PAUSE, SILENCE_TAIL_SECONDS, Fragment,
                            estimate_words, tokenize)


def texts(words):
    return [word["text"] for word in words]


def assert_well_formed(words):
    assert [word["id"] for word in words] == list(range(len(words)))
    previous_end = 0.0
    for word in words:
        assert word["start"] >= previous_end
        assert word["end"] >= word["start"]
        assert round(word["start"], 3) == word["start"]
        assert round(word["end"], 3) == word["end"]
        previous_end = word["end"]


def test_tokenize_latin_keeps_attached_punctuation():
    tokens = tokenize('"Hi," she said. Don\'t stop - 3.5 km!')
    assert [t.text for t in tokens] == ['"Hi,"', "she", "said.", "Don't", "stop -", "3.5", "km!"]
    assert [t.weight for t in tokens] == [2, 3, 4, 4, 4, 2, 2]


def test_tokenize_cjk_one_word_per_character():
    tokens = tokenize("你好，世界。我用GPU加速")
    assert [t.text for t in tokens] == ["你", "好，", "世", "界。", "我", "用", "GPU", "加", "速"]
    assert [t.weight for t in tokens] == [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 3, 1.5, 1.5]


def test_words_proportional_to_letters():
    duration = 2.0 + SILENCE_TAIL_SECONDS
    words = estimate_words([Fragment("The garden opens", 0.0, duration)])
    assert texts(words) == ["The", "garden", "opens"]
    assert_well_formed(words)
    assert words[0]["start"] == 0.0
    # No punctuation: words are contiguous and fill the span before the tail.
    assert words[-1]["end"] == pytest.approx(2.0, abs=0.002)
    lengths = [w["end"] - w["start"] for w in words]
    assert lengths[1] == pytest.approx(2 * lengths[0], abs=0.003)


def test_punctuation_leaves_a_pause_but_not_after_the_last_word():
    words = estimate_words([Fragment("Hello, big world. Bye.", 0.0, 3.0)])
    assert texts(words) == ["Hello,", "big", "world.", "Bye."]
    scale = (3.0 - SILENCE_TAIL_SECONDS) / (5 + 3 + 5 + 3 + COMMA_PAUSE + SENTENCE_PAUSE)
    assert words[1]["start"] - words[0]["end"] == pytest.approx(COMMA_PAUSE * scale, abs=0.002)
    assert words[3]["start"] - words[2]["end"] == pytest.approx(SENTENCE_PAUSE * scale, abs=0.002)
    assert words[-1]["end"] == pytest.approx(3.0 - SILENCE_TAIL_SECONDS, abs=0.002)


def test_cjk_characters_share_time_evenly():
    words = estimate_words([Fragment("你好世界", 0.0, 1.5)])
    assert texts(words) == ["你", "好", "世", "界"]
    lengths = {round(w["end"] - w["start"], 3) for w in words}
    assert lengths == {0.3}


def test_fragments_are_anchored_at_their_offsets():
    fragments = [Fragment("First part here.", 0.0, 1.7), Fragment("Second part.", 1.7, 1.2)]
    words = estimate_words(fragments)
    assert texts(words) == ["First", "part", "here.", "Second", "part."]
    assert_well_formed(words)
    assert words[3]["start"] == 1.7
    assert words[2]["end"] <= 1.7 - SILENCE_TAIL_SECONDS + 0.001


def test_short_fragment_keeps_half_its_duration_for_speech():
    words = estimate_words([Fragment("Hi", 0.0, 0.4)])
    assert words == [{"id": 0, "start": 0.0, "end": 0.2, "text": "Hi"}]


def test_empty_and_zero_length_fragments_are_skipped():
    fragments = [Fragment("...", 0.0, 0.5), Fragment("Okay", 0.5, 0.0), Fragment("Yes.", 0.5, 1.0)]
    words = estimate_words(fragments)
    assert texts(words) == ["Yes."]
    assert words[0]["id"] == 0 and words[0]["start"] == 0.5


def test_many_fragments_stay_monotonic():
    fragments, offset = [], 0.0
    for index in range(12):
        duration = 0.31 + 0.137 * index
        fragments.append(Fragment(f"Word{index}, another one; and 你好。", offset, duration))
        offset += duration
    words = estimate_words(fragments)
    assert_well_formed(words)
    assert words[-1]["end"] <= round(offset, 3)


def test_word_texts_come_from_the_source_text_when_counts_match():
    fragments = [Fragment("I bought 42 apples，", 0.0, 1.5), Fragment("then left。", 1.5, 1.0)]
    words = estimate_words(fragments, source_text="I bought 42 apples, then left")
    assert texts(words) == ["I", "bought", "42", "apples,", "then", "left"]
    assert words[4]["start"] == 1.5


def test_fragment_texts_are_used_when_counts_differ():
    fragments = [Fragment("I bought forty two apples。", 0.0, 2.0)]
    words = estimate_words(fragments, source_text="I bought 42 apples.")
    assert texts(words) == ["I", "bought", "forty", "two", "apples。"]
