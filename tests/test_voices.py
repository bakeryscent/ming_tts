import json
import re

import pytest
import soundfile as sf

from app.errors import BadRequest, Conflict, MethodNotAllowed, NotFound
from app.voices import EN_BUILTIN_VOICES, VoiceRegistry, detect_language
from tests.helpers import ROOT, wav_bytes


def test_seed_copies_presets_once(tmp_path):
    target = tmp_path / "voices"
    registry = VoiceRegistry(target, ROOT / "voices")
    assert registry.seed() == 2
    assert sorted(p.name for p in target.iterdir()) == [
        "DB30.json", "DB30.wav", "lingguang.json", "lingguang.wav"]
    assert registry.seed() == 0


def test_seed_leaves_a_populated_dir_alone(tmp_path):
    target = tmp_path / "voices"
    target.mkdir()
    (target / "mine.wav").write_bytes(wav_bytes(2.0))
    assert VoiceRegistry(target, ROOT / "voices").seed() == 0
    assert not (target / "lingguang.wav").exists()


def test_presets_and_builtins_are_listed(registry):
    voices = {voice.id: voice for voice in registry.list()}
    for voice_id in EN_BUILTIN_VOICES:
        assert voices[voice_id].kind == "builtin"
        assert voices[voice_id].language == "en"
    assert voices["US_Stella"].public() == {
        "id": "US_Stella", "kind": "builtin", "language": "en", "gender": "female",
        "accent": "American", "description": "narration, conversational, warm, youthful"}
    assert voices["灵小柔"].kind == "builtin" and voices["灵小柔"].language == "zh"
    assert voices["灵昭野_生气"].emotion == "angry"
    lingguang = voices["lingguang"]
    assert lingguang.kind == "clone" and lingguang.language == "zh"
    assert lingguang.prompt_text == "诶，你今天有空吗？我们能不能聊聊这件事啊？"
    assert lingguang.prompt_seconds == pytest.approx(3.29, abs=0.01)
    assert "prompt_text" not in lingguang.public()
    assert voices["DB30"].prompt_text == "感谢你的认可。"


def test_get_resolves_each_kind(registry):
    assert registry.get("US_James").kind == "builtin"
    assert registry.get("DB30").wav_path.name == "DB30.wav"
    assert registry.get("灵楷").language == "zh"
    with pytest.raises(NotFound):
        registry.get("灵昭野")  # emotion-variant voices are only valid with a variant suffix
    with pytest.raises(NotFound):
        registry.get("nobody")
    with pytest.raises(NotFound):
        registry.get("../etc/passwd")


def test_unknown_chinese_name_needs_the_catalog(tmp_path):
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    registry = VoiceRegistry(tmp_path / "voices", bundled)
    with pytest.raises(NotFound):
        registry.get("灵小柔")


def test_create_clone_stores_mono_16bit_at_original_rate(registry, voices_dir):
    voice = registry.create_clone(
        wav_bytes(3.0, sample_rate=48000, channels=2), "Hello there, this is me.", "en")
    assert re.fullmatch(r"clone_[0-9a-f]{8}", voice.id)
    info = sf.info(str(voices_dir / f"{voice.id}.wav"))
    assert (info.channels, info.samplerate, info.subtype) == (1, 48000, "PCM_16")
    assert info.duration == pytest.approx(3.0)
    meta = json.loads((voices_dir / f"{voice.id}.json").read_text(encoding="utf-8"))
    assert meta["prompt_text"] == "Hello there, this is me."
    assert meta["language"] == "en"
    assert meta["prompt_seconds"] == 3.0
    assert "created_at" in meta
    assert registry.get(voice.id).prompt_text == "Hello there, this is me."
    assert not list(voices_dir.glob(".*.tmp"))


def test_create_clone_with_id_and_detected_language(registry):
    voice = registry.create_clone(wav_bytes(1.5, fmt="FLAC"), "你好，很高兴认识你。", "auto", "my-voice_1")
    assert voice.id == "my-voice_1"
    assert voice.language == "zh"
    with pytest.raises(Conflict):
        registry.create_clone(wav_bytes(1.5), "again", "en", "my-voice_1")
    with pytest.raises(Conflict):
        registry.create_clone(wav_bytes(1.5), "builtin name", "en", "US_James")


@pytest.mark.parametrize("audio, text, language, voice_id", [
    (wav_bytes(0.5), "too short", "en", None),
    (wav_bytes(20.5), "too long", "en", None),
    (b"not audio at all", "garbage", "en", None),
    (wav_bytes(2.0), "   ", "en", None),
    (wav_bytes(2.0), "bad language", "xx", None),
    (wav_bytes(2.0), "bad id", "en", "../escape"),
])
def test_create_clone_rejects_bad_input(registry, audio, text, language, voice_id):
    with pytest.raises(BadRequest):
        registry.create_clone(audio, text, language, voice_id)


def test_delete(registry, voices_dir):
    voice = registry.create_clone(wav_bytes(2.0), "Delete me.", "en")
    registry.delete(voice.id)
    assert not (voices_dir / f"{voice.id}.wav").exists()
    assert not (voices_dir / f"{voice.id}.json").exists()
    with pytest.raises(NotFound):
        registry.delete(voice.id)
    with pytest.raises(MethodNotAllowed):
        registry.delete("US_Emma")
    with pytest.raises(MethodNotAllowed):
        registry.delete("灵小柔")
    registry.delete("lingguang")
    assert "lingguang" not in {v.id for v in registry.list()}


def test_detect_language():
    assert detect_language("The garden is open.") == "en"
    assert detect_language("花园开放到日落。") == "zh"
    assert detect_language("Mixed 中文 text") == "zh"
