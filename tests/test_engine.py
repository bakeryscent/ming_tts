"""Routing and validation of POST /v1/tts bodies (docs/API.md, "Task routing")."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from app import engine
from app.engine import DEFAULT_SEED, SynthesisPlan, TTSRequest, plan_request
from app.errors import BadRequest, NotFound, PayloadTooLarge


@pytest.fixture
def tts_tasks(ming_dir: Path):
    return engine.load_tts_tasks(ming_dir, required=True)


@pytest.fixture
def plan(registry, tts_tasks):
    def _plan(max_text_chars: int = 1200, **body) -> SynthesisPlan:
        return plan_request(TTSRequest(**body), registry, tts_tasks, max_text_chars)
    return _plan


def test_builtin_voice_routes_to_ip(plan):
    result = plan(text="The garden will remain open until sunset.", voice="US_James")
    assert result.task == "IP"
    assert result.controls == {"ip": "US_James"}
    assert result.use_spk_emb and not result.use_zero_spk_emb
    assert result.spk_emb_path.endswith("fingerprints/US_James.pt")
    assert result.prompt_wav_path is None
    assert result.language == "en" and result.language_id is None
    assert result.seed == DEFAULT_SEED and result.sample_rate == 24000
    assert result.flow_steps == 16 and result.leveling


def test_builtin_voice_without_a_fingerprint_uses_no_speaker(plan):
    result = plan(text="The garden will remain open until sunset.", voice="US_Emma")
    assert result.use_zero_spk_emb and not result.use_spk_emb
    assert result.spk_emb_path is None


def test_flow_steps_and_leveling_are_passed_on(plan):
    result = plan(text="Hello.", voice="US_James", flow_steps=10, leveling=False)
    assert result.flow_steps == 10 and not result.leveling


@pytest.mark.parametrize("steps", [4, 8, 32])
def test_unknown_flow_steps_are_rejected(steps):
    with pytest.raises(ValidationError):
        TTSRequest(text="Hello.", voice="US_James", flow_steps=steps)


def test_chinese_builtin_voice(plan):
    result = plan(text="傍晚的风从山口吹过来。", voice="灵小柔")
    assert (result.task, result.controls, result.language) == ("IP", {"ip": "灵小柔"}, "zh")


def test_builtin_voice_in_a_new_language_gets_language_id(plan):
    result = plan(text="Der Garten bleibt bis Sonnenuntergang geöffnet.", voice="US_Emma",
                  language="de")
    assert result.task == "IP" and result.language_id == "de"


def test_clone_voice_routes_to_tts_with_reference(plan, voices_dir):
    result = plan(text="雨后的山谷很安静。", voice="lingguang", seed=7)
    assert result.task == "TTS"
    assert result.prompt_wav_path == str(voices_dir / "lingguang.wav")
    assert result.prompt_text == "诶，你今天有空吗？我们能不能聊聊这件事啊？"
    assert result.use_spk_emb and not result.use_zero_spk_emb
    assert result.language_id == "zh"  # auto-detected; Ming's README passes it for cloning
    assert result.seed == 7


def test_clone_voice_in_a_new_language_routes_to_multilingual(plan):
    result = plan(text="週末、駅前の喫茶店で抹茶パフェを食べました。", voice="DB30", language="ja")
    assert (result.task, result.language_id) == ("MULTILINGUAL", "ja")
    assert result.prompt_text == "感谢你的认可。"


def test_style_routes_to_voice_design(plan):
    result = plan(text="Glaciers move slowly.", style="A calm male documentary narrator.")
    assert result.task == "VOICE_DESIGN"
    assert result.controls == {"style": "A calm male documentary narrator."}
    assert result.use_zero_spk_emb and result.voice is None


def test_instruction_routes_to_freeform(plan):
    result = plan(text="The old radio crackled.", instruction="Warm, intimate storytelling.")
    assert result.task == "FREEFORM"
    assert result.instruction == "Warm, intimate storytelling."
    assert result.controls is None and result.use_zero_spk_emb


def test_instruction_with_a_voice_uses_half_of_its_fingerprint(plan):
    result = plan(text="The old radio crackled.", voice="US_James", instruction="Warm storytelling.")
    assert result.task == "FREEFORM" and result.voice == "US_James"
    assert result.use_spk_emb and not result.use_zero_spk_emb
    assert result.spk_emb_path.endswith("fingerprints_direction/US_James.pt")


def test_instruction_works_with_a_voice_that_skips_the_plain_fingerprint(plan):
    result = plan(text="The old radio crackled.", voice="US_Emma", instruction="Warm storytelling.")
    assert result.spk_emb_path.endswith("fingerprints_direction/US_Emma.pt")


def test_emotion_control_needs_only_the_speaker_embedding(plan):
    result = plan(text="所有测试都通过了！", voice="lingguang", controls={"emotion": "高兴"})
    assert result.task == "EMOTION"
    assert result.controls == {"emotion": "高兴"}
    assert result.prompt_wav_path and result.use_spk_emb
    assert result.prompt_text is None


def test_basic_controls_fill_missing_attributes(plan):
    result = plan(text="今晚的观测将在云层散开后开始。", voice="lingguang", controls={"speed": "快速"})
    assert result.task == "BASIC"
    assert result.controls == {"speed": "快速", "pitch": "中", "volume": "中"}


def test_dialect_control_drops_language_id(plan):
    result = plan(text="今日天气几好。", voice="lingguang", controls={"dialect": "cantonese"})
    assert (result.task, result.language_id) == ("DIALECT", None)


@pytest.mark.parametrize("body, error", [
    ({"text": "   ", "voice": "US_James"}, BadRequest),
    ({"text": "hi"}, BadRequest),
    ({"text": "hi", "voice": "US_James", "style": "deep voice"}, BadRequest),
    ({"text": "hi", "voice": "lingguang", "instruction": "whisper"}, BadRequest),
    ({"text": "hi", "voice": "悟空", "instruction": "whisper"}, BadRequest),
    ({"text": "hi", "style": "deep voice", "instruction": "whisper"}, BadRequest),
    ({"text": "hi", "style": "deep voice", "controls": {"emotion": "高兴"}}, BadRequest),
    ({"text": "hi", "voice": "US_James", "controls": {"emotion": "高兴"}}, BadRequest),
    ({"text": "hi", "voice": "lingguang", "controls": {"emotion": "happy"}}, BadRequest),
    ({"text": "hi", "voice": "lingguang", "controls": {"speed": "fast"}}, BadRequest),
    ({"text": "hi", "voice": "lingguang", "controls": {"dialect": "texan"}}, BadRequest),
    ({"text": "hi", "voice": "lingguang", "controls": {"emotion": "高兴", "speed": "快速"}}, BadRequest),
    ({"text": "hi", "voice": "lingguang", "controls": {"tone": "x"}}, BadRequest),
    ({"text": "hi", "voice": "nobody"}, NotFound),
])
def test_invalid_combinations(plan, body, error):
    with pytest.raises(error):
        plan(**body)


def test_text_limit_is_413(plan):
    with pytest.raises(PayloadTooLarge):
        plan(max_text_chars=10, text="x" * 11, voice="US_James")


def test_ming_validation_is_applied(tts_tasks):
    bad = SynthesisPlan(task="MULTILINGUAL", text="Hallo.", language="de", language_id=None)
    with pytest.raises(BadRequest, match="requires language_id"):
        engine._validate_with_ming(bad, tts_tasks)


def test_every_route_passes_ming_validation(plan, tts_tasks):
    bodies = [
        {"voice": "US_James"}, {"voice": "lingguang"}, {"voice": "DB30", "language": "fr"},
        {"style": "bright"}, {"instruction": "slow"},
        {"voice": "lingguang", "controls": {"emotion": "悲伤"}},
        {"voice": "lingguang", "controls": {"volume": "高"}},
        {"voice": "lingguang", "controls": {"dialect": "chuanyu dialect"}},
    ]
    for body in bodies:
        result = plan(text="Hello there.", **body)
        kwargs = result.synthesize_kwargs()
        kwargs.pop("max_length")
        request = tts_tasks.build_task_generation_kwargs(kwargs.pop("task"), **kwargs)
        assert request["text"] == "Hello there."


def test_payload_roundtrip(plan):
    result = plan(text="Hello.", voice="lingguang", sample_rate=44100, cfg=1.5)
    assert SynthesisPlan.from_payload(result.to_payload()) == result


def test_display_fragments_keep_the_callers_words(ming_dir):
    engine.add_ming_to_path(ming_dir)
    text = ("I bought 42 apples at the market this morning, and then I walked "
            "all the way home along the river while the sun came up over the hills.")
    fragments = engine.display_fragments(text, None, 24)
    assert len(fragments) > 1
    assert "42" in fragments[0]
    assert " ".join(fragments).replace("  ", " ").startswith("I bought 42 apples")


def test_planning_without_a_ming_checkout_skips_value_checks(registry):
    result = plan_request(
        TTSRequest(text="hi", voice="lingguang", controls={"emotion": "anything"}),
        registry, None, 1200)
    assert result.task == "EMOTION"
