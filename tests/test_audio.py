import base64
import io
import wave

import numpy as np
import pytest
import soundfile as sf

from app.audio import encode_audio, float_to_pcm16, pcm16_to_wav, pcm_duration_seconds


def test_float_to_pcm16_clips_and_scales():
    pcm = float_to_pcm16(np.array([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0]))
    values = np.frombuffer(pcm, dtype="<i2").tolist()
    assert values == [-32767, -32767, 0, 16383, 32767, 32767]


def test_float_to_pcm16_flattens_channel_dim():
    assert len(float_to_pcm16(np.zeros((1, 100), dtype=np.float32))) == 200


def test_wav_roundtrip_keeps_samples_and_rate():
    samples = (np.sin(np.linspace(0, 20, 2400)) * 0.5).astype(np.float32)
    pcm = float_to_pcm16(samples)
    wav = pcm16_to_wav(pcm, 24000)
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    with wave.open(io.BytesIO(wav)) as reader:
        assert reader.getframerate() == 24000
        assert reader.getnchannels() == 1
        assert reader.getsampwidth() == 2
        assert reader.readframes(reader.getnframes()) == pcm
    decoded, rate = sf.read(io.BytesIO(wav), dtype="int16")
    assert rate == 24000
    assert decoded.tobytes() == pcm


def test_encode_audio_formats():
    pcm = float_to_pcm16(np.full(441, 0.25))
    assert base64.b64decode(encode_audio(pcm, 44100, "pcm")) == pcm
    assert base64.b64decode(encode_audio(pcm, 44100, "wav")) == pcm16_to_wav(pcm, 44100)
    with pytest.raises(ValueError):
        encode_audio(pcm, 44100, "mp3")


def test_pcm_duration_seconds():
    assert pcm_duration_seconds(b"\x00" * 48000, 24000) == 1.0
