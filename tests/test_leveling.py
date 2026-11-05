import numpy as np

from app.leveling import level

SAMPLE_RATE = 44100


def speech_like(seconds: float, *, dull_after: float | None = None) -> np.ndarray:
    """Harmonic tone with short pauses; optionally its upper band fades later on."""
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    bright = np.ones_like(t) if dull_after is None else np.where(t < dull_after, 1.0, 0.25)
    signal = 0.4 * np.sin(2 * np.pi * 150 * t) + 0.2 * bright * np.sin(2 * np.pi * 4000 * t)
    pauses = (t % 2.0) < 1.7
    return (signal * pauses).astype(np.float32)


def band_energy_db(samples: np.ndarray, start: float, end: float, low: float, high: float) -> float:
    part = samples[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]
    power = np.abs(np.fft.rfft(part)) ** 2
    freqs = np.fft.rfftfreq(len(part), 1 / SAMPLE_RATE)
    return float(10 * np.log10(power[(freqs >= low) & (freqs < high)].sum() / power.sum() + 1e-12))


def test_keeps_length_and_stays_in_range():
    samples = speech_like(6.0)
    output = level(samples, SAMPLE_RATE)
    assert output.shape == samples.shape
    assert output.dtype == np.float32
    assert np.abs(output).max() <= 0.98 + 1e-6


def test_short_audio_is_unchanged():
    samples = speech_like(0.5)
    assert level(samples, SAMPLE_RATE) is samples


def test_silence_is_unchanged():
    samples = np.zeros(SAMPLE_RATE * 3, dtype=np.float32)
    assert level(samples, SAMPLE_RATE) is samples


def test_reduces_a_fall_in_brightness():
    samples = speech_like(8.0, dull_after=4.0)
    before = band_energy_db(samples, 0.2, 1.5, 3000, 6000) - band_energy_db(samples, 4.2, 5.5, 3000, 6000)
    output = level(samples, SAMPLE_RATE)
    after = band_energy_db(output, 0.2, 1.5, 3000, 6000) - band_energy_db(output, 4.2, 5.5, 3000, 6000)
    assert after < before - 3


def test_lowers_noise_in_pauses():
    t = np.arange(SAMPLE_RATE * 6) / SAMPLE_RATE
    speech = 0.4 * np.sin(2 * np.pi * 150 * t) * ((t % 2.0) < 1.5)
    noise = 0.0005 * np.random.default_rng(1).standard_normal(len(t))
    samples = (speech + noise).astype(np.float32)
    pause = (t > 1.65) & (t < 1.9)
    before = np.sqrt(np.mean(samples[pause] ** 2))
    after = np.sqrt(np.mean(level(samples, SAMPLE_RATE)[pause] ** 2))
    assert after < before / 3
