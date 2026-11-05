"""Keep tone and loudness steady across one synthesis.

Ming's tone and level rise and fall with every sentence, which is obvious on
headphones over a long text. This pulls six bands and the overall level toward
the audio's own average, slowly and within limits, and lowers the noise floor
in pauses so it does not change from sentence to sentence. It reduces the swing
without removing it: the drop in clarity along a sentence comes from the model.
"""

from __future__ import annotations

import numpy as np

FRAME = 2048
HOP = 512
BAND_EDGES_HZ = (0, 500, 1500, 3000, 6000, 12000, 22050)
MAX_BAND_DB = 6.0
MAX_LEVEL_DB = 4.0
VOICED_BELOW_PEAK_DB = 30.0
GATE_BELOW_PEAK_DB = 50.0
GATE_RANGE_DB = 8.0
GATE_DEPTH_DB = -30.0
PEAK_LIMIT = 0.98
MIN_SECONDS = 1.0

_WINDOW = np.hanning(FRAME + 1)[:-1]


def _stft(samples: np.ndarray) -> np.ndarray:
    padded = np.pad(samples, FRAME // 2, mode="reflect")
    count = 1 + (len(padded) - FRAME) // HOP
    starts = HOP * np.arange(count)
    frames = np.stack([padded[start:start + FRAME] * _WINDOW for start in starts], axis=1)
    return np.fft.rfft(frames, axis=0)


def _istft(spectrum: np.ndarray, length: int) -> np.ndarray:
    frames = np.fft.irfft(spectrum, FRAME, axis=0) * _WINDOW[:, None]
    size = FRAME + HOP * (spectrum.shape[1] - 1)
    output = np.zeros(size)
    weight = np.zeros(size)
    for index in range(spectrum.shape[1]):
        start = index * HOP
        output[start:start + FRAME] += frames[:, index]
        weight[start:start + FRAME] += _WINDOW ** 2
    output /= np.maximum(weight, 1e-8)
    return output[FRAME // 2:FRAME // 2 + length]


def _smooth(values: np.ndarray, seconds: float, sample_rate: int) -> np.ndarray:
    size = max(3, int(seconds * sample_rate / HOP) | 1)
    kernel = np.hanning(size + 2)[1:-1]
    kernel /= kernel.sum()
    padded = np.pad(values, size // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def level(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Return ``samples`` (mono float) leveled, same length; short audio is unchanged."""
    if len(samples) < MIN_SECONDS * sample_rate or np.abs(samples).max() < 1e-6:
        return samples
    spectrum = _stft(samples.astype(np.float64))
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(FRAME, 1 / sample_rate)
    edges = list(zip(BAND_EDGES_HZ, BAND_EDGES_HZ[1:]))
    bands = np.stack([power[(freqs >= low) & (freqs < high)].sum(axis=0) for low, high in edges])
    total = bands.sum(axis=0) + 1e-12
    loudness = 10 * np.log10(total)
    peak = np.percentile(loudness, 95)
    voiced = loudness > peak - VOICED_BELOW_PEAK_DB
    if voiced.sum() < 5:
        return samples

    smooth_total = _smooth(total, 0.4, sample_rate)
    balance = 10 * np.log10(
        np.stack([_smooth(band, 0.4, sample_rate) for band in bands]) / (smooth_total + 1e-12) + 1e-12)
    band_gain = np.clip(balance[:, voiced].mean(axis=1, keepdims=True) - balance, -MAX_BAND_DB, MAX_BAND_DB)
    band_gain[:, ~voiced] = 0
    band_gain = np.stack([_smooth(gain, 0.15, sample_rate) for gain in band_gain])

    slow = _smooth(loudness, 0.5, sample_rate)
    level_gain = np.clip(np.median(slow[voiced]) - slow, -MAX_LEVEL_DB, MAX_LEVEL_DB)
    level_gain[~voiced] = 0
    level_gain = _smooth(level_gain, 0.2, sample_rate)

    centers = np.array([(low + high) / 2 for low, high in edges])
    curve = np.stack([np.interp(freqs, centers, band_gain[:, frame]) + level_gain[frame]
                      for frame in range(band_gain.shape[1])], axis=1)
    gate = _smooth(np.clip((loudness - (peak - GATE_BELOW_PEAK_DB)) / GATE_RANGE_DB, 0, 1), 0.03, sample_rate)
    floor = 10 ** (GATE_DEPTH_DB / 20)
    gains = 10 ** (curve / 20) * (floor + (1 - floor) * gate)

    output = _istft(spectrum * gains, len(samples))
    return (output / max(1.0, np.abs(output).max() / PEAK_LIMIT)).astype(np.float32)
