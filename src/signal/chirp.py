"""
Biomimetic CF-FM chirp generation inspired by chiropteran sonar.

The signal has two components:
- CF (Constant Frequency): detects velocity via Doppler shift
- FM (Frequency Modulated): provides range precision via time-of-flight

Reference: Simmons (1979) — "Perception of echo phase information in bat sonar"
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# Reference parameters — do not modify without regression test
CHIRP_PARAMS = {
    "cf_freq": 20_000,        # Hz — Doppler component
    "fm_start": 20_000,       # Hz — sweep start (descending)
    "fm_end": 1_000,          # Hz — sweep end (descending)
    "cf_duration": 0.005,     # seconds
    "fm_duration": 0.015,     # seconds
    "sample_rate": 44_100,    # Hz
    "amplitude": 0.8,         # 0–1
}


def generate_cf(
    freq: float,
    duration: float,
    sample_rate: int,
    amplitude: float,
) -> NDArray[np.float64]:
    """Generate CF (constant frequency) component.

    Args:
        freq: frequency in Hz
        duration: duration in seconds
        sample_rate: sampling rate in Hz
        amplitude: peak amplitude (0-1)

    Returns:
        CF signal — shape: (n_samples,)
    """
    t = np.arange(int(duration * sample_rate)) / sample_rate
    # t: (n_samples,) — float64
    return amplitude * np.sin(2 * np.pi * freq * t)


def generate_fm(
    f_start: float,
    f_end: float,
    duration: float,
    sample_rate: int,
    amplitude: float,
) -> NDArray[np.float64]:
    """Generate FM component with exponential descending sweep.

    Exponential sweep distributes energy more uniformly on a logarithmic
    scale, aligned with human auditory perception and chiropteran response.

    Args:
        f_start: start frequency in Hz
        f_end: end frequency in Hz
        duration: duration in seconds
        sample_rate: sampling rate in Hz
        amplitude: peak amplitude (0-1)

    Returns:
        FM signal — shape: (n_samples,)
    """
    n_samples = int(duration * sample_rate)
    t = np.arange(n_samples) / sample_rate
    # t: (n_samples,) — float64

    # Exponential sweep: f(t) = f_start * (f_end/f_start)^(t/duration)
    # Instantaneous phase: integral of 2*pi*f(t) dt
    ratio = f_end / f_start
    beta = duration / np.log(ratio)
    phase = 2 * np.pi * f_start * beta * (np.power(ratio, t / duration) - 1)
    # phase: (n_samples,) — float64

    return amplitude * np.sin(phase)


def generate_chirp(
    cf_freq: float | None = None,
    fm_start: float | None = None,
    fm_end: float | None = None,
    cf_duration: float | None = None,
    fm_duration: float | None = None,
    sample_rate: int | None = None,
    amplitude: float | None = None,
) -> NDArray[np.float64]:
    """Generate complete CF-FM chirp by concatenating CF and FM components.

    All parameters are optional; defaults from CHIRP_PARAMS are used.

    Returns:
        CF-FM chirp — shape: (n_cf + n_fm,) — float64
    """
    cf_freq = cf_freq or CHIRP_PARAMS["cf_freq"]
    fm_start = fm_start or CHIRP_PARAMS["fm_start"]
    fm_end = fm_end or CHIRP_PARAMS["fm_end"]
    cf_duration = cf_duration or CHIRP_PARAMS["cf_duration"]
    fm_duration = fm_duration or CHIRP_PARAMS["fm_duration"]
    sample_rate = sample_rate or CHIRP_PARAMS["sample_rate"]
    amplitude = amplitude or CHIRP_PARAMS["amplitude"]

    cf = generate_cf(cf_freq, cf_duration, sample_rate, amplitude)
    # cf: (n_cf,) — float64
    fm = generate_fm(fm_start, fm_end, fm_duration, sample_rate, amplitude)
    # fm: (n_fm,) — float64

    # Crossfade at CF→FM transition to avoid clicks
    overlap = min(32, len(cf), len(fm))
    fade_out = np.linspace(1.0, 0.0, overlap)
    fade_in = np.linspace(0.0, 1.0, overlap)
    cf[-overlap:] *= fade_out
    fm[:overlap] *= fade_in

    chirp = np.concatenate([cf, fm])
    # chirp: (n_cf + n_fm,) — float64
    return chirp


def chirp_duration(sample_rate: int | None = None) -> float:
    """Total chirp duration in seconds."""
    sr = sample_rate or CHIRP_PARAMS["sample_rate"]
    return len(generate_chirp(sample_rate=sr)) / sr


def add_silence(
    signal: NDArray[np.float64],
    pre_ms: float = 0.0,
    post_ms: float = 100.0,
    sample_rate: int | None = None,
) -> NDArray[np.float64]:
    """Add silence before and/or after the signal.

    Useful for simulating time-of-flight: the echo appears in the
    trailing silence region.

    Args:
        signal: input signal — shape: (n,)
        pre_ms: silence before in milliseconds
        post_ms: silence after in milliseconds
        sample_rate: sampling rate

    Returns:
        Signal with silence — shape: (n + pre + post,)
    """
    sr = sample_rate or CHIRP_PARAMS["sample_rate"]
    pre_samples = int(pre_ms * sr / 1000)
    post_samples = int(post_ms * sr / 1000)
    return np.concatenate([
        np.zeros(pre_samples),
        signal,
        np.zeros(post_samples),
    ])
