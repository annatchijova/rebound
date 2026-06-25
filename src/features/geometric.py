"""
Geometric property estimation from Room Impulse Response.

- Distance to nearest obstacle (first significant peak)
- Estimated space width (lateral reflection pattern)
- Echo intensity (relative energy of reflections)

Speed of sound: 343 m/s at 20C.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

from src.signal.chirp import CHIRP_PARAMS

SPEED_OF_SOUND = 343.0  # m/s at 20C


def estimate_distance(
    rir: NDArray[np.float64],
    sample_rate: int | None = None,
    min_distance_m: float = 0.15,
    prominence: float = 0.1,
) -> float:
    """Estimate distance to nearest reflector from RIR.

    Finds the first significant peak in the RIR after the direct impulse.
    Distance is d = (t_peak * v_sound) / 2 because sound travels
    round-trip.

    Args:
        rir: Room Impulse Response — shape: (n,)
        sample_rate: sampling rate
        min_distance_m: minimum distance to consider (filters artifacts)
        prominence: minimum peak prominence (relative to maximum)

    Returns:
        Estimated distance in meters. 0.0 if no peak detected.
    """
    sr = sample_rate or CHIRP_PARAMS["sample_rate"]

    rir_abs = np.abs(rir)
    rir_norm = rir_abs / (np.max(rir_abs) + 1e-12)
    # rir_norm: (n,) — float64, range [0, 1]

    # Minimum index corresponding to min_distance_m
    min_samples = int(2 * min_distance_m * sr / SPEED_OF_SOUND)

    peaks, properties = find_peaks(
        rir_norm[min_samples:],
        prominence=prominence,
        distance=int(0.001 * sr),  # minimum 1 ms between peaks
    )

    if len(peaks) == 0:
        return 0.0

    # First significant peak
    first_peak = peaks[0] + min_samples
    time_s = first_peak / sr
    distance = (time_s * SPEED_OF_SOUND) / 2.0
    return float(distance)


def estimate_echo_strength(
    rir: NDArray[np.float64],
    sample_rate: int | None = None,
    min_distance_m: float = 0.15,
) -> float:
    """Compute relative intensity of first echo in dB.

    Ratio between echo energy and direct impulse energy.

    Args:
        rir: Room Impulse Response — shape: (n,)
        sample_rate: sampling rate
        min_distance_m: minimum distance for echo search

    Returns:
        Echo intensity in dB (negative = echo weaker than direct)
    """
    sr = sample_rate or CHIRP_PARAMS["sample_rate"]
    min_samples = int(2 * min_distance_m * sr / SPEED_OF_SOUND)

    rir_abs = np.abs(rir)

    # Direct impulse energy (first samples)
    direct_energy = np.max(rir_abs[:min_samples]) if min_samples > 0 else np.max(rir_abs[:10])

    # First echo energy
    echo_energy = np.max(rir_abs[min_samples:]) if min_samples < len(rir_abs) else 0.0

    if direct_energy < 1e-12:
        return -60.0

    ratio = echo_energy / direct_energy
    if ratio < 1e-12:
        return -60.0

    return float(20 * np.log10(ratio))


def detect_reflection_pattern(
    rir: NDArray[np.float64],
    sample_rate: int | None = None,
    prominence: float = 0.05,
) -> dict[str, float | int]:
    """Analyze reflection pattern to infer geometry.

    Returns:
        Dictionary with:
        - "n_peaks": number of detected peaks
        - "first_distance_m": distance to first reflector
        - "echo_strength_db": first echo intensity
        - "mean_peak_interval_ms": mean interval between peaks
        - "peak_decay_rate": peak decay rate (absorption indicator)
    """
    sr = sample_rate or CHIRP_PARAMS["sample_rate"]

    rir_abs = np.abs(rir)
    rir_norm = rir_abs / (np.max(rir_abs) + 1e-12)

    peaks, properties = find_peaks(
        rir_norm,
        prominence=prominence,
        distance=int(0.0005 * sr),  # minimum 0.5 ms between peaks
    )

    result: dict[str, float | int] = {
        "n_peaks": len(peaks),
        "first_distance_m": estimate_distance(rir, sr),
        "echo_strength_db": estimate_echo_strength(rir, sr),
        "mean_peak_interval_ms": 0.0,
        "peak_decay_rate": 0.0,
    }

    if len(peaks) < 2:
        return result

    # Mean interval between peaks
    intervals = np.diff(peaks) / sr * 1000  # ms
    result["mean_peak_interval_ms"] = float(np.mean(intervals))

    # Decay rate: slope of log(amplitude) vs time
    peak_amplitudes = rir_norm[peaks]
    if len(peak_amplitudes) >= 2:
        log_amps = np.log(peak_amplitudes + 1e-12)
        peak_times = peaks / sr
        coeffs = np.polyfit(peak_times, log_amps, 1)
        result["peak_decay_rate"] = float(coeffs[0])

    return result
