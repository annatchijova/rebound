"""
Stair detection and analysis via periodic echo patterns in the RIR.

Stairs produce periodic echoes detectable via autocorrelation. Each step
(tread + riser) reflects the chirp at different time-of-flight values.
The autocorrelation of the extracted RIR shows peaks with regular spacing —
a spectral signature no flat surface can replicate.

Periodicity is the evidence. Standard geometry is the prior.

Prior source: ISO 9386, IBC (International Building Code), Eurocode,
Blondel's rule (2r + t = 63 cm) adopted internationally.

Limitations:
    L-005: Prior based on worldwide standard (0.29 m tread, 0.175 m riser).
           Non-standard stairs produce estimation error.
    L-006: Up/down direction requires gyroscope — manual parameter in desktop demo.
    L-007: Stairs with >20 steps may exceed useful acoustic range of chirp.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

from src.features.geometric import SPEED_OF_SOUND

# Worldwide stair geometry prior — ISO/IBC/Eurocode/Blondel
STAIR_PRIOR = {
    "tread_m": 0.29,          # tread depth — worldwide range: 0.28–0.30 m
    "riser_m": 0.175,         # riser height — worldwide range: 0.17–0.18 m
    "blondel_cm": 63.0,       # 2r + t = 63 cm — Blondel's rule
    "tread_tolerance": 0.02,  # +/- 2 cm — acceptable detection margin
    "riser_tolerance": 0.01,  # +/- 1 cm — acceptable detection margin
}


def detect_stair_periodicity(
    rir: NDArray[np.float64],
    sample_rate: int,
    prominence_high: float = 0.15,
    prominence_low: float = 0.05,
    search_window_frac: float = 0.2,
) -> dict[str, bool | float | int]:
    """Analyze extracted RIR for periodic echo pattern indicating stairs.

    Two-pass algorithm:
        Pass 1: detect strong peaks with high prominence — close stairs.
        Pass 2: if Pass 1 confirms periodicity, search for weaker peaks
                at predicted positions, extending the count of detected
                stairs without introducing false positives.

    Validated empirically (see BLOQUE_1.md): SNR > 35 dB required.

    Returns:
        {
            "is_stair": bool,
            "confidence": float,         # 0.0 – 1.0
            "echo_spacing_m": float,
            "n_steps_detected": int,
        }
    """
    result: dict[str, bool | float | int] = {
        "is_stair": False,
        "confidence": 0.0,
        "echo_spacing_m": 0.0,
        "n_steps_detected": 0,
    }

    rir_abs = np.abs(rir)
    rir_norm = rir_abs / (np.max(rir_abs) + 1e-12)

    tread_min = STAIR_PRIOR["tread_m"] - STAIR_PRIOR["tread_tolerance"]
    tread_max = STAIR_PRIOR["tread_m"] + STAIR_PRIOR["tread_tolerance"]
    lag_min = int(2 * tread_min / SPEED_OF_SOUND * sample_rate)
    min_start = max(lag_min // 2, 3)

    # PASS 1: strong peaks
    peaks_strong, _ = find_peaks(
        rir_norm[min_start:],
        prominence=prominence_high,
        distance=lag_min // 2,
    )
    peaks_strong = peaks_strong + min_start

    if len(peaks_strong) < 3:
        return result

    spacings = np.diff(peaks_strong)
    mean_spacing = float(np.mean(spacings))
    if mean_spacing < 1:
        return result

    cv = float(np.std(spacings) / mean_spacing)
    echo_spacing_m = float(mean_spacing / sample_rate * SPEED_OF_SOUND / 2)

    # Reject if Pass 1 doesn't show stair-like pattern
    if cv > 0.20 or not (tread_min <= echo_spacing_m <= tread_max):
        return result

    # PASS 2: extend with weaker peaks at predicted positions
    extended_peaks = list(peaks_strong)
    search_window = int(mean_spacing * search_window_frac)
    next_expected = peaks_strong[-1] + int(mean_spacing)

    while next_expected + search_window < len(rir_norm):
        start = max(0, next_expected - search_window)
        end = min(len(rir_norm), next_expected + search_window)
        window = rir_norm[start:end]

        local_peaks, _ = find_peaks(window, prominence=prominence_low)
        if len(local_peaks) > 0:
            absolute = local_peaks + start
            best_idx = int(np.argmin(np.abs(absolute - next_expected)))
            extended_peaks.append(int(absolute[best_idx]))
            next_expected = absolute[best_idx] + int(mean_spacing)
        else:
            break

    n_steps = len(extended_peaks)

    periodicity_score = max(0, 1.0 - cv / 0.20)
    count_score = min(1.0, n_steps / 5.0)
    confidence = periodicity_score * 0.5 + count_score * 0.5

    result["is_stair"] = True
    result["confidence"] = float(round(confidence, 3))
    result["echo_spacing_m"] = float(round(echo_spacing_m, 4))
    result["n_steps_detected"] = int(n_steps)

    return result


def estimate_stair_geometry(
    echo_spacing_m: float,
    n_steps: int,
) -> dict[str, float | bool]:
    """Estimate stair dimensions using worldwide prior.

    Args:
        echo_spacing_m: measured distance between consecutive echoes
        n_steps: number of steps detected in the RIR

    Returns:
        {
            "run_total_m": float,     # total horizontal length
            "rise_total_m": float,    # total vertical height
            "tread_m": float,         # tread depth used
            "riser_m": float,         # riser height (from prior)
            "prior_used": bool,       # True if prior was used for riser
        }

    Calculation:
        tread_m    = echo_spacing_m  (measured acoustically)
        riser_m    = STAIR_PRIOR["riser_m"]  (prior — not acoustically measurable)
        run_total  = n_steps * tread_m
        rise_total = n_steps * riser_m
    """
    tread_m = echo_spacing_m
    riser_m = STAIR_PRIOR["riser_m"]

    return {
        "run_total_m": round(n_steps * tread_m, 2),
        "rise_total_m": round(n_steps * riser_m, 2),
        "tread_m": round(tread_m, 4),
        "riser_m": riser_m,
        "prior_used": True,
    }


def estimate_stair_direction(gyroscope_pitch_deg: float) -> str:
    """Infer stair direction from device tilt angle.

    Args:
        gyroscope_pitch_deg: device pitch angle in degrees.
            Positive = pointing up, negative = pointing down.

    Returns:
        "ascending" | "descending" | "undetermined"

    Threshold: +/-10 degrees — within that range returns "undetermined".

    NOTE: In the desktop demo (hackathon) this value is passed as a
    manual parameter. Real gyroscope integration is future work.
    """
    if gyroscope_pitch_deg > 10.0:
        return "ascending"
    elif gyroscope_pitch_deg < -10.0:
        return "descending"
    return "undetermined"


def build_stair_message(geometry: dict[str, float | bool], direction: str) -> str:
    """Build user-facing navigation message for detected stairs.

    Args:
        geometry: output from estimate_stair_geometry
        direction: output from estimate_stair_direction

    Returns:
        Plain-language message for voice synthesis.

    Examples:
        "Descending staircase. 8 steps.
         Approximate length: 2.3 meters. Height: 1.4 meters."

        "Staircase detected. 8 steps.
         Approximate length: 2.3 meters."
    """
    n_steps = round(geometry["run_total_m"] / geometry["tread_m"])
    run = geometry["run_total_m"]
    rise = geometry["rise_total_m"]

    if direction == "ascending":
        prefix = "Ascending staircase"
    elif direction == "descending":
        prefix = "Descending staircase"
    else:
        prefix = "Staircase detected"

    msg = f"{prefix}. {n_steps} steps."
    msg += f" Approximate length: {run:.1f} meters."

    if direction != "undetermined":
        msg += f" Height: {rise:.1f} meters."

    return msg


def synthesize_stair_rir(
    n_steps: int = 8,
    tread_m: float | None = None,
    attenuation_per_step: float = 0.85,
    sample_rate: int = 44_100,
    base_distance_m: float = 0.5,
    rir_length_s: float = 0.1,
) -> NDArray[np.float64]:
    """Synthesize a RIR with periodic peaks simulating stair echoes.

    Useful for testing and dataset augmentation. Each step produces
    an echo at increasing delay with geometric attenuation.

    Args:
        n_steps: number of stair steps
        tread_m: tread depth (None = use prior)
        attenuation_per_step: amplitude multiplier per step
        sample_rate: sampling rate
        base_distance_m: distance to first step
        rir_length_s: total RIR length in seconds

    Returns:
        Synthetic stair RIR — shape: (n_samples,)
    """
    tread = tread_m or STAIR_PRIOR["tread_m"]
    n_samples = int(rir_length_s * sample_rate)
    rir = np.zeros(n_samples)

    # Direct impulse
    rir[0] = 1.0

    for i in range(n_steps):
        distance = base_distance_m + i * tread
        delay_s = 2 * distance / SPEED_OF_SOUND  # round-trip
        delay_samples = int(delay_s * sample_rate)

        amplitude = attenuation_per_step ** (i + 1)

        if delay_samples < n_samples:
            rir[delay_samples] += amplitude

    return rir
