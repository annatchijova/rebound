"""
Haptic feedback patterns (simulated for demo).

In production, these patterns would translate to vibrations on a
mobile device. In the hackathon demo, they are visualized on screen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class HapticPattern:
    """Vibration haptic pattern."""
    name: str
    intensities: list[float]   # 0-1 per time slot
    durations_ms: list[float]  # duration of each slot
    description: str


# Predefined patterns
PATTERNS: dict[str, HapticPattern] = {
    "none": HapticPattern(
        name="none",
        intensities=[0.0],
        durations_ms=[100],
        description="No feedback",
    ),
    "single_pulse": HapticPattern(
        name="single_pulse",
        intensities=[0.7, 0.0],
        durations_ms=[150, 100],
        description="Single pulse — corridor detected",
    ),
    "double_pulse": HapticPattern(
        name="double_pulse",
        intensities=[0.8, 0.0, 0.8, 0.0],
        durations_ms=[100, 50, 100, 50],
        description="Double pulse — nearby wall",
    ),
    "double_pulse_slow": HapticPattern(
        name="double_pulse_slow",
        intensities=[0.6, 0.0, 0.6, 0.0],
        durations_ms=[200, 150, 200, 150],
        description="Slow double pulse — doorway",
    ),
    "continuous_low": HapticPattern(
        name="continuous_low",
        intensities=[0.3],
        durations_ms=[500],
        description="Low continuous vibration — corner",
    ),
    "continuous_high": HapticPattern(
        name="continuous_high",
        intensities=[0.9],
        durations_ms=[500],
        description="High continuous vibration — very close obstacle",
    ),
    "stair_alert": HapticPattern(
        name="stair_alert",
        intensities=[0.8, 0.0, 0.5, 0.0, 0.5, 0.0],
        durations_ms=[300, 100, 100, 50, 100, 50],
        description="Long-short-short (Morse S) — staircase detected",
    ),
}


def get_pattern(name: str) -> HapticPattern:
    """Get a pattern by name."""
    return PATTERNS.get(name, PATTERNS["none"])


def pattern_to_waveform(
    pattern: HapticPattern,
    sample_rate: int = 1000,
) -> NDArray[np.float64]:
    """Convert haptic pattern to waveform for visualization.

    Args:
        pattern: pattern to convert
        sample_rate: samples per second (for visualization)

    Returns:
        Waveform — shape: (n_samples,)
    """
    segments = []
    for intensity, duration_ms in zip(pattern.intensities, pattern.durations_ms):
        n_samples = int(duration_ms * sample_rate / 1000)
        segments.append(np.full(n_samples, intensity))

    return np.concatenate(segments) if segments else np.zeros(1)
