"""Tests for spectral and geometric feature extraction."""

import numpy as np
import pytest

from src.signal.chirp import CHIRP_PARAMS, generate_chirp
from src.features.geometric import (
    SPEED_OF_SOUND,
    detect_reflection_pattern,
    estimate_distance,
    estimate_echo_strength,
)


def _make_synthetic_rir(
    distances_m: list[float],
    attenuations: list[float],
    sample_rate: int = 44100,
    length_s: float = 0.1,
) -> np.ndarray:
    """Create synthetic RIR with peaks at given distances."""
    n = int(length_s * sample_rate)
    rir = np.zeros(n)
    rir[0] = 1.0

    for dist, att in zip(distances_m, attenuations):
        delay_s = 2 * dist / SPEED_OF_SOUND
        delay_samples = int(delay_s * sample_rate)
        if delay_samples < n:
            rir[delay_samples] = att

    return rir


class TestEstimateDistance:
    def test_known_distance(self):
        """Verify correct detection of a reflector at 2m."""
        rir = _make_synthetic_rir([2.0], [0.5])
        dist = estimate_distance(rir, sample_rate=44100)
        assert abs(dist - 2.0) < 0.1, f"Estimated distance: {dist}, expected: 2.0"

    def test_close_reflector(self):
        """Reflector at 0.5m."""
        rir = _make_synthetic_rir([0.5], [0.6])
        dist = estimate_distance(rir, sample_rate=44100)
        assert abs(dist - 0.5) < 0.1

    def test_no_reflection(self):
        """No reflections, distance = 0."""
        rir = np.zeros(4410)
        rir[0] = 1.0
        dist = estimate_distance(rir, sample_rate=44100)
        assert dist == 0.0

    def test_multiple_reflectors_returns_first(self):
        """With multiple reflectors, returns the nearest one."""
        rir = _make_synthetic_rir([1.0, 3.0, 5.0], [0.5, 0.3, 0.2])
        dist = estimate_distance(rir, sample_rate=44100)
        assert abs(dist - 1.0) < 0.15


class TestEchoStrength:
    def test_strong_echo(self):
        rir = _make_synthetic_rir([2.0], [0.8])
        strength = estimate_echo_strength(rir, sample_rate=44100)
        assert -5 < strength < 0

    def test_weak_echo(self):
        rir = _make_synthetic_rir([2.0], [0.1])
        strength = estimate_echo_strength(rir, sample_rate=44100)
        assert strength < -15


class TestReflectionPattern:
    def test_corridor_pattern(self):
        """Corridor: many regular peaks (parallel lateral reflections)."""
        distances = [1.0, 2.0, 3.0, 4.0, 5.0]
        attenuations = [0.5, 0.3, 0.2, 0.15, 0.1]
        rir = _make_synthetic_rir(distances, attenuations, length_s=0.2)
        pattern = detect_reflection_pattern(rir, sample_rate=44100)
        assert pattern["n_peaks"] >= 4
        assert pattern["first_distance_m"] > 0

    def test_open_space(self):
        """Open space: few peaks, slow decay."""
        rir = _make_synthetic_rir([8.0], [0.1], length_s=0.2)
        pattern = detect_reflection_pattern(rir, sample_rate=44100)
        assert pattern["n_peaks"] <= 3
