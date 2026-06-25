"""Tests for stair detection and geometry estimation."""

import numpy as np
import pytest

from src.signal.stairs import (
    STAIR_PRIOR,
    build_stair_message,
    detect_stair_periodicity,
    estimate_stair_direction,
    estimate_stair_geometry,
    synthesize_stair_rir,
)
from src.features.geometric import SPEED_OF_SOUND


class TestDetectStairPeriodicity:
    def test_synthetic_stair_rir(self):
        """Synthetic RIR with 0.29m periodicity should be detected
        with confidence > 0.5."""
        sr = 44100
        rir = synthesize_stair_rir(
            n_steps=8,
            tread_m=0.29,
            sample_rate=sr,
        )
        result = detect_stair_periodicity(rir, sr)

        assert result["is_stair"]
        assert result["confidence"] > 0.5
        assert result["n_steps_detected"] >= 4
        assert abs(result["echo_spacing_m"] - 0.29) < STAIR_PRIOR["tread_tolerance"]

    def test_many_steps(self):
        """12-step staircase should be detected."""
        sr = 44100
        rir = synthesize_stair_rir(n_steps=12, sample_rate=sr, rir_length_s=0.15)
        result = detect_stair_periodicity(rir, sr)

        assert result["is_stair"]
        assert result["n_steps_detected"] >= 6

    def test_flat_surface_not_detected(self):
        """Flat wall RIR (no periodicity) should not be detected as stairs."""
        sr = 44100
        n = int(0.1 * sr)
        rir = np.zeros(n)
        rir[0] = 1.0
        delay = int(2 * 2.0 / SPEED_OF_SOUND * sr)
        if delay < n:
            rir[delay] = 0.5

        result = detect_stair_periodicity(rir, sr)
        assert not result["is_stair"]

    def test_irregular_spacing_not_detected(self):
        """Non-periodic echoes should not be detected as stairs."""
        sr = 44100
        n = int(0.1 * sr)
        rir = np.zeros(n)
        rir[0] = 1.0
        for dist in [0.5, 1.2, 2.8, 5.0]:
            delay = int(2 * dist / SPEED_OF_SOUND * sr)
            if delay < n:
                rir[delay] = 0.3

        result = detect_stair_periodicity(rir, sr)
        assert not result["is_stair"]


class TestEstimateStairGeometry:
    def test_standard_staircase(self):
        """With spacing=0.29m and n_steps=8, verify geometry."""
        geometry = estimate_stair_geometry(
            echo_spacing_m=0.29,
            n_steps=8,
        )

        assert abs(geometry["run_total_m"] - 2.32) < 0.01
        assert abs(geometry["rise_total_m"] - 1.40) < 0.01
        assert geometry["prior_used"]
        assert geometry["tread_m"] == 0.29
        assert geometry["riser_m"] == STAIR_PRIOR["riser_m"]

    def test_narrow_tread(self):
        """Narrower tread (0.27m) should still compute correctly."""
        geometry = estimate_stair_geometry(
            echo_spacing_m=0.27,
            n_steps=10,
        )

        assert abs(geometry["run_total_m"] - 2.70) < 0.01
        assert geometry["prior_used"]


class TestEstimateStairDirection:
    def test_ascending(self):
        assert estimate_stair_direction(15.0) == "ascending"
        assert estimate_stair_direction(45.0) == "ascending"

    def test_descending(self):
        assert estimate_stair_direction(-15.0) == "descending"
        assert estimate_stair_direction(-30.0) == "descending"

    def test_undetermined(self):
        assert estimate_stair_direction(5.0) == "undetermined"
        assert estimate_stair_direction(-5.0) == "undetermined"
        assert estimate_stair_direction(0.0) == "undetermined"


class TestBuildStairMessage:
    def test_ascending(self):
        geometry = estimate_stair_geometry(0.29, 8)
        msg = build_stair_message(geometry, "ascending")
        assert "Ascending" in msg
        assert "8 steps" in msg
        assert "2.3" in msg
        assert "Height" in msg

    def test_descending(self):
        geometry = estimate_stair_geometry(0.29, 10)
        msg = build_stair_message(geometry, "descending")
        assert "Descending" in msg
        assert "10 steps" in msg

    def test_undetermined(self):
        geometry = estimate_stair_geometry(0.29, 6)
        msg = build_stair_message(geometry, "undetermined")
        assert "detected" in msg
        assert "Height" not in msg


class TestSynthesizeStairRIR:
    def test_shape(self):
        rir = synthesize_stair_rir(n_steps=8, sample_rate=44100)
        expected = int(0.1 * 44100)
        assert rir.shape == (expected,)

    def test_direct_impulse(self):
        rir = synthesize_stair_rir(n_steps=5)
        assert rir[0] == 1.0

    def test_periodic_peaks(self):
        """Verify peaks appear at regular intervals."""
        sr = 44100
        tread = 0.29
        n_steps = 6
        rir = synthesize_stair_rir(
            n_steps=n_steps,
            tread_m=tread,
            sample_rate=sr,
            base_distance_m=0.5,
        )

        # Find non-zero positions (excluding direct impulse)
        peak_indices = np.where(rir[1:] > 0)[0] + 1
        assert len(peak_indices) == n_steps

        # Verify spacing
        spacings = np.diff(peak_indices)
        expected_spacing = int(2 * tread / SPEED_OF_SOUND * sr)
        for s in spacings:
            assert abs(s - expected_spacing) <= 1

    def test_amplitude_decay(self):
        """Peak amplitudes should decay geometrically."""
        rir = synthesize_stair_rir(n_steps=5, attenuation_per_step=0.8)
        peak_indices = np.where(rir[1:] > 0)[0] + 1
        amplitudes = rir[peak_indices]
        # Each should be ~0.8x the previous
        for i in range(1, len(amplitudes)):
            ratio = amplitudes[i] / amplitudes[i-1]
            assert abs(ratio - 0.8) < 0.01
