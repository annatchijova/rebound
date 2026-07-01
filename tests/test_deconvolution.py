"""Tests for Wiener deconvolution and RIR estimation."""

import numpy as np
import pytest

from src.signal.chirp import generate_chirp, add_silence, CHIRP_PARAMS
from src.signal.deconvolution import (
    adaptive_wiener,
    estimate_snr,
    wiener_deconvolution,
)


def _simulate_echo(
    chirp: np.ndarray,
    delay_samples: int,
    attenuation: float = 0.5,
    noise_level: float = 0.01,
) -> np.ndarray:
    """Simulate a received signal with delayed echo + noise."""
    total_len = len(chirp) + delay_samples + 100
    received = np.zeros(total_len)
    received[:len(chirp)] = chirp
    received[delay_samples:delay_samples + len(chirp)] += attenuation * chirp
    received += noise_level * np.random.default_rng(42).standard_normal(total_len)
    return received


class TestWienerDeconvolution:
    def test_rir_has_peak_at_delay(self):
        """RIR should show a peak at the echo position."""
        sr = CHIRP_PARAMS["sample_rate"]
        chirp = generate_chirp()
        delay_m = 2.0
        delay_samples = int(2 * delay_m * sr / 343.0)

        received = _simulate_echo(chirp, delay_samples, attenuation=0.3, noise_level=0.001)
        rir = wiener_deconvolution(received, chirp, regularization=0.01)

        rir_abs = np.abs(rir)
        peak_idx = np.argmax(rir_abs[10:]) + 10

        assert abs(peak_idx - delay_samples) < 10, (
            f"Peak at {peak_idx}, expected near {delay_samples}"
        )

    def test_output_shape(self):
        chirp = generate_chirp()
        received = np.random.default_rng(0).standard_normal(2000)
        rir = wiener_deconvolution(received, chirp)
        n_linear = len(received) + len(chirp) - 1
        assert rir.shape == (n_linear,)

    def test_output_dtype(self):
        chirp = generate_chirp()
        received = np.random.default_rng(0).standard_normal(2000)
        rir = wiener_deconvolution(received, chirp)
        assert rir.dtype == np.float64


class TestAdaptiveWiener:
    def test_high_snr_less_regularization(self):
        """With high SNR, regularization is lower, giving more resolution."""
        chirp = generate_chirp()
        sr = CHIRP_PARAMS["sample_rate"]
        delay_samples = int(2 * 1.5 * sr / 343.0)

        received = _simulate_echo(chirp, delay_samples, attenuation=0.4, noise_level=0.0001)

        rir_high = adaptive_wiener(received, chirp, snr_estimate_db=30)
        rir_low = adaptive_wiener(received, chirp, snr_estimate_db=5)

        peak_high = np.max(np.abs(rir_high))
        peak_low = np.max(np.abs(rir_low))
        assert peak_high > peak_low


class TestEstimateSNR:
    def test_clean_signal(self):
        """Signal with burst followed by noise floor should give high SNR."""
        rng = np.random.default_rng(42)
        n = 6000
        signal = rng.standard_normal(n) * 0.001  # noise floor
        # Add a burst in the first third (simulating chirp echo)
        signal[:2000] += np.sin(2 * np.pi * 1000 * np.arange(2000) / 44100) * 0.5
        snr = estimate_snr(signal)
        assert snr > 10

    def test_noisy_signal(self):
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(1000)
        snr = estimate_snr(noise)
        assert -3 < snr < 3

    def test_with_noise_region(self):
        signal = np.zeros(1000)
        signal[100:200] = 1.0
        snr = estimate_snr(signal, noise_region=(800, 1000))
        assert snr > 20

    def test_reverberant_signal_not_underestimated(self):
        """Reverberant signals should not have SNR severely underestimated.

        A decaying signal with clear quiet windows at the end should have
        the noise floor estimated from those quiet windows, not from the
        reverberant tail.
        """
        rng = np.random.default_rng(42)
        n = 8000
        t = np.arange(n) / 44100
        # Fast decay so the tail is clearly noise-dominated
        signal = np.exp(-200 * t) * np.sin(2 * np.pi * 500 * t)
        signal += rng.standard_normal(n) * 0.001
        snr = estimate_snr(signal)
        assert snr > 5
