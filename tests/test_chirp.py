"""Tests for CF-FM chirp generation."""

import numpy as np
import pytest

from src.signal.chirp import (
    CHIRP_PARAMS,
    add_silence,
    chirp_duration,
    generate_cf,
    generate_chirp,
    generate_fm,
)


class TestGenerateCF:
    def test_shape(self):
        sr = 44100
        dur = 0.005
        cf = generate_cf(20000, dur, sr, 0.8)
        expected_samples = int(dur * sr)
        assert cf.shape == (expected_samples,)

    def test_amplitude(self):
        cf = generate_cf(1000, 0.01, 44100, 0.5)
        assert np.max(np.abs(cf)) <= 0.5 + 1e-6

    def test_frequency(self):
        """Verify dominant frequency matches expected."""
        sr = 44100
        freq = 5000
        cf = generate_cf(freq, 0.02, sr, 0.8)
        spectrum = np.abs(np.fft.rfft(cf))
        freqs = np.fft.rfftfreq(len(cf), 1.0 / sr)
        dominant_freq = freqs[np.argmax(spectrum)]
        assert abs(dominant_freq - freq) < sr / len(cf) * 2


class TestGenerateFM:
    def test_shape(self):
        sr = 44100
        dur = 0.015
        fm = generate_fm(20000, 1000, dur, sr, 0.8)
        expected_samples = int(dur * sr)
        assert fm.shape == (expected_samples,)

    def test_amplitude(self):
        fm = generate_fm(20000, 1000, 0.015, 44100, 0.8)
        assert np.max(np.abs(fm)) <= 0.8 + 1e-6

    def test_frequency_sweep(self):
        """Verify frequency content sweeps from high to low."""
        sr = 44100
        fm = generate_fm(10000, 1000, 0.02, sr, 0.8)
        half = len(fm) // 2
        spec_first = np.abs(np.fft.rfft(fm[:half]))
        spec_second = np.abs(np.fft.rfft(fm[half:]))
        freqs_first = np.fft.rfftfreq(half, 1.0 / sr)
        freqs_second = np.fft.rfftfreq(len(fm) - half, 1.0 / sr)
        dominant_first = freqs_first[np.argmax(spec_first)]
        dominant_second = freqs_second[np.argmax(spec_second)]
        assert dominant_first > dominant_second


class TestGenerateChirp:
    def test_default_params(self):
        chirp = generate_chirp()
        sr = CHIRP_PARAMS["sample_rate"]
        expected_cf = int(CHIRP_PARAMS["cf_duration"] * sr)
        expected_fm = int(CHIRP_PARAMS["fm_duration"] * sr)
        assert chirp.shape == (expected_cf + expected_fm,)

    def test_dtype(self):
        chirp = generate_chirp()
        assert chirp.dtype == np.float64

    def test_amplitude_bounded(self):
        chirp = generate_chirp()
        assert np.max(np.abs(chirp)) <= CHIRP_PARAMS["amplitude"] + 1e-6

    def test_custom_params(self):
        chirp = generate_chirp(cf_freq=10000, fm_start=10000, fm_end=500)
        assert len(chirp) > 0

    def test_no_dc_offset(self):
        chirp = generate_chirp()
        assert abs(np.mean(chirp)) < 0.1


class TestChirpDuration:
    def test_duration(self):
        dur = chirp_duration()
        expected = CHIRP_PARAMS["cf_duration"] + CHIRP_PARAMS["fm_duration"]
        assert abs(dur - expected) < 1e-3


class TestAddSilence:
    def test_pre_silence(self):
        signal = np.ones(100)
        result = add_silence(signal, pre_ms=10, post_ms=0, sample_rate=44100)
        pre_samples = int(10 * 44100 / 1000)
        assert len(result) == 100 + pre_samples
        assert np.all(result[:pre_samples] == 0)

    def test_post_silence(self):
        signal = np.ones(100)
        result = add_silence(signal, pre_ms=0, post_ms=10, sample_rate=44100)
        post_samples = int(10 * 44100 / 1000)
        assert len(result) == 100 + post_samples
        assert np.all(result[-post_samples:] == 0)
