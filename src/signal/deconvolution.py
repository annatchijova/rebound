"""
Room Impulse Response (RIR) extraction via Wiener deconvolution.

Given emitted chirp x(t) and captured signal y(t) = x(t) * h(t) + n(t),
estimates h(t) (the RIR) via frequency-domain deconvolution with
Wiener regularization for numerical stability.

DISCARDED: direct cross-correlation — produces artifacts when SNR < 10 dB.
REASON: peak detector fails with nearby wall echoes (< 0.5 m).
ADOPTED: Wiener deconvolution with adaptive regularization.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import fft as sp_fft


def wiener_deconvolution(
    received: NDArray[np.float64],
    reference: NDArray[np.float64],
    regularization: float = 0.01,
) -> NDArray[np.float64]:
    """Estimate RIR via frequency-domain Wiener deconvolution.

    H(f) = Y(f) * X*(f) / (|X(f)|^2 + epsilon)

    where epsilon is the regularization factor preventing division
    by zero and controlling the resolution/noise trade-off.

    Args:
        received: captured signal y(t) — shape: (n,)
        reference: emitted chirp x(t) — shape: (m,)
        regularization: epsilon factor for stability. Typical values:
            0.001 for high SNR (>20 dB), 0.01-0.1 for moderate SNR.

    Returns:
        Estimated RIR h(t) — shape: (n,) — float64
    """
    # n_fft must cover linear deconvolution length to prevent circular aliasing
    n_linear = len(received) + len(reference) - 1
    # Power of 2 for efficient FFT
    n_fft = 1 << (n_linear - 1).bit_length()

    Y = sp_fft.rfft(received, n=n_fft)
    X = sp_fft.rfft(reference, n=n_fft)
    # Y, X: (n_fft//2 + 1,) — complex128

    X_conj = np.conj(X)
    power = np.abs(X) ** 2
    # power: (n_fft//2 + 1,) — float64

    H = (Y * X_conj) / (power + regularization)
    # H: (n_fft//2 + 1,) — complex128

    rir = sp_fft.irfft(H, n=n_fft)
    # rir: (n_fft,) — float64

    # Truncate to linear deconvolution length.
    # Using n_linear preserves all echoes within the captured window,
    # including the last len(reference)-1 samples that were previously
    # discarded (up to ~6 m of range at 44100 Hz with 20 ms chirp).
    rir = rir[:n_linear]
    return rir


def adaptive_wiener(
    received: NDArray[np.float64],
    reference: NDArray[np.float64],
    snr_estimate_db: float = 20.0,
) -> NDArray[np.float64]:
    """Wiener deconvolution with adaptive regularization based on estimated SNR.

    Adjusts epsilon automatically: more regularization when SNR is low,
    less when high.

    Args:
        received: captured signal — shape: (n,)
        reference: emitted chirp — shape: (m,)
        snr_estimate_db: estimated SNR in dB

    Returns:
        Estimated RIR — shape: (n,) — float64
    """
    snr_linear = 10 ** (snr_estimate_db / 10)
    regularization = 1.0 / snr_linear
    return wiener_deconvolution(received, reference, regularization)


def estimate_snr(
    signal: NDArray[np.float64],
    noise_region: tuple[int, int] | None = None,
) -> float:
    """Estimate SNR in dB from the signal.

    Noise floor estimation strategy:
    - If noise_region is given explicitly, uses that.
    - Otherwise, uses the 5th percentile of windowed power as a robust
      estimate of the noise floor. This avoids the bias of the old method
      (last 10% of signal) which overestimates noise in reverberant signals
      where the tail still contains significant late reflections.

    Args:
        signal: complete signal — shape: (n,)
        noise_region: tuple (start, end) indices of a noise-only region

    Returns:
        Estimated SNR in dB
    """
    if noise_region is not None:
        noise = signal[noise_region[0]:noise_region[1]]
        noise_power = float(np.mean(noise ** 2))
    else:
        # Robust noise floor: 5th percentile of windowed power
        window_size = max(int(len(signal) * 0.02), 16)
        n_windows = max(1, len(signal) // window_size)
        windowed_power = np.array([
            np.mean(signal[i * window_size:(i + 1) * window_size] ** 2)
            for i in range(n_windows)
        ])
        noise_power = float(np.percentile(windowed_power, 5))

    signal_power = float(np.mean(signal ** 2))

    if signal_power < 1e-12:
        return -60.0  # Silent signal — SNR effectively −∞
    if noise_power < 1e-12:
        return 60.0  # Negligible noise

    snr = signal_power / noise_power
    return float(10 * np.log10(max(snr, 1e-12)))
