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

    rir = rir[: len(received)]
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

    If no noise region is specified, uses the last 10% of samples
    as the noise floor estimate.

    Args:
        signal: complete signal — shape: (n,)
        noise_region: tuple (start, end) indices of a noise-only region

    Returns:
        Estimated SNR in dB
    """
    if noise_region is not None:
        noise = signal[noise_region[0]:noise_region[1]]
    else:
        n_noise = max(int(len(signal) * 0.1), 1)
        noise = signal[-n_noise:]

    noise_power = np.mean(noise ** 2)
    signal_power = np.mean(signal ** 2)

    if signal_power < 1e-12:
        return -60.0  # Silent signal — SNR effectively −∞
    if noise_power < 1e-12:
        return 60.0  # Negligible noise

    snr = signal_power / noise_power
    return float(10 * np.log10(max(snr, 1e-12)))
