"""
Real-time audio capture with sounddevice.

In the hackathon demo, capture can be real (microphone) or simulated
(pyroomacoustics RIR convolved with chirp).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve

from src.signal.chirp import CHIRP_PARAMS, generate_chirp


def simulate_capture(
    rir: NDArray[np.float64],
    chirp: NDArray[np.float64] | None = None,
    noise_level: float = 0.01,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Simulate echo capture by convolving chirp with RIR.

    Args:
        rir: Room Impulse Response — shape: (n,)
        chirp: emitted chirp (None = generate default)
        noise_level: Gaussian noise level
        seed: seed for reproducibility

    Returns:
        Simulated captured signal — shape: (n_chirp + n_rir - 1,)
    """
    if chirp is None:
        chirp = generate_chirp()

    # Convolution = linear propagation model
    captured = fftconvolve(chirp, rir, mode="full")
    # captured: (n_chirp + n_rir - 1,) — float64

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(captured)) * noise_level
    captured += noise

    return captured


def capture_realtime(
    duration_s: float = 0.15,
    sample_rate: int | None = None,
    channels: int = 1,
) -> NDArray[np.float64]:
    """Capture audio in real-time from microphone.

    Args:
        duration_s: capture duration in seconds
        sample_rate: sampling rate
        channels: number of channels (1=mono)

    Returns:
        Captured audio — shape: (n_samples,) if mono
    """
    import sounddevice as sd

    sr = sample_rate or CHIRP_PARAMS["sample_rate"]
    n_samples = int(duration_s * sr)

    recording = sd.rec(
        n_samples,
        samplerate=sr,
        channels=channels,
        dtype="float64",
        blocking=True,
    )

    if channels == 1:
        return recording.flatten()
    return recording


def emit_and_capture(
    chirp: NDArray[np.float64] | None = None,
    capture_duration_s: float = 0.15,
    sample_rate: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Emit chirp through speaker and capture echo via microphone.

    NOTE: Requires hardware with speaker and microphone.
    In the demo, use simulate_capture() instead.

    Returns:
        Tuple (emitted_chirp, captured_echo)
    """
    import sounddevice as sd

    sr = sample_rate or CHIRP_PARAMS["sample_rate"]
    if chirp is None:
        chirp = generate_chirp()

    n_capture = int(capture_duration_s * sr)
    padded = np.zeros(n_capture)
    padded[:len(chirp)] = chirp

    # Simultaneous play and record
    captured = sd.playrec(
        padded.reshape(-1, 1),
        samplerate=sr,
        channels=1,
        dtype="float64",
        blocking=True,
    )

    return chirp, captured.flatten()
