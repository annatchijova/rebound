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


def calibrate_hardware_latency(
    sample_rate: int | None = None,
    n_trials: int = 5,
) -> int:
    """Measure hardware round-trip latency in samples.

    Emits a short impulse and measures the delay before it appears
    in the recording. Averages over n_trials for stability.

    Args:
        sample_rate: sampling rate
        n_trials: number of calibration trials

    Returns:
        Estimated latency in samples
    """
    import sounddevice as sd

    sr = sample_rate or CHIRP_PARAMS["sample_rate"]
    duration_samples = int(0.05 * sr)  # 50 ms calibration signal

    # Short click impulse
    impulse = np.zeros(duration_samples)
    impulse[10] = 0.8

    latencies = []
    for _ in range(n_trials):
        captured = sd.playrec(
            impulse.reshape(-1, 1),
            samplerate=sr,
            channels=1,
            dtype="float64",
            blocking=True,
        ).flatten()

        # Find first significant energy in captured signal
        threshold = 0.1 * np.max(np.abs(captured))
        above = np.where(np.abs(captured) > threshold)[0]
        if len(above) > 0:
            latencies.append(int(above[0]))

    if not latencies:
        return 0
    return int(np.median(latencies))


def emit_and_capture(
    chirp: NDArray[np.float64] | None = None,
    capture_duration_s: float = 0.15,
    sample_rate: int | None = None,
    hardware_latency_samples: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Emit chirp through speaker and capture echo via microphone.

    NOTE: Requires hardware with speaker and microphone.
    In the demo, use simulate_capture() instead.

    Args:
        chirp: chirp signal (None = generate default)
        capture_duration_s: capture duration in seconds
        sample_rate: sampling rate
        hardware_latency_samples: round-trip hardware latency to compensate
            (measured via calibrate_hardware_latency). The captured signal
            is shifted left by this amount so the RIR peak reflects true
            acoustic distance rather than distance + hardware delay.

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
    ).flatten()

    # Compensate hardware latency
    if hardware_latency_samples > 0 and hardware_latency_samples < len(captured):
        captured = captured[hardware_latency_samples:]

    return chirp, captured
