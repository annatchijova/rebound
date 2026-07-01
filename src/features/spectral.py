"""
Spectral feature extraction from Room Impulse Response.

Features extracted:
- Mel spectrogram: perceptual time-frequency representation
- MFCCs: cepstral coefficients for spectral information compression
- RT60: reverberation time — indicator of room size

Reference: Dokmanic et al. (2013) — "Acoustic echoes reveal room shape"
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.signal.chirp import CHIRP_PARAMS


def compute_mel_spectrogram(
    signal: NDArray[np.float64],
    sample_rate: int | None = None,
    n_fft: int = 512,
    hop_length: int = 128,
    n_mels: int = 64,
    fmin: float = 500.0,
    fmax: float | None = 9_000,  # era None → sr/2 = 22050
) -> NDArray[np.float32]:
    """Compute mel spectrogram of the signal.

    Args:
        signal: audio signal — shape: (n_samples,)
        sample_rate: sampling rate
        n_fft: FFT window size
        hop_length: hop between windows
        n_mels: number of mel bands
        fmin: minimum frequency
        fmax: maximum frequency (default: sample_rate/2)

    Returns:
        Mel spectrogram in dB — shape: (n_mels, n_frames) — float32
    """
    import librosa

    sr = sample_rate or CHIRP_PARAMS["sample_rate"]
    fmax = fmax or sr / 2

    S = librosa.feature.melspectrogram(
        y=signal.astype(np.float32),
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    # S: (n_mels, n_frames) — float32

    S_db = librosa.power_to_db(S, ref=np.max)
    # S_db: (n_mels, n_frames) — float32
    return S_db


def compute_mfccs(
    signal: NDArray[np.float64],
    sample_rate: int | None = None,
    n_mfcc: int = 13,
    n_fft: int = 512,
    hop_length: int = 128,
    n_mels: int = 64,
) -> NDArray[np.float32]:
    """Compute MFCCs (Mel-Frequency Cepstral Coefficients).

    Args:
        signal: audio signal — shape: (n_samples,)
        sample_rate: sampling rate
        n_mfcc: number of coefficients
        n_fft: FFT window size
        hop_length: hop between windows
        n_mels: number of mel bands

    Returns:
        MFCCs — shape: (n_mfcc, n_frames) — float32
    """
    import librosa

    sr = sample_rate or CHIRP_PARAMS["sample_rate"]

    mfccs = librosa.feature.mfcc(
        y=signal.astype(np.float32),
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    # mfccs: (n_mfcc, n_frames) — float32
    return mfccs


def compute_rt60(
    rir: NDArray[np.float64],
    sample_rate: int | None = None,
) -> float:
    """Estimate RT60 (reverberation time) from the RIR.

    Method: Schroeder backward integration.
    RT60 is the time for energy to decay 60 dB from peak.
    If the decay doesn't reach 60 dB, extrapolates from T20 or T30.

    Args:
        rir: Room Impulse Response — shape: (n,)
        sample_rate: sampling rate

    Returns:
        Estimated RT60 in seconds
    """
    sr = sample_rate or CHIRP_PARAMS["sample_rate"]

    # Schroeder backward integration
    energy = rir ** 2
    # energy: (n,) — float64

    schroeder = np.cumsum(energy[::-1])[::-1]
    schroeder_db = 10 * np.log10(schroeder / (schroeder[0] + 1e-12) + 1e-12)
    # schroeder_db: (n,) — float64

    # Ensure monotonicity for searchsorted
    neg_schroeder = -schroeder_db
    if not np.all(np.diff(neg_schroeder) >= 0):
        # Force monotonicity via cumulative maximum
        neg_schroeder = np.maximum.accumulate(neg_schroeder)

    # Find T30 (decay from -5 dB to -35 dB) and extrapolate to T60
    idx_5 = np.searchsorted(neg_schroeder, 5)
    idx_35 = np.searchsorted(neg_schroeder, 35)

    if idx_35 >= len(schroeder_db) or idx_35 <= idx_5:
        # Not enough decay — try T20 (-5 to -25 dB)
        idx_25 = np.searchsorted(neg_schroeder, 25)
        if idx_25 >= len(schroeder_db) or idx_25 <= idx_5:
            # Very dry room or very short signal
            return 0.0
        t20 = (idx_25 - idx_5) / sr
        return float(t20 * 3.0)  # Extrapolate T20 → T60

    t30 = (idx_35 - idx_5) / sr
    return float(t30 * 2.0)  # Extrapolate T30 → T60


def compute_spectral_centroid(
    signal: NDArray[np.float64],
    sample_rate: int | None = None,
    n_fft: int = 512,
) -> float:
    """Compute spectral centroid (energy-weighted mean frequency).

    Useful as a quick indicator of dominant frequency content.

    Args:
        signal: audio signal — shape: (n_samples,)
        sample_rate: sampling rate
        n_fft: FFT window size

    Returns:
        Spectral centroid in Hz
    """
    sr = sample_rate or CHIRP_PARAMS["sample_rate"]
    spectrum = np.abs(np.fft.rfft(signal, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    # spectrum, freqs: (n_fft//2 + 1,)

    total_energy = np.sum(spectrum)
    if total_energy < 1e-12:
        return 0.0
    return float(np.sum(freqs * spectrum) / total_energy)


def extract_features(
    rir: NDArray[np.float64],
    sample_rate: int | None = None,
    n_mels: int = 64,
    n_mfcc: int = 13,
    target_frames: int = 32,
) -> dict[str, NDArray[np.float32] | float]:
    """Extract all features from a RIR.

    Returns:
        Dictionary with:
        - "mel_spectrogram": (n_mels, target_frames) — float32
        - "mfccs": (n_mfcc, target_frames) — float32
        - "rt60": float — seconds
        - "spectral_centroid": float — Hz
    """
    sr = sample_rate or CHIRP_PARAMS["sample_rate"]

    mel = compute_mel_spectrogram(rir, sr, n_mels=n_mels)
    mfccs = compute_mfccs(rir, sr, n_mfcc=n_mfcc)
    rt60 = compute_rt60(rir, sr)
    centroid = compute_spectral_centroid(rir, sr)

    # Normalize frames to target_frames via interpolation or truncation
    mel = _resize_frames(mel, target_frames)
    mfccs = _resize_frames(mfccs, target_frames)

    return {
        "mel_spectrogram": mel,
        "mfccs": mfccs,
        "rt60": rt60,
        "spectral_centroid": centroid,
    }


def _resize_frames(
    matrix: NDArray[np.float32],
    target_frames: int,
) -> NDArray[np.float32]:
    """Resize temporal axis to target_frames.

    Args:
        matrix: shape (features, n_frames)
        target_frames: desired number of frames

    Returns:
        shape (features, target_frames) — float32
    """
    from scipy.ndimage import zoom

    n_feat, n_frames = matrix.shape
    if n_frames == target_frames:
        return matrix
    if n_frames == 0:
        return np.zeros((n_feat, target_frames), dtype=np.float32)
    factor = target_frames / n_frames
    resized = zoom(matrix, (1.0, factor), order=1)
    # Ensure exact dimension (zoom may give +/-1)
    return resized[:, :target_frames].astype(np.float32)
