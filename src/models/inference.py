"""
Inference pipeline for the trained ReboundCNN model.

Loads checkpoint (model weights + scaler statistics) and provides
a single function to classify a RIR with properly normalized inputs.

Usage:
    from src.models.inference import load_model, predict

    model, scaler, device = load_model("models/checkpoints/best_model.pt")
    result = predict(model, scaler, rir, sample_rate=44100, device=device)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from src.features.spectral import extract_features
from src.models.classifier import ReboundCNN
from src.signal.stairs import detect_stair_periodicity
from src.simulation.room_generator import SPACE_CLASSES


def load_model(
    checkpoint_path: str,
    n_classes: int = 5,   # era 6
    device: str | None = None,
) -> tuple[ReboundCNN, dict[str, float], str]:
    """Load trained model and scaler from checkpoint.

    Args:
        checkpoint_path: path to .pt checkpoint file
        n_classes: number of output classes
        device: "cuda", "cpu", or None (auto-detect)

    Returns:
        (model, scaler, device)
        scaler keys: rt60_mean, rt60_std, centroid_mean, centroid_std
    """
    if device is None:
        device = "cpu"
        try:
            if torch.cuda.is_available():
                torch.zeros(1, device="cuda")
                device = "cuda"
        except Exception:
            pass

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = ReboundCNN(n_classes=n_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    scaler = checkpoint.get("scaler", {
        "rt60_mean": 0.0, "rt60_std": 1.0,
        "centroid_mean": 0.0, "centroid_std": 1.0,
    })

    return model, scaler, device


def predict(
    model: ReboundCNN,
    scaler: dict[str, float],
    rir: NDArray[np.float64],
    sample_rate: int = 44_100,
    device: str = "cpu",
    n_mels: int = 64,
    n_mfcc: int = 13,
    target_frames: int = 32,
) -> dict[str, str | float | int]:
    """Run inference on a single RIR.

    Args:
        model: loaded ReboundCNN
        scaler: normalization statistics from training
        rir: Room Impulse Response — shape: (n,)
        sample_rate: sampling rate in Hz
        device: compute device
        n_mels: mel bands (must match training)
        n_mfcc: MFCC coefficients (must match training)
        target_frames: temporal frames (must match training)

    Returns:
        {
            "class_name": str,
            "class_id": int,
            "confidence": float,
            "distance_m": float,
            "probabilities": dict[str, float],
        }
    """
    features = extract_features(
        rir, sample_rate=sample_rate,
        n_mels=n_mels, n_mfcc=n_mfcc, target_frames=target_frames,
    )

    mel = torch.from_numpy(features["mel_spectrogram"]).unsqueeze(0).unsqueeze(0)
    # mel: (1, 1, n_mels, target_frames)

    rt60_norm = (features["rt60"] - scaler["rt60_mean"]) / scaler["rt60_std"]
    centroid_norm = (features["spectral_centroid"] - scaler["centroid_mean"]) / scaler["centroid_std"]
    scalars = torch.tensor([[rt60_norm, centroid_norm]], dtype=torch.float32)
    # scalars: (1, 2)

    mel = mel.to(device)
    scalars = scalars.to(device)

    with torch.no_grad():
        class_logits, dist_pred = model(mel, scalars)

    probs = torch.softmax(class_logits, dim=1).squeeze(0).cpu().numpy()
    class_id = int(probs.argmax())
    confidence = float(probs[class_id])
    distance_m = float(dist_pred.squeeze().cpu())

    # El detector corre siempre. El CNN no tiene clase "stairs",
    # entonces el detector es la única vía de detección de escaleras.
    stairs = detect_stair_periodicity(rir, sample_rate=sample_rate)

    # Si el detector confirma escalera con confianza alta, override.
    if stairs["is_stair"] and stairs["confidence"] >= 0.7:
        final_class_name = "stairs"
        final_confidence = float(stairs["confidence"])
        override_applied = True
    else:
        final_class_name = SPACE_CLASSES[class_id]
        final_confidence = confidence
        override_applied = False

    return {
        "class_name": final_class_name,
        "class_id": class_id,                          # del CNN, no override
        "confidence": final_confidence,
        "cnn_class_name": SPACE_CLASSES[class_id],     # lo que dijo el CNN
        "cnn_confidence": confidence,                  # confianza del CNN
        "override_applied": override_applied,
        "distance_m": distance_m,
        "probabilities": {
            SPACE_CLASSES[i]: float(round(probs[i], 4))
            for i in range(len(probs))
        },
        "stairs_detection": stairs,
    }
