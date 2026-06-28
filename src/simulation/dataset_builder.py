"""
Training dataset construction.

Pipeline:
1. Generate room configurations (room_generator)
2. Compute RIR for each configuration (pyroomacoustics)
3. Extract spectral features from each RIR
4. Save dataset as .npz for training
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.features.spectral import extract_features
from src.signal.chirp import CHIRP_PARAMS
from src.simulation.room_generator import (
    SPACE_CLASSES,
    RoomConfig,
    generate_dataset_configs,
    generate_rir,
)


def build_dataset(
    n_per_class: int = 500,
    sample_rate: int | None = None,
    n_mels: int = 64,
    n_mfcc: int = 13,
    target_frames: int = 32,
    output_dir: str = "data/processed",
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, NDArray]:
    """Build the complete dataset: generate RIRs and extract features.

    Args:
        n_per_class: RIRs per class
        sample_rate: sampling rate
        n_mels: mel bands
        n_mfcc: MFCC coefficients
        target_frames: normalized temporal frames
        output_dir: output directory
        seed: seed
        verbose: print progress

    Returns:
        Dictionary with dataset arrays
    """
    sr = sample_rate or CHIRP_PARAMS["sample_rate"]
    rng = np.random.default_rng(seed + 100)

    configs = generate_dataset_configs(n_per_class=n_per_class, seed=seed)
    total = len(configs)

    mel_specs = np.zeros((total, n_mels, target_frames), dtype=np.float32)
    mfccs_all = np.zeros((total, n_mfcc, target_frames), dtype=np.float32)
    rt60s = np.zeros(total, dtype=np.float32)
    centroids = np.zeros(total, dtype=np.float32)
    labels = np.zeros(total, dtype=np.int64)
    distances = np.zeros(total, dtype=np.float32)
    config_ids = np.arange(total, dtype=np.int64)

    for i, config in enumerate(configs):
        if verbose and i % 100 == 0:
            print(f"  [{i}/{total}] Generating class={config.class_name} "
                  f"dist={config.distance_m:.2f}m")

        rir = generate_rir(config, sample_rate=sr)

        features = extract_features(
            rir,
            sample_rate=sr,
            n_mels=n_mels,
            n_mfcc=n_mfcc,
            target_frames=target_frames,
        )

        mel_specs[i] = features["mel_spectrogram"]
        mfccs_all[i] = features["mfccs"]
        rt60s[i] = features["rt60"]
        centroids[i] = features["spectral_centroid"]
        labels[i] = config.class_id
        distances[i] = config.distance_m

    dataset = {
        "mel_spectrograms": mel_specs,
        "mfccs": mfccs_all,
        "rt60": rt60s,
        "spectral_centroid": centroids,
        "labels": labels,
        "distances": distances,
        "config_ids": config_ids,
    }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(out_path / "dataset.npz", **dataset)

    meta = {
        "n_per_class": n_per_class,
        "total_samples": total,
        "classes": SPACE_CLASSES,
        "sample_rate": sr,
        "n_mels": n_mels,
        "n_mfcc": n_mfcc,
        "target_frames": target_frames,
        "seed": seed,
    }
    with open(out_path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print(f"\nDataset saved to {out_path}/")
        print(f"  Total: {total} samples")
        for cid, cname in SPACE_CLASSES.items():
            count = np.sum(labels == cid)
            print(f"  {cname}: {count}")

    return dataset


if __name__ == "__main__":
    print("Building REBOUND dataset...")
    print("=" * 50)
    dataset = build_dataset(n_per_class=500, verbose=True)
    print("\nNoise augmentation diferida a v2 (ver future/augmentation.py).")
