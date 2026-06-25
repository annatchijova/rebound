"""
Training dataset construction.

Pipeline:
1. Generate room configurations (room_generator)
2. Compute RIR for each configuration (pyroomacoustics)
3. Extract spectral features from each RIR
4. Save dataset as .npz for training

For stairs class: uses synthesized stair RIRs from stairs.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.features.spectral import extract_features
from src.signal.chirp import CHIRP_PARAMS
from src.signal.stairs import synthesize_stair_rir
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

    for i, config in enumerate(configs):
        if verbose and i % 100 == 0:
            print(f"  [{i}/{total}] Generating class={config.class_name} "
                  f"dist={config.distance_m:.2f}m")

        if config.class_id == 5:
            # Stairs: use synthesized stair RIR
            n_steps = rng.integers(5, 15)
            tread = rng.uniform(0.27, 0.31)
            rir = synthesize_stair_rir(
                n_steps=n_steps,
                tread_m=tread,
                sample_rate=sr,
                base_distance_m=config.distance_m,
            )
        else:
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


def add_noise_augmentation(
    dataset: dict[str, NDArray],
    snr_range_db: tuple[float, float] = (5.0, 30.0),
    n_augmented: int = 1,
    seed: int = 123,
) -> dict[str, NDArray]:
    """Add noise augmentation to existing dataset.

    For each original sample, generates n_augmented copies with Gaussian
    noise at different SNRs.
    """
    rng = np.random.default_rng(seed)

    n_orig = len(dataset["mel_spectrograms"])

    aug_mels = []
    aug_mfccs = []
    aug_rt60 = []
    aug_centroid = []
    aug_labels = []
    aug_distances = []

    for i in range(n_orig):
        for _ in range(n_augmented):
            snr_db = rng.uniform(*snr_range_db)
            snr_linear = 10 ** (snr_db / 10)

            mel = dataset["mel_spectrograms"][i].copy()
            signal_power = np.mean(mel ** 2)
            noise_power = signal_power / snr_linear
            noise = rng.standard_normal(mel.shape).astype(np.float32) * np.sqrt(noise_power)
            mel_noisy = mel + noise

            aug_mels.append(mel_noisy)
            aug_mfccs.append(dataset["mfccs"][i])
            aug_rt60.append(dataset["rt60"][i])
            aug_centroid.append(dataset["spectral_centroid"][i])
            aug_labels.append(dataset["labels"][i])
            aug_distances.append(dataset["distances"][i])

    return {
        "mel_spectrograms": np.concatenate([
            dataset["mel_spectrograms"],
            np.array(aug_mels),
        ]),
        "mfccs": np.concatenate([
            dataset["mfccs"],
            np.array(aug_mfccs),
        ]),
        "rt60": np.concatenate([
            dataset["rt60"],
            np.array(aug_rt60),
        ]),
        "spectral_centroid": np.concatenate([
            dataset["spectral_centroid"],
            np.array(aug_centroid),
        ]),
        "labels": np.concatenate([
            dataset["labels"],
            np.array(aug_labels),
        ]),
        "distances": np.concatenate([
            dataset["distances"],
            np.array(aug_distances),
        ]),
    }


if __name__ == "__main__":
    print("Building REBOUND dataset...")
    print("=" * 50)
    dataset = build_dataset(n_per_class=500, verbose=True)
    print("\nApplying noise augmentation...")
    dataset_aug = add_noise_augmentation(dataset, n_augmented=1)
    np.savez_compressed("data/processed/dataset_augmented.npz", **dataset_aug)
    print(f"Augmented dataset: {len(dataset_aug['labels'])} total samples")
