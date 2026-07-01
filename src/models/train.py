"""
CNN classifier training with CUDA support.

Usage:
    python3 -m src.models.train --data data/processed/dataset.npz --epochs 50

Strategy:
- Combined loss: CrossEntropy (classification) + MSE (distance)
- Adjustable loss weights via alpha
- Early stopping by validation loss
- Saves best checkpoint by validation loss
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from src.models.classifier import ReboundCNN, count_parameters


class ReboundDataset(Dataset):
    """Dataset of mel spectrograms + scalar features."""

    def __init__(self, data_path: str):
        data = np.load(data_path)
        self.mel = torch.from_numpy(data["mel_spectrograms"]).unsqueeze(1)
        # mel: (N, 1, n_mels, n_frames) — float32

        rt60 = data["rt60"].astype(np.float32)
        centroid = data["spectral_centroid"].astype(np.float32)

        # Standardize scalars to mean=0, std=1
        self.rt60_mean, self.rt60_std = float(rt60.mean()), float(rt60.std() + 1e-8)
        self.centroid_mean, self.centroid_std = float(centroid.mean()), float(centroid.std() + 1e-8)

        rt60_norm = (rt60 - self.rt60_mean) / self.rt60_std
        centroid_norm = (centroid - self.centroid_mean) / self.centroid_std

        self.scalars = torch.from_numpy(
            np.stack([rt60_norm, centroid_norm], axis=1)
        )
        # scalars: (N, 2) — float32, standardized

        self.labels = torch.from_numpy(data["labels"]).long()
        # labels: (N,) — int64

        self.distances = torch.from_numpy(data["distances"]).float().unsqueeze(1)
        # distances: (N, 1) — float32

        self.config_ids = torch.from_numpy(
            data["config_ids"] if "config_ids" in data.files
            else np.arange(len(data["labels"]), dtype=np.int64)
        ).long()
        # config_ids: (N,) — int64

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.mel[idx], self.scalars[idx], self.labels[idx], self.distances[idx]


def train(
    data_path: str,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    alpha: float = 0.5,
    val_split: float = 0.2,
    checkpoint_dir: str = "models/checkpoints",
    device: str | None = None,
) -> dict:
    """Train the ReboundCNN model.

    Args:
        data_path: path to dataset .npz
        epochs: number of epochs
        batch_size: batch size
        lr: learning rate
        alpha: distance loss weight (0=class only, 1=distance only)
        val_split: validation fraction
        checkpoint_dir: directory for saving checkpoints
        device: "cuda", "cpu", or None (auto-detect)

    Returns:
        Dictionary with training metrics
    """
    if device is None:
        device = "cpu"
        try:
            if torch.cuda.is_available():
                torch.zeros(1, device="cuda")
                device = "cuda"
        except Exception:
            pass
    print(f"Device: {device}")

    full_dataset = ReboundDataset(data_path)

    # Stratified split: ensures each class is proportionally represented
    labels_np = full_dataset.labels.numpy()
    all_indices = np.arange(len(labels_np))

    rng_split = np.random.default_rng(42)
    train_indices = []
    val_indices = []

    for class_id in np.unique(labels_np):
        class_mask = labels_np == class_id
        class_indices = all_indices[class_mask].tolist()
        rng_split.shuffle(class_indices)
        n_val_class = max(1, int(len(class_indices) * val_split))
        val_indices.extend(class_indices[:n_val_class])
        train_indices.extend(class_indices[n_val_class:])

    # Recalculate normalization stats using ONLY training data (fixes data leakage)
    train_rt60 = full_dataset.scalars[train_indices, 0]
    train_centroid = full_dataset.scalars[train_indices, 1]
    # The dataset was already normalized with full stats; we need raw values.
    # Reverse the initial normalization, then re-normalize with train-only stats.
    raw_rt60 = train_rt60 * full_dataset.rt60_std + full_dataset.rt60_mean
    raw_centroid = train_centroid * full_dataset.centroid_std + full_dataset.centroid_mean

    train_rt60_mean = float(raw_rt60.mean())
    train_rt60_std = float(raw_rt60.std() + 1e-8)
    train_centroid_mean = float(raw_centroid.mean())
    train_centroid_std = float(raw_centroid.std() + 1e-8)

    # Re-normalize ALL scalars using train-only statistics
    all_raw_rt60 = full_dataset.scalars[:, 0] * full_dataset.rt60_std + full_dataset.rt60_mean
    all_raw_centroid = full_dataset.scalars[:, 1] * full_dataset.centroid_std + full_dataset.centroid_mean
    full_dataset.scalars[:, 0] = (all_raw_rt60 - train_rt60_mean) / train_rt60_std
    full_dataset.scalars[:, 1] = (all_raw_centroid - train_centroid_mean) / train_centroid_std
    full_dataset.rt60_mean = train_rt60_mean
    full_dataset.rt60_std = train_rt60_std
    full_dataset.centroid_mean = train_centroid_mean
    full_dataset.centroid_std = train_centroid_std

    train_ds = Subset(full_dataset, train_indices)
    val_ds = Subset(full_dataset, val_indices)
    n_train = len(train_indices)
    n_val = len(val_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    print(f"Train: {n_train}, Val: {n_val}")

    model = ReboundCNN(n_classes=6).to(device)
    print(f"Parameters: {count_parameters(model):,}")

    class_criterion = nn.CrossEntropyLoss()
    dist_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    scaler_stats = {
        "rt60_mean": full_dataset.rt60_mean,
        "rt60_std": full_dataset.rt60_std,
        "centroid_mean": full_dataset.centroid_mean,
        "centroid_std": full_dataset.centroid_std,
    }
    history: dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_dist_mae": [],
    }

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for mel, scalars, labels, distances in train_loader:
            mel = mel.to(device)
            scalars = scalars.to(device)
            labels = labels.to(device)
            distances = distances.to(device)

            optimizer.zero_grad()
            class_logits, dist_pred = model(mel, scalars)

            loss_class = class_criterion(class_logits, labels)
            loss_dist = dist_criterion(dist_pred, distances)
            loss = (1 - alpha) * loss_class + alpha * loss_dist

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * mel.size(0)

        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        correct = 0
        dist_errors = []

        with torch.no_grad():
            for mel, scalars, labels, distances in val_loader:
                mel = mel.to(device)
                scalars = scalars.to(device)
                labels = labels.to(device)
                distances = distances.to(device)

                class_logits, dist_pred = model(mel, scalars)

                loss_class = class_criterion(class_logits, labels)
                loss_dist = dist_criterion(dist_pred, distances)
                loss = (1 - alpha) * loss_class + alpha * loss_dist
                val_loss += loss.item() * mel.size(0)

                preds = class_logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                dist_errors.append(
                    (dist_pred - distances).abs().cpu().numpy()
                )

        val_loss /= n_val
        val_acc = correct / n_val
        val_mae = float(np.mean(np.concatenate(dist_errors)))

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_dist_mae"].append(val_mae)

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.3f} | "
                f"Val MAE dist: {val_mae:.3f}m"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "scaler": scaler_stats,
            }, ckpt_path / "best_model.pt")

    torch.save({
        "epoch": epochs - 1,
        "model_state_dict": model.state_dict(),
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "scaler": scaler_stats,
    }, ckpt_path / "last_model.pt")

    with open(ckpt_path / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val_loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {ckpt_path}/")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ReboundCNN")
    parser.add_argument("--data", default="data/processed/dataset.npz")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    train(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha=args.alpha,
    )
