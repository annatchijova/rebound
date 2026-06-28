"""
Multi-output CNN for space classification + distance regression.

Architecture designed for:
- Input: mel spectrogram (1, n_mels, n_frames) + scalar features (rt60, centroid)
- Output 1: space class (6 classes, softmax)
- Output 2: distance to obstacle (regression, meters)

Both outputs in a single forward pass for efficiency.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReboundCNN(nn.Module):
    """Multi-output CNN for acoustic space classification.

    Input:
        mel: (batch, 1, n_mels, n_frames) — mel spectrogram
        scalars: (batch, 2) — [rt60, spectral_centroid_normalized]

    Output:
        class_logits: (batch, 6) — logits per class
        distance: (batch, 1) — estimated distance in meters
    """

    def __init__(
        self,
        n_mels: int = 64,
        n_frames: int = 32,
        n_classes: int = 5,   # era 6
    ):
        super().__init__()

        self.conv_block = nn.Sequential(
            # Block 1: (1, 64, 32) -> (16, 32, 16)
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2: (16, 32, 16) -> (32, 16, 8)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3: (32, 16, 8) -> (64, 8, 4)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Flattened feature map size
        conv_out_h = n_mels // 8
        conv_out_w = n_frames // 8
        conv_flat = 64 * conv_out_h * conv_out_w

        # Shared trunk: conv features + scalar features
        self.trunk = nn.Sequential(
            nn.Linear(conv_flat + 2, 128),  # +2 for rt60 and centroid
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Classification head
        self.class_head = nn.Linear(64, n_classes)

        # Distance regression head
        self.distance_head = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus(),  # smooth positive constraint, gradient never zero
        )

    def forward(
        self,
        mel: torch.Tensor,
        scalars: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            mel: (batch, 1, n_mels, n_frames) — float32
            scalars: (batch, 2) — [rt60, spectral_centroid_normalized]

        Returns:
            class_logits: (batch, 6)
            distance: (batch, 1)
        """
        # mel: (batch, 1, n_mels, n_frames)
        x = self.conv_block(mel)
        # x: (batch, 64, n_mels//8, n_frames//8)

        x = x.flatten(1)
        # x: (batch, conv_flat)

        x = torch.cat([x, scalars], dim=1)
        # x: (batch, conv_flat + 2)

        features = self.trunk(x)
        # features: (batch, 64)

        class_logits = self.class_head(features)
        # class_logits: (batch, 6)

        distance = self.distance_head(features)
        # distance: (batch, 1)

        return class_logits, distance


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
