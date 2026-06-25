"""
Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting.

When the model adapts to a specific user, EWC protects important parameters
of the base model by penalizing large changes to weights that are critical
for previously learned tasks.

Reference: Kirkpatrick et al. (2017) — "Overcoming catastrophic forgetting
in neural networks" — PNAS
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class EWC:
    """Elastic Weight Consolidation.

    Computes the diagonal Fisher Information Matrix over the base model
    dataset and uses it as a penalty during per-user fine-tuning.

    Usage:
        # After training the base model
        ewc = EWC(model, base_dataloader, device)

        # During per-user fine-tuning
        loss = task_loss + ewc.penalty(model)
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: str = "cpu",
        lambda_ewc: float = 1000.0,
    ):
        """
        Args:
            model: already-trained base model
            dataloader: base dataset dataloader
            device: compute device
            lambda_ewc: EWC penalty weight. Typical values: 100-10000.
                Higher = more conservative (less adaptation, less forgetting).
        """
        self.lambda_ewc = lambda_ewc
        self.device = device

        # Save copy of base model parameters
        self.base_params: dict[str, torch.Tensor] = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        # Compute diagonal Fisher Information Matrix
        self.fisher: dict[str, torch.Tensor] = self._compute_fisher(
            model, dataloader
        )

    def _compute_fisher(
        self,
        model: nn.Module,
        dataloader: DataLoader,
    ) -> dict[str, torch.Tensor]:
        """Compute the diagonal of the Fisher Information Matrix.

        The Fisher measures how much each parameter contributes to the
        model output. Parameters with high Fisher are "important" and
        should be preserved.
        """
        fisher: dict[str, torch.Tensor] = {
            name: torch.zeros_like(param)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        model.eval()
        n_samples = 0

        for batch in dataloader:
            mel, scalars, labels, _ = batch
            mel = mel.to(self.device)
            scalars = scalars.to(self.device)
            labels = labels.to(self.device)

            model.zero_grad()
            class_logits, _ = model(mel, scalars)
            loss = nn.functional.cross_entropy(class_logits, labels)
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.detach() ** 2

            n_samples += 1

        if n_samples > 0:
            for name in fisher:
                fisher[name] /= n_samples

        return fisher

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Compute the EWC penalty.

        penalty = (lambda/2) * sum_i F_i * (theta_i - theta_base_i)^2

        where F_i is the Fisher for parameter i, theta_i the current value,
        and theta_base_i the base model value.

        Args:
            model: current model (being fine-tuned)

        Returns:
            Scalar penalty
        """
        loss = torch.tensor(0.0, device=self.device)

        for name, param in model.named_parameters():
            if param.requires_grad and name in self.fisher:
                fisher_val = self.fisher[name].to(self.device)
                base_val = self.base_params[name].to(self.device)
                loss += (fisher_val * (param - base_val) ** 2).sum()

        return (self.lambda_ewc / 2) * loss
