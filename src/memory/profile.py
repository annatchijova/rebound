"""
User acoustic-kinetic profile.

Stores Bayesian priors, interaction history, and personalized
configuration. Persisted as JSON between sessions.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.simulation.room_generator import SPACE_CLASSES

logger = logging.getLogger(__name__)

N_CLASSES = len(SPACE_CLASSES)

_VALID_USER_ID = re.compile(r'[a-zA-Z0-9_\-]{1,64}')

# Valid user actions
USER_ACTIONS = ("advance", "hesitate", "retreat", "ignore")


@dataclass
class UserProfile:
    """Adaptive user profile for the navigation system."""

    user_id: str
    # Bayesian priors per class — updated with each interaction
    class_weights: list[float] = field(
        default_factory=lambda: [1.0] * N_CLASSES
    )
    # Interaction counts per class
    class_counts: list[int] = field(
        default_factory=lambda: [0] * N_CLASSES
    )
    # Correct/incorrect counts per class
    class_correct: list[int] = field(
        default_factory=lambda: [0] * N_CLASSES
    )
    total_sessions: int = 0
    total_interactions: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def update_implicit(
        self,
        prediction: int,
        user_action: str,
    ) -> None:
        """Update from implicit user signal.

        Args:
            prediction: predicted class (0-5)
            user_action: "advance", "hesitate", "retreat", "ignore"
        """
        if user_action not in USER_ACTIONS:
            raise ValueError(f"Invalid user action: {user_action!r}. Must be one of {USER_ACTIONS}")

        self.total_interactions += 1
        self.class_counts[prediction] += 1
        self.updated_at = time.time()

        adjustments = {
            "advance": 1.05,     # confirms prediction
            "hesitate": 0.98,    # mild uncertainty
            "retreat": 0.90,     # prediction likely incorrect
            "ignore": 1.0,       # neutral
        }

        factor = adjustments[user_action]
        self.class_weights[prediction] *= factor

        # Normalize so mean is 1.0
        mean_w = sum(self.class_weights) / N_CLASSES
        if mean_w > 0:
            self.class_weights = [w / mean_w for w in self.class_weights]

    def update_explicit(
        self,
        prediction: int,
        correct: bool,
    ) -> None:
        """Update from explicit user signal.

        Args:
            prediction: predicted class
            correct: True if user confirms the prediction
        """
        self.total_interactions += 1
        self.class_counts[prediction] += 1
        self.updated_at = time.time()

        if correct:
            self.class_correct[prediction] += 1
            self.class_weights[prediction] *= 1.1
        else:
            self.class_weights[prediction] *= 0.85

        mean_w = sum(self.class_weights) / N_CLASSES
        if mean_w > 0:
            self.class_weights = [w / mean_w for w in self.class_weights]

    def get_prior_weights(self) -> NDArray[np.float64]:
        """Return confidence weights per class for this user.

        Applied as Bayesian prior over softmax probabilities:
        p_adjusted[i] = p_model[i] * weight[i] / sum(p_model * weights)

        Returns:
            Weights — shape: (n_classes,)
        """
        return np.array(self.class_weights, dtype=np.float64)

    def get_accuracy_by_class(self) -> dict[str, float]:
        """Return accuracy per class based on explicit feedback."""
        result = {}
        for i, name in SPACE_CLASSES.items():
            if self.class_counts[i] > 0:
                result[name] = self.class_correct[i] / self.class_counts[i]
            else:
                result[name] = 0.0
        return result

    def start_session(self) -> None:
        """Register start of a new session."""
        self.total_sessions += 1
        self.updated_at = time.time()

    @staticmethod
    def _validate_user_id(user_id: str) -> None:
        """Validate user_id to prevent path traversal."""
        if not _VALID_USER_ID.fullmatch(user_id):
            raise ValueError(
                f"Invalid user_id: {user_id!r}. "
                "Must be 1-64 alphanumeric characters, hyphens, or underscores."
            )

    def save(self, directory: str = "data/profiles") -> Path:
        """Persist profile to JSON."""
        self._validate_user_id(self.user_id)
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        filepath = path / f"{self.user_id}.json"
        with open(filepath, "w") as f:
            json.dump(asdict(self), f, indent=2)
        return filepath

    @classmethod
    def load(cls, user_id: str, directory: str = "data/profiles") -> UserProfile:
        """Load existing profile or create a new one."""
        cls._validate_user_id(user_id)
        filepath = Path(directory) / f"{user_id}.json"
        if filepath.exists():
            try:
                with open(filepath) as f:
                    data = json.load(f)
                return cls(**data)
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                logger.error("Corrupt profile %s: %s. Creating new profile.", user_id, e)
                return cls(user_id=user_id)
        return cls(user_id=user_id)

    def to_summary(self) -> dict:
        """Compact summary for sending to Qwen."""
        return {
            "user_id": self.user_id,
            "total_sessions": self.total_sessions,
            "total_interactions": self.total_interactions,
            "class_weights": {
                SPACE_CLASSES[i]: round(w, 3)
                for i, w in enumerate(self.class_weights)
            },
            "accuracy_by_class": self.get_accuracy_by_class(),
        }
