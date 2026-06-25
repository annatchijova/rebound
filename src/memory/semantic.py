"""
Consolidated semantic memory.

Stores generalized knowledge about the user extracted from patterns
in episodic memory. Unlike episodic memory, semantic memory does not
decay — it is updated and consolidated.

Cognitive analogy: episodic remembers "yesterday I hesitated at a door".
Semantic knows "this user needs extra confidence at doorways".
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from src.memory.episodic import EpisodicMemory


@dataclass
class SemanticEntry:
    """A semantic knowledge entry."""
    key: str
    value: str
    confidence: float  # 0-1, how certain we are about this knowledge
    evidence_count: int  # how many observations support this
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class SemanticMemory:
    """Consolidated semantic memory of the user profile.

    Stores abstract knowledge derived from episodic patterns:
    - Feedback preferences
    - Difficulties by space type
    - Movement patterns
    - Auditory calibration
    """

    def __init__(self):
        self.entries: dict[str, SemanticEntry] = {}

    def update(
        self,
        key: str,
        value: str,
        confidence: float = 0.5,
    ) -> None:
        """Update or create a semantic entry.

        If the entry already exists, increments evidence_count and
        adjusts confidence with a moving average.
        """
        if key in self.entries:
            entry = self.entries[key]
            entry.evidence_count += 1
            alpha = 1.0 / (entry.evidence_count + 1)
            entry.confidence = (1 - alpha) * entry.confidence + alpha * confidence
            entry.value = value
            entry.updated_at = time.time()
        else:
            self.entries[key] = SemanticEntry(
                key=key,
                value=value,
                confidence=confidence,
                evidence_count=1,
            )

    def retrieve(self, key: str) -> SemanticEntry | None:
        """Look up an entry by key."""
        return self.entries.get(key)

    def retrieve_by_prefix(self, prefix: str) -> list[SemanticEntry]:
        """Find entries whose key starts with prefix."""
        return [
            entry for key, entry in self.entries.items()
            if key.startswith(prefix)
        ]

    def retrieve_high_confidence(self, threshold: float = 0.7) -> list[SemanticEntry]:
        """Return entries with high confidence."""
        return [
            entry for entry in self.entries.values()
            if entry.confidence >= threshold
        ]

    def consolidate_from_episodic(
        self,
        episodic: EpisodicMemory,
        min_observations: int = 5,
    ) -> list[str]:
        """Extract patterns from episodic memory and consolidate them.

        Looks for repeated patterns in episodes and generates semantic
        entries. This is the "consolidation" that occurs during the
        system's "sleep".

        Args:
            episodic: episodic memory to analyze
            min_observations: minimum observations to consolidate

        Returns:
            List of new or updated keys
        """
        updated_keys: list[str] = []

        if len(episodic) < min_observations:
            return updated_keys

        # Pattern: actions per class
        class_actions: dict[str, dict[str, int]] = {}
        for ep in episodic.episodes:
            cls = ep.prediction_class
            if cls not in class_actions:
                class_actions[cls] = {}
            action = ep.user_action
            class_actions[cls][action] = class_actions[cls].get(action, 0) + 1

        for cls, actions in class_actions.items():
            total = sum(actions.values())
            if total < min_observations:
                continue

            # Detect frequent hesitation in this class
            hesitate_rate = actions.get("hesitate", 0) / total
            if hesitate_rate > 0.3:
                key = f"difficulty_{cls}"
                self.update(
                    key,
                    f"User hesitates frequently in {cls} ({hesitate_rate:.0%} of the time)",
                    confidence=min(hesitate_rate, 0.95),
                )
                updated_keys.append(key)

            # Detect frequent retreat
            retreat_rate = actions.get("retreat", 0) / total
            if retreat_rate > 0.2:
                key = f"retreat_pattern_{cls}"
                self.update(
                    key,
                    f"User retreats in {cls} ({retreat_rate:.0%})",
                    confidence=min(retreat_rate, 0.95),
                )
                updated_keys.append(key)

            # Detect high user confidence
            advance_rate = actions.get("advance", 0) / total
            if advance_rate > 0.7:
                key = f"confident_{cls}"
                self.update(
                    key,
                    f"User advances confidently in {cls} ({advance_rate:.0%})",
                    confidence=min(advance_rate, 0.95),
                )
                updated_keys.append(key)

        return updated_keys

    def to_context_dict(self) -> dict[str, dict[str, Any]]:
        """Convert to dictionary for Qwen context."""
        return {
            key: {
                "value": entry.value,
                "confidence": round(entry.confidence, 3),
                "evidence": entry.evidence_count,
            }
            for key, entry in self.entries.items()
        }

    def save(self, filepath: str) -> None:
        """Persist to JSON."""
        data = {k: asdict(v) for k, v in self.entries.items()}
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> SemanticMemory:
        """Load from JSON."""
        mem = cls()
        path = Path(filepath)
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            for key, entry_data in data.items():
                mem.entries[key] = SemanticEntry(**entry_data)
        return mem
