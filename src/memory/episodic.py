"""
Episodic memory with temporal decay.

Stores recent navigation events. Old events decay in relevance and
are removed, simulating natural episodic forgetting.

Track MemoryAgent: "efficient storage, retrieval, and selective forgetting"
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Episode:
    """A single navigation event."""
    timestamp: float
    session_id: int
    prediction_class: str
    prediction_confidence: float
    distance_m: float
    user_action: str
    features_summary: dict[str, float]
    relevance: float = 1.0  # decays over time

    def to_dict(self) -> dict:
        return asdict(self)


class EpisodicMemory:
    """Episodic buffer with temporal decay and limited capacity.

    Implements selective forgetting: old episodes decay in relevance
    and are removed when they fall below a threshold.
    """

    def __init__(
        self,
        max_episodes: int = 200,
        decay_rate: float = 0.995,
        relevance_threshold: float = 0.1,
    ):
        self.max_episodes = max_episodes
        self.decay_rate = decay_rate
        self.relevance_threshold = relevance_threshold
        self.episodes: list[Episode] = []

    def store(self, episode: Episode) -> None:
        """Store a new episode and apply decay to existing ones."""
        for ep in self.episodes:
            ep.relevance *= self.decay_rate

        self.episodes.append(episode)

        # Selective forgetting: remove episodes below relevance threshold
        self.episodes = [
            ep for ep in self.episodes
            if ep.relevance >= self.relevance_threshold
        ]

        # If still over capacity, remove oldest
        if len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[-self.max_episodes:]

    def retrieve_recent(self, n: int = 10) -> list[Episode]:
        """Return the N most recent episodes."""
        return self.episodes[-n:]

    def retrieve_by_class(self, class_name: str, n: int = 5) -> list[Episode]:
        """Return episodes of a specific class, sorted by relevance."""
        filtered = [
            ep for ep in self.episodes
            if ep.prediction_class == class_name
        ]
        filtered.sort(key=lambda ep: ep.relevance, reverse=True)
        return filtered[:n]

    def retrieve_by_action(self, action: str) -> list[Episode]:
        """Return episodes where the user performed a specific action."""
        return [
            ep for ep in self.episodes
            if ep.user_action == action
        ]

    def get_session_summary(self, session_id: int) -> dict[str, Any]:
        """Summary of a specific session."""
        session_eps = [
            ep for ep in self.episodes
            if ep.session_id == session_id
        ]
        if not session_eps:
            return {"session_id": session_id, "n_events": 0}

        classes = [ep.prediction_class for ep in session_eps]
        actions = [ep.user_action for ep in session_eps]

        return {
            "session_id": session_id,
            "n_events": len(session_eps),
            "classes_visited": list(set(classes)),
            "actions": {a: actions.count(a) for a in set(actions)},
            "mean_confidence": sum(ep.prediction_confidence for ep in session_eps) / len(session_eps),
        }

    def decay_older_than(self, n_sessions: int, current_session: int) -> int:
        """Apply aggressive decay to episodes from old sessions.

        Args:
            n_sessions: sessions to keep
            current_session: current session ID

        Returns:
            Number of episodes removed
        """
        threshold_session = current_session - n_sessions
        n_before = len(self.episodes)

        for ep in self.episodes:
            if ep.session_id < threshold_session:
                ep.relevance *= 0.5  # aggressive decay

        self.episodes = [
            ep for ep in self.episodes
            if ep.relevance >= self.relevance_threshold
        ]

        return n_before - len(self.episodes)

    def to_context_string(self, n: int = 10) -> str:
        """Convert recent episodes to string for Qwen context."""
        recent = self.retrieve_recent(n)
        lines = []
        for ep in recent:
            lines.append(
                f"{ep.prediction_class}@{ep.distance_m:.1f}m "
                f"conf={ep.prediction_confidence:.2f} "
                f"action={ep.user_action}"
            )
        return " | ".join(lines)

    def __len__(self) -> int:
        return len(self.episodes)

    def stats(self) -> dict[str, Any]:
        """Episodic buffer statistics."""
        if not self.episodes:
            return {"n_episodes": 0}
        relevances = [ep.relevance for ep in self.episodes]
        return {
            "n_episodes": len(self.episodes),
            "mean_relevance": sum(relevances) / len(relevances),
            "min_relevance": min(relevances),
            "max_relevance": max(relevances),
        }
