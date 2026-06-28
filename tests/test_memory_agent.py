"""Tests for the adaptive memory system."""

import json
import time

import numpy as np
import pytest

from src.memory.agent import MockQwenMemoryAgent, AgentResponse
from src.memory.episodic import Episode, EpisodicMemory
from src.memory.profile import UserProfile, N_CLASSES
from src.memory.semantic import SemanticMemory


class TestUserProfile:
    def test_create_default(self):
        profile = UserProfile(user_id="test")
        assert len(profile.class_weights) == N_CLASSES
        assert all(w == 1.0 for w in profile.class_weights)

    def test_update_implicit_advance(self):
        profile = UserProfile(user_id="test")
        profile.update_implicit(0, "advance")
        assert profile.class_weights[0] > 1.0
        assert profile.total_interactions == 1

    def test_update_implicit_retreat(self):
        profile = UserProfile(user_id="test")
        profile.update_implicit(1, "retreat")
        assert profile.class_weights[1] < 1.0

    def test_update_explicit_correct(self):
        profile = UserProfile(user_id="test")
        profile.update_explicit(2, correct=True)
        assert profile.class_correct[2] == 1

    def test_get_prior_weights(self):
        profile = UserProfile(user_id="test")
        weights = profile.get_prior_weights()
        assert weights.shape == (N_CLASSES,)
        assert weights.dtype == np.float64

    def test_save_and_load(self, tmp_path):
        profile = UserProfile(user_id="test_save")
        profile.update_implicit(0, "advance")
        profile.save(str(tmp_path))

        loaded = UserProfile.load("test_save", str(tmp_path))
        assert loaded.user_id == "test_save"
        assert loaded.total_interactions == 1

    def test_to_summary(self):
        profile = UserProfile(user_id="test")
        summary = profile.to_summary()
        assert "user_id" in summary
        assert "class_weights" in summary


class TestEpisodicMemory:
    def _make_episode(self, cls: str = "nearby_wall", action: str = "advance", session: int = 1):
        return Episode(
            timestamp=time.time(),
            session_id=session,
            prediction_class=cls,
            prediction_confidence=0.85,
            distance_m=1.5,
            user_action=action,
            features_summary={"rt60": 0.3, "spectral_centroid": 3000},
        )

    def test_store(self):
        mem = EpisodicMemory(max_episodes=100)
        ep = self._make_episode()
        mem.store(ep)
        assert len(mem) == 1

    def test_decay(self):
        mem = EpisodicMemory(max_episodes=100, decay_rate=0.5, relevance_threshold=0.1)
        ep1 = self._make_episode()
        mem.store(ep1)
        assert mem.episodes[0].relevance == 1.0

        ep2 = self._make_episode()
        mem.store(ep2)
        assert mem.episodes[0].relevance < 1.0

    def test_selective_forgetting(self):
        mem = EpisodicMemory(max_episodes=100, decay_rate=0.5, relevance_threshold=0.2)
        for i in range(20):
            mem.store(self._make_episode(session=i))
        assert len(mem) < 20

    def test_retrieve_by_class(self):
        mem = EpisodicMemory()
        mem.store(self._make_episode(cls="corridor"))
        mem.store(self._make_episode(cls="nearby_wall"))
        mem.store(self._make_episode(cls="corridor"))

        result = mem.retrieve_by_class("corridor")
        assert len(result) == 2

    def test_to_context_string(self):
        mem = EpisodicMemory()
        mem.store(self._make_episode())
        ctx = mem.to_context_string()
        assert "nearby_wall" in ctx


class TestSemanticMemory:
    def test_update_new(self):
        mem = SemanticMemory()
        mem.update("difficulty_doorway", "user hesitates at doorways", 0.7)
        entry = mem.retrieve("difficulty_doorway")
        assert entry is not None
        assert entry.evidence_count == 1

    def test_update_existing(self):
        mem = SemanticMemory()
        mem.update("key1", "value1", 0.5)
        mem.update("key1", "value2", 0.9)
        entry = mem.retrieve("key1")
        assert entry.evidence_count == 2
        assert entry.value == "value2"

    def test_retrieve_high_confidence(self):
        mem = SemanticMemory()
        mem.update("high", "important", 0.9)
        mem.update("low", "not important", 0.2)
        high = mem.retrieve_high_confidence(0.7)
        assert len(high) == 1
        assert high[0].key == "high"

    def test_consolidate_from_episodic(self):
        episodic = EpisodicMemory()
        for i in range(10):
            episodic.store(Episode(
                timestamp=time.time(),
                session_id=1,
                prediction_class="doorway",
                prediction_confidence=0.7,
                distance_m=1.0,
                user_action="hesitate",
                features_summary={},
            ))

        semantic = SemanticMemory()
        keys = semantic.consolidate_from_episodic(episodic, min_observations=5)
        assert "difficulty_doorway" in keys

    def test_save_and_load(self, tmp_path):
        mem = SemanticMemory()
        mem.update("test_key", "test_value", 0.8)
        filepath = str(tmp_path / "semantic.json")
        mem.save(filepath)

        loaded = SemanticMemory.load(filepath)
        entry = loaded.retrieve("test_key")
        assert entry is not None
        assert entry.value == "test_value"


class TestMockMemoryAgent:
    def test_process_observation(self):
        agent = MockQwenMemoryAgent(api_key="mock")
        profile = UserProfile(user_id="test")
        episodic = EpisodicMemory()
        semantic = SemanticMemory()

        response = agent.process_observation(
            user_profile=profile,
            episodic=episodic,
            semantic=semantic,
            prediction={"class": "nearby_wall", "confidence": 0.85, "distance_m": 1.2},
            features_summary={"rt60": 0.3, "spectral_centroid": 3000, "echo_strength": -10},
            user_action="advance",
            session_id=1,
        )

        assert isinstance(response, AgentResponse)
        assert "wall" in response.navigation_instruction.lower() or "1.2" in response.navigation_instruction
        assert response.haptic_pattern in ("double_pulse", "continuous_high", "single_pulse")

    def test_apply_memory_ops(self):
        agent = MockQwenMemoryAgent(api_key="mock")
        profile = UserProfile(user_id="test")
        episodic = EpisodicMemory()
        semantic = SemanticMemory()

        response = agent.process_observation(
            user_profile=profile,
            episodic=episodic,
            semantic=semantic,
            prediction={"class": "doorway", "confidence": 0.7, "distance_m": 1.0},
            features_summary={"rt60": 0.4, "spectral_centroid": 3200, "echo_strength": -8},
            user_action="hesitate",
            session_id=1,
        )

        agent.apply_memory_ops(response, profile, episodic, semantic, current_session=1)

        entry = semantic.retrieve("difficulty_doorway")
        assert entry is not None

    def test_adaptation_over_iterations(self):
        """Verify profile evolves coherently over 10+ iterations."""
        agent = MockQwenMemoryAgent(api_key="mock")
        profile = UserProfile(user_id="adaptive_test")
        episodic = EpisodicMemory()
        semantic = SemanticMemory()
        profile.start_session()

        for i in range(15):
            response = agent.process_observation(
                user_profile=profile,
                episodic=episodic,
                semantic=semantic,
                prediction={"class": "corridor", "confidence": 0.9, "distance_m": 0.8},
                features_summary={"rt60": 0.2, "spectral_centroid": 2800, "echo_strength": -6},
                user_action="advance",
                session_id=1,
            )
            agent.apply_memory_ops(response, profile, episodic, semantic, current_session=1)
            profile.update_implicit(4, "advance")  # 4 = corridor

            episodic.store(Episode(
                timestamp=time.time(),
                session_id=1,
                prediction_class="corridor",
                prediction_confidence=0.9,
                distance_m=0.8,
                user_action="advance",
                features_summary={"rt60": 0.2, "spectral_centroid": 2800},
            ))

        weights = profile.get_prior_weights()
        assert weights[4] > 1.0
        assert profile.total_interactions == 15

    def test_stairs_response(self):
        """Verify stairs get stair_alert haptic pattern."""
        agent = MockQwenMemoryAgent(api_key="mock")
        profile = UserProfile(user_id="test_stairs")
        episodic = EpisodicMemory()
        semantic = SemanticMemory()

        response = agent.process_observation(
            user_profile=profile,
            episodic=episodic,
            semantic=semantic,
            prediction={"class": "stairs", "confidence": 0.8, "distance_m": 1.5},
            features_summary={"rt60": 0.5, "spectral_centroid": 2500, "echo_strength": -8},
            user_action="hesitate",
            session_id=1,
        )

        assert response.haptic_pattern == "stair_alert"
        assert "stair" in response.navigation_instruction.lower()
