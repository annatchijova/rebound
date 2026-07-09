"""
Qwen Memory Agent — brain of the adaptive memory system.

Integrates with Qwen Cloud (DashScope API) to reason about:
- What to remember and what to forget (selective forgetting)
- How to interpret navigation context
- What instructions to give the user based on their profile
- How to adjust confidence priors

This module is the core of the MemoryAgent hackathon track.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

from src.memory.episodic import EpisodicMemory, Episode
from src.memory.profile import UserProfile
from src.memory.semantic import SemanticMemory
from src.simulation.room_generator import SPACE_CLASSES

# Qwen Cloud API (DashScope)
QWEN_API_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
QWEN_MODEL = "qwen-plus"

SYSTEM_PROMPT = """\
You are REBOUND's Memory Agent — a biomimetic sonar navigation system for visually \
impaired users. Your role is to manage the user's adaptive memory profile.

You receive acoustic classification results from a CNN classifier and user behavioral \
signals. You must decide:

1. **Navigation instruction**: Clear, concise guidance for the user based on the \
detected space and their profile.
2. **Confidence adjustments**: How to modify the Bayesian priors for each space class \
based on this interaction.
3. **Memory operations**: What to store, update, or forget in the user's memory.
4. **Haptic pattern**: What vibration feedback to suggest.

## Memory Types You Manage

- **Episodic**: Recent events with temporal decay. Store new events, decay old ones.
- **Semantic**: Consolidated knowledge about the user (patterns, preferences). \
Update when you detect a pattern across episodes.
- **Procedural**: Calibration data. Update when user behavior suggests miscalibration.

## Selective Forgetting Rules

- Episodic memories older than 20 sessions should decay aggressively
- Contradicted semantic memories should have confidence reduced
- If a user consistently succeeds in a class, reduce the episodic detail for that class
- Never forget procedural calibration data unless explicitly overridden

## Space Classes

The system classifies spaces into: open_space, nearby_wall, doorway, corner, \
corridor, stairs.

## User Profile Context

The user profile contains:
- class_weights: Bayesian priors per space class (1.0 = neutral)
- accuracy_by_class: historical accuracy per class
- total_interactions: experience level

Adjust your guidance based on experience level:
- New user (<20 interactions): More descriptive, slower, confirmatory
- Experienced user (>100 interactions): Concise, fast, confidence-based

## Response Format

ALWAYS respond with valid JSON matching this schema:
{
  "navigation_instruction": "string — clear instruction for the user",
  "confidence_adjustment": {"class_name": multiplier, ...},
  "memory_ops": [
    {"op": "store_episodic|update_semantic|decay_episodic|reduce_semantic_confidence",
     "key": "string (for semantic ops)",
     "value": "string",
     "older_than_sessions": number (for decay ops)}
  ],
  "haptic_pattern": "single_pulse|double_pulse|continuous_low|continuous_high|double_pulse_slow|stair_alert|none",
  "reasoning": "brief explanation of your memory decisions"
}
"""


@dataclass
class AgentResponse:
    """Parsed response from the Memory Agent."""
    navigation_instruction: str
    confidence_adjustment: dict[str, float]
    memory_ops: list[dict[str, Any]]
    haptic_pattern: str
    reasoning: str
    raw_response: str


class QwenMemoryAgent:
    """Memory agent using Qwen for adaptive reasoning."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = QWEN_MODEL,
        base_url: str = QWEN_API_URL,
    ):
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.model = model
        self.base_url = base_url
        self.client = httpx.Client(timeout=30.0)

    def process_observation(
        self,
        user_profile: UserProfile,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        prediction: dict[str, Any],
        features_summary: dict[str, float],
        user_action: str,
        session_id: int,
    ) -> AgentResponse:
        """Process a sonar observation and decide memory actions.

        Args:
            user_profile: user profile
            episodic: current episodic memory
            semantic: current semantic memory
            prediction: {"class": str, "confidence": float, "distance_m": float}
            features_summary: {"rt60": float, "spectral_centroid": float, "echo_strength": float}
            user_action: user action ("advance", "hesitate", "retreat", "ignore")
            session_id: current session ID

        Returns:
            AgentResponse with instructions and memory operations
        """
        context = self._build_context(
            user_profile, episodic, semantic,
            prediction, features_summary, user_action, session_id
        )

        raw = self._call_qwen(context)
        return self._parse_response(raw)

    def _build_context(
        self,
        profile: UserProfile,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        prediction: dict[str, Any],
        features: dict[str, float],
        action: str,
        session_id: int,
    ) -> str:
        """Build context message for Qwen."""
        context = {
            "user_profile": profile.to_summary(),
            "episodic_recent": episodic.to_context_string(n=10),
            "episodic_stats": episodic.stats(),
            "semantic_knowledge": semantic.to_context_dict(),
            "current_observation": {
                "prediction": prediction,
                "features_summary": features,
                "user_action": action,
                "session_id": session_id,
            },
        }
        return json.dumps(context, indent=2, ensure_ascii=False)

    def _call_qwen(self, user_message: str) -> str:
        """Call Qwen Cloud API (DashScope compatible)."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.3,
            "response_format": {"type": "json_object"},
        }

        import time as _time

        for attempt in range(3):
            try:
                response = self.client.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except (httpx.HTTPStatusError, httpx.RequestError, KeyError, json.JSONDecodeError) as e:
                if attempt < 2:
                    _time.sleep(2 ** attempt)
                    continue
                return json.dumps({
                    "navigation_instruction": "Processing error. Proceed with caution.",
                    "confidence_adjustment": {},
                    "memory_ops": [],
                    "haptic_pattern": "double_pulse",
                    "reasoning": f"API error after 3 attempts: {type(e).__name__}",
                    })

    def _parse_response(self, raw: str) -> AgentResponse:
        """Parse JSON response from Qwen."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return AgentResponse(
                navigation_instruction="Error processing agent response.",
                confidence_adjustment={},
                memory_ops=[],
                haptic_pattern="none",
                reasoning=f"JSON parse error: {raw[:200]}",
                raw_response=raw,
            )

        return AgentResponse(
            navigation_instruction=data.get("navigation_instruction", ""),
            confidence_adjustment=data.get("confidence_adjustment", {}),
            memory_ops=data.get("memory_ops", []),
            haptic_pattern=data.get("haptic_pattern", "none"),
            reasoning=data.get("reasoning", ""),
            raw_response=raw,
        )

    def apply_memory_ops(
        self,
        response: AgentResponse,
        profile: UserProfile,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        current_session: int,
    ) -> None:
        """Apply memory operations decided by Qwen.

        Args:
            response: agent response
            profile: user profile
            episodic: episodic memory
            semantic: semantic memory
            current_session: current session ID
        """
        # Apply confidence adjustments to profile
        for class_name, multiplier in response.confidence_adjustment.items():
            class_id = None
            for cid, cname in SPACE_CLASSES.items():
                if cname == class_name:
                    class_id = cid
                    break
            if class_id is not None:
                multiplier = max(0.1, min(10.0, multiplier))
                profile.class_weights[class_id] *= multiplier

        # Normalize weights
        mean_w = sum(profile.class_weights) / len(profile.class_weights)
        if mean_w > 0:
            profile.class_weights = [w / mean_w for w in profile.class_weights]

        # Apply memory operations
        for op in response.memory_ops:
            op_type = op.get("op", "")

            if op_type == "store_episodic":
                pass  # Episode already stored before calling the agent

            elif op_type == "update_semantic":
                key = op.get("key", "")
                value = op.get("value", "")
                if len(key) > 128 or len(value) > 512:
                    continue
                if key and value:
                    semantic.update(key, value, confidence=0.6)

            elif op_type == "decay_episodic":
                n_sessions = op.get("older_than_sessions", 20)
                episodic.decay_older_than(n_sessions, current_session)

            elif op_type == "reduce_semantic_confidence":
                key = op.get("key", "")
                semantic.reduce_confidence(key, factor=0.7)

            else:
                logger.warning("Unknown op_type from LLM: %s", op_type)

    def close(self) -> None:
        """Close the HTTP client."""
        self.client.close()


class AsyncQwenMemoryAgent:
    """Async memory agent using Qwen for the FastAPI server (non-blocking)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = QWEN_MODEL,
        base_url: str = QWEN_API_URL,
    ):
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.model = model
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self._sync_agent = QwenMemoryAgent.__new__(QwenMemoryAgent)

    async def process_observation(
        self,
        user_profile: UserProfile,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        prediction: dict[str, Any],
        features_summary: dict[str, float],
        user_action: str,
        session_id: int,
    ) -> AgentResponse:
        context = QwenMemoryAgent._build_context(
            self._sync_agent, user_profile, episodic, semantic,
            prediction, features_summary, user_action, session_id
        )
        raw = await self._call_qwen(context)
        return QwenMemoryAgent._parse_response(self._sync_agent, raw)

    async def _call_qwen(self, user_message: str) -> str:
        """Call Qwen Cloud API asynchronously."""
        import asyncio as _asyncio

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.3,
            "response_format": {"type": "json_object"},
        }

        for attempt in range(3):
            try:
                response = await self.client.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except (httpx.HTTPStatusError, httpx.RequestError, KeyError, json.JSONDecodeError) as e:
                if attempt < 2:
                    await _asyncio.sleep(2 ** attempt)
                    continue
                return json.dumps({
                    "navigation_instruction": "Processing error. Proceed with caution.",
                    "confidence_adjustment": {},
                    "memory_ops": [],
                    "haptic_pattern": "double_pulse",
                    "reasoning": f"API error after 3 attempts: {type(e).__name__}",
                })

    def apply_memory_ops(
        self,
        response: AgentResponse,
        profile: UserProfile,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        current_session: int,
    ) -> None:
        """Delegates to sync implementation (CPU-only, no I/O)."""
        QwenMemoryAgent.apply_memory_ops(
            self._sync_agent, response, profile, episodic, semantic, current_session
        )

    async def close(self) -> None:
        """Close the async HTTP client."""
        await self.client.aclose()


class MockQwenMemoryAgent(QwenMemoryAgent):
    """Mock agent for testing and demo without API key.

    Generates deterministic responses based on simple rules.
    """

    def _call_qwen(self, user_message: str) -> str:
        """Generate response without calling the API."""
        return _mock_response_from_context(user_message)


def _mock_response_from_context(user_message: str) -> str:
    """Shared mock response logic for sync and async mock agents."""
    context = json.loads(user_message)
    obs = context["current_observation"]
    prediction = obs["prediction"]
    action = obs["user_action"]
    profile = context["user_profile"]

    class_name = prediction["class"]
    distance = prediction["distance_m"]

    instructions = {
        "open_space": "Open space. No nearby obstacles.",
        "nearby_wall": f"Wall detected at {distance:.1f} meters ahead.",
        "doorway": f"Opening detected at {distance:.1f} meters. Doorway.",
        "corner": f"Corner detected at {distance:.1f} meters.",
        "corridor": f"Corridor. Lateral walls at {distance:.1f} meters.",
        "stairs": f"Staircase detected at {distance:.1f} meters ahead.",
    }

    instruction = instructions.get(class_name, f"{class_name} at {distance:.1f}m")

    adj = {}
    if action == "hesitate":
        adj[class_name] = 0.95
    elif action == "retreat":
        adj[class_name] = 0.85
    elif action == "advance":
        adj[class_name] = 1.05

    memory_ops = [
        {"op": "store_episodic", "value": f"user {action} at {class_name}@{distance:.1f}m"}
    ]

    total = profile.get("total_interactions", 0)
    if total > 0 and total % 10 == 0:
        memory_ops.append({"op": "decay_episodic", "older_than_sessions": 20})

    if action == "hesitate":
        memory_ops.append({
            "op": "update_semantic",
            "key": f"difficulty_{class_name}",
            "value": f"User hesitates frequently at {class_name}",
        })

    haptic_map = {
        "open_space": "none",
        "nearby_wall": "continuous_high" if distance < 0.5 else "double_pulse",
        "doorway": "double_pulse_slow",
        "corner": "continuous_low",
        "corridor": "single_pulse",
        "stairs": "stair_alert",
    }

    response = {
        "navigation_instruction": instruction,
        "confidence_adjustment": adj,
        "memory_ops": memory_ops,
        "haptic_pattern": haptic_map.get(class_name, "none"),
        "reasoning": f"Mock response for {class_name}, action={action}",
    }
    return json.dumps(response)


class MockAsyncQwenMemoryAgent(AsyncQwenMemoryAgent):
    """Async mock agent for testing the FastAPI server without API key."""

    async def _call_qwen(self, user_message: str) -> str:
        return _mock_response_from_context(user_message)
