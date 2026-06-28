
"""
FastAPI server for the Memory Agent on Alibaba Cloud.

Endpoints:
- POST /process    — process an observation and return instructions
- GET  /profile    — return a user's profile
- POST /session    — start a new session
- GET  /health     — health check

Deploy: Alibaba Cloud ECS with Docker or Function Compute.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.memory.agent import MockQwenMemoryAgent, QwenMemoryAgent
from src.memory.episodic import Episode, EpisodicMemory
from src.memory.profile import UserProfile
from src.memory.semantic import SemanticMemory
from src.simulation.room_generator import SPACE_CLASSES, CLASS_NAMES_TO_ID


_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if api_key:
        _state["agent"] = QwenMemoryAgent(api_key=api_key)
    else:
        _state["agent"] = MockQwenMemoryAgent(api_key="mock")

    _state["profiles"] = {}
    _state["episodic"] = {}
    _state["semantic"] = {}
    yield
    _state["agent"].close()


app = FastAPI(
    title="REBOUND Memory Agent API",
    version="0.1.0",
    description="Biomimetic sonar navigation memory agent powered by Qwen",
    lifespan=lifespan,
)


class PredictionInput(BaseModel):
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    distance_m: float = Field(ge=0.0)


class FeaturesInput(BaseModel):
    rt60: float
    spectral_centroid: float
    echo_strength: float


class ObservationRequest(BaseModel):
    user_id: str
    prediction: PredictionInput
    features_summary: FeaturesInput
    user_action: Literal["advance", "hesitate", "retreat", "ignore"]
    session_id: int = Field(ge=1)


class ObservationResponse(BaseModel):
    navigation_instruction: str
    confidence_adjustment: dict[str, float]
    memory_ops: list[dict]
    haptic_pattern: str
    reasoning: str
    profile_summary: dict
    episodic_count: int
    semantic_count: int


class SessionRequest(BaseModel):
    user_id: str


def _get_or_create_user(user_id: str) -> tuple[UserProfile, EpisodicMemory, SemanticMemory]:
    """Get or create user state."""
    if user_id not in _state["profiles"]:
        _state["profiles"][user_id] = UserProfile.load(user_id)
        _state["episodic"][user_id] = EpisodicMemory()
        _state["semantic"][user_id] = SemanticMemory()

    return (
        _state["profiles"][user_id],
        _state["episodic"][user_id],
        _state["semantic"][user_id],
    )


@app.post("/process", response_model=ObservationResponse)
async def process_observation(req: ObservationRequest) -> ObservationResponse:
    """Process a sonar observation and return instructions."""
    profile, episodic, semantic = _get_or_create_user(req.user_id)
    agent = _state["agent"]

    prediction = {
        "class": req.prediction.class_name,
        "confidence": req.prediction.confidence,
        "distance_m": req.prediction.distance_m,
    }
    features = {
        "rt60": req.features_summary.rt60,
        "spectral_centroid": req.features_summary.spectral_centroid,
        "echo_strength": req.features_summary.echo_strength,
    }

    response = agent.process_observation(
        user_profile=profile,
        episodic=episodic,
        semantic=semantic,
        prediction=prediction,
        features_summary=features,
        user_action=req.user_action,
        session_id=req.session_id,
    )

    agent.apply_memory_ops(response, profile, episodic, semantic, req.session_id)

    class_id = CLASS_NAMES_TO_ID.get(req.prediction.class_name)
    if class_id is not None:
        profile.update_implicit(class_id, req.user_action)

    episodic.store(Episode(
        timestamp=time.time(),
        session_id=req.session_id,
        prediction_class=req.prediction.class_name,
        prediction_confidence=req.prediction.confidence,
        distance_m=req.prediction.distance_m,
        user_action=req.user_action,
        features_summary=features,
    ))

    return ObservationResponse(
        navigation_instruction=response.navigation_instruction,
        confidence_adjustment=response.confidence_adjustment,
        memory_ops=response.memory_ops,
        haptic_pattern=response.haptic_pattern,
        reasoning=response.reasoning,
        profile_summary=profile.to_summary(),
        episodic_count=len(episodic),
        semantic_count=len(semantic.entries),
    )


@app.get("/profile/{user_id}")
async def get_profile(user_id: str) -> dict:
    """Return a user's profile."""
    profile, episodic, semantic = _get_or_create_user(user_id)
    return {
        "profile": profile.to_summary(),
        "episodic_stats": episodic.stats(),
        "semantic": semantic.to_context_dict(),
    }


@app.post("/session")
async def start_session(req: SessionRequest) -> dict:
    """Start a new session for a user."""
    profile, _, _ = _get_or_create_user(req.user_id)
    profile.start_session()
    return {
        "user_id": req.user_id,
        "session_number": profile.total_sessions,
    }


@app.get("/health")
async def health() -> dict:
    """Health check."""
    return {
        "status": "ok",
        "agent_type": type(_state.get("agent", None)).__name__,
        "active_users": len(_state.get("profiles", {})),
    }
