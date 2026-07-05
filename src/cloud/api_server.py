
"""
FastAPI server for REBOUND on Alibaba Cloud ECS.

Endpoints:
- POST /predict    — raw audio in → CNN classification + distance + stair detection
- POST /process    — classification result → Memory Agent → navigation instruction
- GET  /chirp      — download reference chirp (for mobile client sync)
- GET  /profile    — return a user's profile
- POST /session    — start a new session
- GET  /health     — health check

Deploy: Alibaba Cloud ECS with Docker.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.memory.agent import AsyncQwenMemoryAgent, MockAsyncQwenMemoryAgent
from src.memory.episodic import Episode, EpisodicMemory
from src.memory.profile import UserProfile
from src.memory.semantic import SemanticMemory
from src.signal.chirp import generate_chirp, CHIRP_PARAMS
from src.signal.deconvolution import adaptive_wiener, estimate_snr
from src.simulation.room_generator import SPACE_CLASSES, CLASS_NAMES_TO_ID

logger = logging.getLogger(__name__)

TRAINING_SAMPLE_RATE = CHIRP_PARAMS["sample_rate"]  # 44100
CHECKPOINT_PATH = os.environ.get(
    "REBOUND_CHECKPOINT", "models/checkpoints/best_model.pt"
)

_state: dict = {}
_user_locks: dict[str, asyncio.Lock] = {}

_VALID_USER_ID = re.compile(r'[a-zA-Z0-9_\-]{1,64}')


def _validate_user_id(user_id: str) -> None:
    """Validate user_id format to prevent path traversal."""
    if not _VALID_USER_ID.fullmatch(user_id):
        raise HTTPException(400, f"Invalid user_id: {user_id!r}")


def _load_model_at_startup() -> None:
    """Load CNN model and chirp reference at server startup."""
    from src.models.inference import load_model

    ckpt = Path(CHECKPOINT_PATH)
    if ckpt.exists():
        model, scaler, device = load_model(str(ckpt), device="cpu")
        _state["model"] = model
        _state["scaler"] = scaler
        _state["device"] = device
        logger.info("Model loaded from %s on %s", ckpt, device)
    else:
        _state["model"] = None
        _state["scaler"] = None
        _state["device"] = "cpu"
        logger.warning("No checkpoint at %s — /predict disabled", ckpt)

    _state["chirp_ref"] = generate_chirp()


@asynccontextmanager
async def lifespan(app: FastAPI):
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if api_key:
        _state["agent"] = AsyncQwenMemoryAgent(api_key=api_key)
    else:
        _state["agent"] = MockAsyncQwenMemoryAgent(api_key="mock")

    _state["profiles"] = {}
    _state["episodic"] = {}
    _state["semantic"] = {}

    _load_model_at_startup()

    yield
    await _state["agent"].close()


app = FastAPI(
    title="REBOUND API",
    version="0.2.0",
    description="Biomimetic sonar navigation — Qwen Cloud Hackathon",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


VALID_CLASS_NAMES = set(SPACE_CLASSES.values()) | {"stairs"}


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


def _get_user_lock(user_id: str) -> asyncio.Lock:
    """Get or create per-user lock."""
    if user_id not in _user_locks:
        _user_locks[user_id] = asyncio.Lock()
    return _user_locks[user_id]


def _get_or_create_user(user_id: str) -> tuple[UserProfile, EpisodicMemory, SemanticMemory]:
    """Get or create user state. Must be called under per-user lock."""
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
    _validate_user_id(req.user_id)

    if req.prediction.class_name not in VALID_CLASS_NAMES:
        raise HTTPException(400, f"Invalid class_name: {req.prediction.class_name!r}")

    lock = _get_user_lock(req.user_id)
    async with lock:
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

        response = await agent.process_observation(
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
    _validate_user_id(user_id)
    lock = _get_user_lock(user_id)
    async with lock:
        profile, episodic, semantic = _get_or_create_user(user_id)
        return {
            "profile": profile.to_summary(),
            "episodic_stats": episodic.stats(),
            "semantic": semantic.to_context_dict(),
        }


@app.post("/session")
async def start_session(req: SessionRequest) -> dict:
    """Start a new session for a user."""
    _validate_user_id(req.user_id)
    lock = _get_user_lock(req.user_id)
    async with lock:
        profile, _, _ = _get_or_create_user(req.user_id)
        profile.start_session()
        return {
            "user_id": req.user_id,
            "session_number": profile.total_sessions,
        }


class AudioPredictRequest(BaseModel):
    """Raw audio from mobile client for server-side prediction."""
    audio_base64: str = Field(
        description="Base64-encoded PCM audio buffer (dtype per audio_dtype)"
    )
    audio_dtype: Literal["float64", "float32"] = Field(
        default="float64",
        description="Numeric dtype of the audio buffer. float32 halves upload size."
    )
    sample_rate: int = Field(
        ge=8000, le=96000,
        description="Client capture sample rate in Hz"
    )
    gyroscope_pitch_deg: float = Field(
        default=0.0,
        description="Device pitch angle for stair direction"
    )


class AudioPredictResponse(BaseModel):
    class_name: str
    class_id: int
    confidence: float
    distance_m: float
    probabilities: dict[str, float]
    stairs_detection: dict
    stair_direction: str
    stair_message: str
    snr_db: float
    features_summary: dict[str, float]


@app.post("/predict", response_model=AudioPredictResponse)
async def predict_from_audio(req: AudioPredictRequest) -> AudioPredictResponse:
    """Full prediction pipeline: raw audio → deconvolution → CNN → result.

    The mobile client captures audio after emitting the chirp, encodes the
    buffer as base64 float64, and sends it here. The server runs the full
    DSP + ML pipeline and returns the classification.
    """
    if _state.get("model") is None:
        raise HTTPException(503, "Model not loaded — no checkpoint available")

    from src.models.inference import predict as run_predict
    from src.signal.stairs import estimate_stair_geometry, estimate_stair_direction, build_stair_message

    t_start = time.perf_counter()
    try:
        audio_bytes = base64.b64decode(req.audio_base64)
        dtype = np.float32 if req.audio_dtype == "float32" else np.float64
        audio = np.frombuffer(audio_bytes, dtype=dtype).astype(np.float64)
    except Exception as e:
        raise HTTPException(400, f"Invalid audio data: {e}")

    if len(audio) < 100:
        raise HTTPException(400, "Audio buffer too short")

    # Resample to training sample rate if needed
    if req.sample_rate != TRAINING_SAMPLE_RATE:
        from scipy.signal import resample
        n_target = int(len(audio) * TRAINING_SAMPLE_RATE / req.sample_rate)
        audio = resample(audio, n_target)

    # Deconvolve: captured signal → RIR
    chirp_ref = _state["chirp_ref"]
    snr_db = estimate_snr(audio)
    rir = adaptive_wiener(audio, chirp_ref, snr_estimate_db=max(snr_db, 1.0))
    t_deconv = time.perf_counter()

    # Run CNN + stair detector
    result = run_predict(
        _state["model"], _state["scaler"], rir,
        sample_rate=TRAINING_SAMPLE_RATE, device=_state["device"],
    )

    # Stair geometry + direction from gyroscope
    stair_direction = "undetermined"
    stair_message = ""
    stairs = result.get("stairs_detection", {})
    if stairs.get("is_stair") and result["class_name"] == "stairs":
        stair_direction = estimate_stair_direction(req.gyroscope_pitch_deg)
        geometry = estimate_stair_geometry(
            stairs["echo_spacing_m"], stairs["n_steps_detected"]
        )
        stair_message = build_stair_message(geometry, stair_direction)

    # Extract features summary for /process
    from src.features.spectral import extract_features
    from src.features.geometric import estimate_echo_strength
    features = extract_features(rir, sample_rate=TRAINING_SAMPLE_RATE)
    echo_strength = estimate_echo_strength(rir, sample_rate=TRAINING_SAMPLE_RATE)

    t_end = time.perf_counter()
    print(
        f"[predict] class={result['class_name']} conf={result['confidence']:.2f} "
        f"snr={snr_db:.1f}dB n={len(audio)} sr_in={req.sample_rate} "
        f"dtype={req.audio_dtype} pitch={req.gyroscope_pitch_deg:.0f} "
        f"t_deconv={(t_deconv - t_start) * 1000:.0f}ms t_total={(t_end - t_start) * 1000:.0f}ms",
        flush=True,
    )

    return AudioPredictResponse(
        class_name=result["class_name"],
        class_id=result["class_id"],
        confidence=result["confidence"],
        distance_m=result["distance_m"],
        probabilities=result["probabilities"],
        stairs_detection=stairs,
        stair_direction=stair_direction,
        stair_message=stair_message,
        snr_db=round(snr_db, 1),
        features_summary={
            "rt60": features["rt60"],
            "spectral_centroid": features["spectral_centroid"],
            "echo_strength": echo_strength,
        },
    )


@app.get("/chirp")
async def get_chirp() -> dict:
    """Return the reference chirp as base64 for mobile client sync.

    The client plays this through the speaker, records the echo,
    and sends the recording to POST /predict.
    """
    chirp = _state["chirp_ref"]
    chirp_b64 = base64.b64encode(chirp.astype(np.float64).tobytes()).decode()
    return {
        "chirp_base64": chirp_b64,
        "sample_rate": TRAINING_SAMPLE_RATE,
        "n_samples": len(chirp),
        "duration_s": len(chirp) / TRAINING_SAMPLE_RATE,
    }


@app.get("/health")
async def health() -> dict:
    """Health check."""
    return {
        "status": "ok",
        "model_loaded": _state.get("model") is not None,
        "agent_type": type(_state.get("agent", None)).__name__,
        "active_users": len(_state.get("profiles", {})),
    }


# Serve frontend PWA — must be LAST (catch-all for static files)
_static_dir = Path(__file__).resolve().parent.parent.parent / "static"
if _static_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="frontend")
