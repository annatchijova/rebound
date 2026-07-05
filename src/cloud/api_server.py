
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
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path

from typing import Literal, Optional

import numpy as np
from fastapi import FastAPI, Header, HTTPException
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

# --- Security / resource limits (env-configurable) ---
# Shared demo token. If unset, endpoints run OPEN and a loud warning is logged.
API_TOKEN = os.environ.get("REBOUND_API_TOKEN", "")
# /predict payload limits: ~1.5 MB decoded, max 2 s of audio at any sample rate
MAX_AUDIO_B64_CHARS = int(os.environ.get("REBOUND_MAX_AUDIO_B64", 2_000_000))
MAX_AUDIO_SECONDS = float(os.environ.get("REBOUND_MAX_AUDIO_SECONDS", 2.0))
# Bound on distinct users kept in memory (LRU eviction, profile saved to disk)
MAX_ACTIVE_USERS = int(os.environ.get("REBOUND_MAX_USERS", 200))
# Concurrent heavy predictions; excess requests get 429 after a short wait
MAX_CONCURRENT_PREDICT = int(os.environ.get("REBOUND_MAX_CONCURRENT_PREDICT", 4))

_state: dict = {}
_user_locks: dict[str, asyncio.Lock] = {}
_user_last_seen: dict[str, float] = {}
_predict_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PREDICT)

_VALID_USER_ID = re.compile(r'[a-zA-Z0-9_\-]{1,64}')


def _check_token(x_api_token: Optional[str]) -> None:
    """Constant-time shared-token check. No-op if REBOUND_API_TOKEN is unset."""
    if not API_TOKEN:
        return
    if not x_api_token or not secrets.compare_digest(x_api_token, API_TOKEN):
        raise HTTPException(401, "Invalid or missing X-API-Token header")


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
        logger.warning("=" * 60)
        logger.warning("DASHSCOPE_API_KEY not set — using MOCK memory agent.")
        logger.warning("Navigation instructions are canned, NOT Qwen-generated.")
        logger.warning("=" * 60)
        print("[startup] WARNING: MOCK agent active (no DASHSCOPE_API_KEY)", flush=True)

    if not API_TOKEN:
        logger.warning("REBOUND_API_TOKEN not set — user-data endpoints are OPEN.")
        print("[startup] WARNING: no API token — /process /session /profile UNAUTHENTICATED", flush=True)

    _state["profiles"] = {}
    _state["episodic"] = {}
    _state["semantic"] = {}

    _load_model_at_startup()

    if _state.get("model") is not None:
        try:
            t0 = time.time()
            from src.models.inference import predict as _warmup_predict
            from src.signal.deconvolution import adaptive_wiener as _warmup_wiener
            _warmup_rir = _warmup_wiener(
                _state["chirp_ref"], _state["chirp_ref"], snr_estimate_db=20.0
            )
            _warmup_predict(
                _state["model"], _state["scaler"], _warmup_rir,
                sample_rate=TRAINING_SAMPLE_RATE, device=_state["device"],
            )
            logger.info("Model warmup complete in %.0fms", (time.time() - t0) * 1000)
        except Exception as e:
            logger.warning("Model warmup failed (non-fatal): %s", e)

    yield
    await _state["agent"].close()


app = FastAPI(
    title="REBOUND API",
    version="0.2.0",
    description="Biomimetic sonar navigation — Qwen Cloud Hackathon",
    lifespan=lifespan,
)

# CORS: the PWA is served from this same origin, so cross-origin access is
# only needed for local dev. Set REBOUND_ALLOWED_ORIGINS="https://mydomain.com"
# in production. Wildcard + credentials is contradictory per spec, so
# credentials are only allowed with an explicit origin list.
_origins = [o.strip() for o in os.environ.get("REBOUND_ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=_origins != ["*"],
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


def _evict_lru_users(keep: str) -> None:
    """Bound memory: keep at most MAX_ACTIVE_USERS in the state dicts.

    Evicts least-recently-seen users (never `keep`), saving their profile to
    disk best-effort. Episodic/semantic memory of evicted users is dropped —
    acceptable for the demo; profiles reload from disk on next visit.
    """
    while len(_state["profiles"]) > MAX_ACTIVE_USERS:
        candidates = [u for u in _user_last_seen if u != keep and u in _state["profiles"]]
        if not candidates:
            break
        oldest = min(candidates, key=lambda u: _user_last_seen[u])
        try:
            _state["profiles"][oldest].save()
        except Exception:
            logger.exception("Failed saving profile for evicted user %s", oldest)
        for d in (_state["profiles"], _state["episodic"], _state["semantic"]):
            d.pop(oldest, None)
        _user_locks.pop(oldest, None)
        _user_last_seen.pop(oldest, None)
        logger.info("Evicted LRU user %s (cap=%d)", oldest, MAX_ACTIVE_USERS)


def _get_or_create_user(user_id: str) -> tuple[UserProfile, EpisodicMemory, SemanticMemory]:
    """Get or create user state. Must be called under per-user lock."""
    _user_last_seen[user_id] = time.time()
    if user_id not in _state["profiles"]:
        _state["profiles"][user_id] = UserProfile.load(user_id)
        _state["episodic"][user_id] = EpisodicMemory()
        _state["semantic"][user_id] = SemanticMemory()
        _evict_lru_users(keep=user_id)

    return (
        _state["profiles"][user_id],
        _state["episodic"][user_id],
        _state["semantic"][user_id],
    )


@app.post("/process", response_model=ObservationResponse)
async def process_observation(
    req: ObservationRequest,
    x_api_token: Optional[str] = Header(default=None, alias="X-API-Token"),
) -> ObservationResponse:
    """Process a sonar observation and return instructions."""
    _check_token(x_api_token)
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
async def get_profile(
    user_id: str,
    x_api_token: Optional[str] = Header(default=None, alias="X-API-Token"),
) -> dict:
    """Return a user's profile."""
    _check_token(x_api_token)
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
async def start_session(
    req: SessionRequest,
    x_api_token: Optional[str] = Header(default=None, alias="X-API-Token"),
) -> dict:
    """Start a new session for a user."""
    _check_token(x_api_token)
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


def _run_prediction_pipeline(
    audio: np.ndarray, sample_rate: int, gyro_pitch_deg: float
) -> AudioPredictResponse:
    """Synchronous heavy pipeline — runs in a worker thread, NOT the event loop."""
    from src.models.inference import predict as run_predict
    from src.signal.stairs import estimate_stair_geometry, estimate_stair_direction, build_stair_message
    from src.features.spectral import extract_features
    from src.features.geometric import estimate_echo_strength

    t_start = time.perf_counter()

    # Resample to training sample rate if needed
    if sample_rate != TRAINING_SAMPLE_RATE:
        from scipy.signal import resample
        n_target = int(len(audio) * TRAINING_SAMPLE_RATE / sample_rate)
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
        stair_direction = estimate_stair_direction(gyro_pitch_deg)
        geometry = estimate_stair_geometry(
            stairs["echo_spacing_m"], stairs["n_steps_detected"]
        )
        stair_message = build_stair_message(geometry, stair_direction)

    # Extract features summary for /process
    features = extract_features(rir, sample_rate=TRAINING_SAMPLE_RATE)
    echo_strength = estimate_echo_strength(rir, sample_rate=TRAINING_SAMPLE_RATE)

    t_end = time.perf_counter()
    print(
        f"[predict] class={result['class_name']} conf={result['confidence']:.2f} "
        f"snr={snr_db:.1f}dB n={len(audio)} sr_in={sample_rate} "
        f"pitch={gyro_pitch_deg:.0f} "
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


@app.post("/predict", response_model=AudioPredictResponse)
async def predict_from_audio(
    req: AudioPredictRequest,
    x_api_token: Optional[str] = Header(default=None, alias="X-API-Token"),
) -> AudioPredictResponse:
    """Full prediction pipeline: raw audio → deconvolution → CNN → result.

    The mobile client captures audio after emitting the chirp, encodes the
    buffer as base64 (float32 or float64), and sends it here. The heavy DSP +
    ML work runs in a worker thread, bounded by a concurrency semaphore.
    """
    _check_token(x_api_token)

    if _state.get("model") is None:
        raise HTTPException(503, "Model not loaded — no checkpoint available")

    # --- Size limits BEFORE any expensive work ---
    if len(req.audio_base64) > MAX_AUDIO_B64_CHARS:
        raise HTTPException(413, "Audio payload too large")

    try:
        audio_bytes = base64.b64decode(req.audio_base64)
        dtype = np.float32 if req.audio_dtype == "float32" else np.float64
        audio = np.frombuffer(audio_bytes, dtype=dtype).astype(np.float64)
    except Exception as e:
        raise HTTPException(400, f"Invalid audio data: {e}")

    if len(audio) < 100:
        raise HTTPException(400, "Audio buffer too short")
    if len(audio) > MAX_AUDIO_SECONDS * req.sample_rate:
        raise HTTPException(413, f"Audio longer than {MAX_AUDIO_SECONDS}s limit")

    # --- Bounded concurrency: reject instead of queueing forever ---
    try:
        await asyncio.wait_for(_predict_semaphore.acquire(), timeout=3.0)
    except asyncio.TimeoutError:
        raise HTTPException(429, "Server busy — retry")
    try:
        return await asyncio.to_thread(
            _run_prediction_pipeline, audio, req.sample_rate, req.gyroscope_pitch_deg
        )
    finally:
        _predict_semaphore.release()


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
