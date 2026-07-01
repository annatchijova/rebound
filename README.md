# REBOUND — Biomimetic Sonar Navigation System

Biomimetic sonar system for indoor navigation designed for visually impaired users.
Inspired by bat CF-FM echolocation. Built for the **Global AI Hackathon Series with Qwen Cloud**.

**Track: MemoryAgent** — persistent adaptive memory across multi-turn, cross-session interactions.

**Author & Developer**
This project was designed, architected, and developed in full by **Olga Vasilieva**. The GitHub repository is hosted under the account of Anna Tchijova (her daughter) for technical convenience. Anna Tchijova had no participation in the code, design, or technical decisions of this project.

---

## What it does

REBOUND emits ultrasonic chirps from the device speaker, listens to echoes via the
microphone, and classifies the acoustic environment in real time. A Qwen-powered Memory
Agent learns user behavior across sessions and personalizes navigation instructions.

Pipeline per observation:
1. Emit a CF-FM chirp (exponential sweep, 8 kHz) through the speaker
2. Capture the echo and deconvolve to obtain the Room Impulse Response (RIR)
3. Extract acoustic features: mel spectrogram (64x32), MFCCs (13x32), RT60, spectral centroid
4. Classify the environment with a lightweight PyTorch CNN (6 classes)
5. Run independent two-pass periodic staircase detector (DSP reinforcement)
6. Qwen Memory Agent generates navigation instruction, updates episodic and semantic memory
7. Adaptive user profile adjusts Bayesian priors across sessions
8. Haptic feedback delivered via device vibration

**CNN classes:** open_space · nearby_wall · doorway · corner · corridor · stairs  
**Staircases:** CNN class + independent DSP detector for mutual reinforcement

---

## Architecture

```
  Mobile (PWA)                        Alibaba Cloud ECS
 ┌──────────────┐                   ┌──────────────────────────────────┐
 │ Web Audio API │── audio buffer ──▶│ POST /predict                    │
 │ getUserMedia  │   + sample_rate   │   ├─ Wiener deconvolution (RIR) │
 │ speaker emit  │                   │   ├─ Feature extraction          │
 │               │                   │   ├─ CNN classifier (6 classes)  │
 │ DeviceOrienta │── gyro pitch ────▶│   ├─ Stair detector (DSP)       │
 │ tion Event    │                   │   └─ predict() result            │
 │               │◀── JSON ─────────│                                  │
 │ navigator     │   instruction     │ POST /process                    │
 │ .vibrate()    │   haptic_pattern  │   ├─ Qwen-Plus Memory Agent     │
 │               │   distance_m      │   ├─ Episodic/Semantic memory   │
 │               │   class_name      │   └─ User profile update        │
 └──────────────┘                   └──────────────────────────────────┘
                                              │
                                     DashScope API (Qwen-Plus)
```

---

## Qwen Cloud Integration

The Memory Agent (`src/memory/agent.py`) uses **Qwen-Plus** via the DashScope API to:
- Reason over sonar observations in natural language
- Decide which memories to store, update, or forget (selective forgetting)
- Generate personalized navigation instructions per user
- Adjust Bayesian confidence priors based on user behavior

The backend API (`src/cloud/api_server.py`) is deployed on **Alibaba Cloud ECS** via Docker,
exposing endpoints for audio prediction, observation processing, user profiles, and session
management. All agent calls are async (`httpx.AsyncClient`) to handle concurrent users.

---

## Results

| Metric | Value |
|---|---|
| CNN classes | 6 (open_space, nearby_wall, doorway, corner, corridor, stairs) |
| val_accuracy (20% stratified split) | 0.953 |
| val_distance_MAE | 0.302 m |
| Tests passing | 88/88 |
| Staircase detector SNR threshold | ~27 dB |
| Training device | NVIDIA RTX 3090 |

Full empirical limitations: [LIMITATIONS.md](LIMITATIONS.md)

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict` | Raw audio in → classification + distance + stair detection |
| `POST` | `/process` | Classification result → Memory Agent → navigation instruction |
| `GET` | `/profile/{user_id}` | User profile, episodic stats, semantic memory |
| `POST` | `/session` | Start new session for user |
| `GET` | `/health` | Health check |

---

## Installation

```bash
pip install -r requirements.txt
```

Set your DashScope API key:
```bash
export DASHSCOPE_API_KEY=your_key_here
```

CUDA-capable GPU required for training. CPU sufficient for inference and demo.

---

## Usage

**Generate dataset and train:**
```bash
python3 -m src.simulation.dataset_builder
python3 -m src.models.train --data data/processed/dataset.npz --epochs 50
```

**Run the local demo (mock agent):**
```bash
python3 -m src.demo.adaptive_demo
```

**Run the local demo (real Qwen API):**
```bash
python3 -m src.demo.adaptive_demo --use-qwen
```

**Start the API server locally:**
```bash
uvicorn src.cloud.api_server:app --host 0.0.0.0 --port 8000
```

**Deploy to Alibaba Cloud ECS:**
```bash
python3 -m src.cloud.deploy --registry <your-acr-registry> --tag v0.1
```

**Run tests:**
```bash
python3 -m pytest tests/ -v
```

---

## Repository Structure

```
src/
  signal/      chirp, capture, deconvolution, staircase detector
  features/    spectral and geometric feature extraction
  simulation/  RIR generation (pyroomacoustics + synthetic stairs)
  models/      CNN classifier (6 classes), training, inference
  memory/      Qwen Memory Agent (sync + async), user profile, episodic/semantic memory
  cloud/       FastAPI API server (async) + Alibaba Cloud ECS deploy script
  feedback/    haptic pattern definitions
  demo/        adaptive real-time visualization (matplotlib)
frontend/      React demo UI (Vite + Recharts)
future/
  ewc.py             Elastic Weight Consolidation — deferred to v2
  augmentation.py    noise augmentation (known bug, see header) — deferred to v2
tests/               88 unit + integration tests
LIMITATIONS.md       empirically measured system constraints
CHANGELOG_2026-07-01.md  33-bug remediation changelog
```

---

## Security

- `user_id` validated against `[a-zA-Z0-9_\-]{1,64}` (path traversal prevention)
- `class_name` validated against whitelist (prompt injection prevention)
- LLM multipliers clamped to `[0.1, 10.0]` (hallucination containment)
- Semantic memory values capped at 512 chars (unbounded growth prevention)
- Per-user `asyncio.Lock` (race condition prevention)
- `.dockerignore` excludes `data/profiles/` and `data/checkpoints/`
- **No JWT/API key authentication yet** — deploy behind reverse proxy with auth

---

## License

Apache License 2.0 — see [LICENSE](LICENSE)
