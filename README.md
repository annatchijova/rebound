# REBOUND — Biomimetic Sonar Navigation System

Biomimetic sonar system for indoor navigation designed for visually impaired users.
Inspired by bat CF-FM echolocation. Built for the **Global AI Hackathon Series with Qwen Cloud**.

**Track: MemoryAgent** — persistent adaptive memory across multi-turn, cross-session interactions.

---

## What it does

REBOUND emits ultrasonic chirps, listens to echoes, and classifies the acoustic environment
in real time. A Qwen-powered Memory Agent learns user behavior across sessions and
personalizes navigation instructions accordingly.

Pipeline per observation:
1. Emit a CF-FM chirp (exponential sweep, 8 kHz)
2. Capture the echo and deconvolve to obtain the Room Impulse Response (RIR)
3. Extract acoustic features: mel spectrogram (64×32), MFCCs (13×32), RT60, spectral centroid
4. Classify the environment with a lightweight PyTorch CNN
5. Run independent two-pass periodic staircase detector
6. Qwen Memory Agent generates navigation instruction, updates episodic and semantic memory
7. Adaptive user profile adjusts Bayesian priors across sessions

**CNN classes:** open_space · nearby_wall · doorway · corner · corridor  
**Staircases:** separate detector — not a CNN class

---

## Qwen Cloud Integration

The Memory Agent (`src/memory/agent.py`) uses **Qwen-Max** via the DashScope API to:
- Reason over sonar observations in natural language
- Decide which memories to store, update, or forget
- Generate personalized navigation instructions per user

The backend API (`src/cloud/api_server.py`) is deployed on **Alibaba Cloud ECS** via Docker
(`src/cloud/deploy.py`), exposing endpoints for observation processing, user profiles,
and session management.

---

## Results

| Metric | Value |
|---|---|
| val_accuracy (20% held-out split) | 0.950 |
| Tests passing | 66/66 |
| Staircase detector SNR threshold | ~27 dB |
| corner→nearby_wall confusion bias | 26/5 (full dataset) |

Full empirical limitations: [LIMITATIONS.md](LIMITATIONS.md)

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    REBOUND Pipeline                 │
│                                                     │
│  Microphone → Chirp capture → Deconvolution (RIR)  │
│       ↓                                             │
│  Feature extraction (mel, MFCCs, RT60, centroid)   │
│       ↓                                             │
│  CNN Classifier (5 classes) + Staircase Detector   │
│       ↓                                             │
│  Qwen-Max Memory Agent (DashScope API)              │
│       ├── Episodic Memory (recent events)           │
│       ├── Semantic Memory (consolidated rules)      │
│       └── User Profile (Bayesian adaptive priors)  │
│       ↓                                             │
│  Navigation instruction + Haptic pattern            │
└─────────────────────────────────────────────────────┘
         ↑                          ↓
   FastAPI backend          Alibaba Cloud ECS
   (src/cloud/)             Docker deployment
```

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
python3 -m pytest tests/ -q
```

---

## Repository Structure

```
src/
  signal/      chirp, capture, deconvolution, staircase detector
  features/    spectral and geometric feature extraction
  simulation/  RIR generation (pyroomacoustics)
  models/      CNN classifier, training, inference
  memory/      Qwen Memory Agent, user profile, episodic/semantic memory
  cloud/       FastAPI API server + Alibaba Cloud ECS deploy script
  demo/        adaptive real-time visualization (matplotlib)
future/
  ewc.py             Elastic Weight Consolidation — deferred to v2
  augmentation.py    noise augmentation (known bug, see header) — deferred to v2
tests/             66 unit tests
LIMITATIONS.md     empirically measured system constraints
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE)
