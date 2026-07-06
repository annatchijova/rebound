# 🦇 REBOUND — Biomimetic Sonar Navigation

![REBOUND logo](visual/3/logo.jpeg)

### Built for someone who can't see the wall in front of them, but deserves to know it's there.

Biomimetic sonar for indoor navigation, designed for visually impaired users.
Inspired by bat CF-FM echolocation. Built for the **Global AI Hackathon Series
with Qwen Cloud** — **Track: MemoryAgent**, persistent adaptive memory across
multi-turn, cross-session interactions.

---

## Why this exists

There are millions of people who navigate the world without sight. A white
cane tells you what's at arm's length. A guide dog tells you what's ahead —
if you're lucky enough to have one, afford one, and live somewhere that
trains them. Almost nobody has real-time knowledge of open space, walls,
doorways, corners, and corridors *before* they get there.

Bats solved this problem 50 million years ago. They emit a chirp, listen to
the echo, and build a picture of the world from sound alone — in real time,
while flying, in total darkness. REBOUND borrows that idea and puts it in a
phone: emit a chirp, capture the echo, run it through signal processing and
a neural classifier, and tell the person what's around them — out loud, in
their language, before they touch it with a cane.

This is not a finished product. It's a hackathon build, made in days, not
years. But the idea underneath it — that echolocation plus an adaptive
memory agent could give a blind person a few extra seconds of certainty in
an unfamiliar space — is real, and worth building further.

---

## What it actually does

1. Emits a CF-FM chirp (8 kHz constant tone + downward sweep) through the
   phone's speaker.
2. Captures the returning echo through the microphone, in a latency-aware
   window that survives Android's variable output delay.
3. Deconvolves the echo into a Room Impulse Response (adaptive Wiener
   filter).
4. Extracts acoustic features — mel spectrogram (64×32), MFCCs (13×32),
   RT60, spectral centroid.
5. Classifies the space with a lightweight PyTorch CNN: **open space,
   nearby wall, doorway, corner, corridor, or stairs.**
6. Runs an independent two-pass staircase detector as DSP reinforcement —
   stairs are a different kind of danger than a wall, and don't get to rely
   on the CNN alone.
7. A **Qwen-powered Memory Agent** takes that classification, remembers
   what it has learned about this user across sessions, and generates a
   spoken navigation instruction personalized to how *this* person
   hesitates, advances, and reacts.
8. Adaptive Bayesian priors mean the system gets quieter about things the
   user has already proven they handle well, and more careful about the
   things that have tripped them up before.
9. Feedback reaches the user through **four independent channels** —
   earcons (rhythm and melodic contour, not just pitch), synthesized
   speech, vibration, and ARIA live regions for VoiceOver/TalkBack — because
   a navigation aid that only works when the screen is visible has already
   failed its user.

---

## Architecture

![REBOUND system architecture](visual/3/architecture_dark.png)

```
  Mobile (PWA)                        Alibaba Cloud ECS
 ┌──────────────┐                   ┌──────────────────────────────────┐
 │ Web Audio API │── audio buffer ──▶│ POST /predict                    │
 │ getUserMedia  │  (f32) + sr       │   ├─ Wiener deconvolution (RIR) │
 │ speaker emit  │                   │   ├─ Feature extraction          │
 │               │                   │   ├─ CNN classifier (6 classes)  │
 │ DeviceOrienta │── gyro pitch ────▶│   ├─ Stair detector (DSP)       │
 │ tionEvent     │                   │   └─ predict() result            │
 │               │◀── JSON ─────────│                                  │
 │ earcons ·     │   instruction     │ POST /process                    │
 │ speech ·      │   distance_m      │   ├─ Qwen-Plus Memory Agent     │
 │ haptics ·     │   class_name      │   ├─ Episodic/Semantic memory   │
 │ ARIA live     │   stair_message   │   └─ User profile update        │
 └──────────────┘                   └──────────────────────────────────┘
                                              │
                                     DashScope API (Qwen-Plus)
```

One scan end to end:

![Signal pipeline](visual/3/pipeline_dark.png)

---

## The Alibaba Cloud part — and why it mattered more than we expected

REBOUND started deployed in Singapore. It worked. It also had almost a full
second of round-trip latency for every scan — noticeable, and for a
navigation tool where "noticeable" can mean *"walked into it before the
warning arrived,"* that latency wasn't a footnote. It was the whole problem.

We didn't guess at the cause. We measured it: a plain `curl` to `/health` —
an endpoint that does *nothing* but return a JSON blob, no model, no Qwen
call — was already taking over a second round-trip. That ruled out the LLM
as the bottleneck and pointed straight at geography. So we did the
unglamorous thing: stood up a second ECS instance from scratch in
**US-Virginia**, migrated the whole stack — Docker image, model checkpoint,
Qwen credentials, Caddy reverse proxy with a real Let's Encrypt certificate
via DuckDNS — and pointed the domain at it.

The latency dropped hard. Not because the code got smarter, but because we
stopped assuming "cloud" means "close enough," and went and checked.

Along the way we also found and fixed a second, subtler problem: `/predict`
— the single most CPU-expensive endpoint in the system — was running its
signal-processing pipeline synchronously *inside* the async event loop. Two
people scanning at once weren't just competing for CPU; they were fully
blocking each other. We moved the pipeline to a thread pool behind a bounded
semaphore, so it now degrades gracefully (`429`, retry) instead of
serializing invisibly. That fix mattered more for concurrent judges testing
the live demo than any latency the region migration alone could have solved.

We treat Alibaba Cloud as more than a place to point a Docker container.
Real deployment means real hardening, and it means being honest when
something is slow instead of hand-waving it as "cloud latency, nothing we
can do."

---

## Qwen Cloud integration

The Memory Agent (`src/memory/agent.py`) uses **Qwen-Plus** via the
DashScope API, fully asynchronously, to:
- Reason over sonar observations in natural language
- Decide which memories to store, update, or forget (selective forgetting)
- Generate personalized navigation instructions per user
- Adjust Bayesian confidence priors based on user behavior — advancing,
  hesitating, retreating

The backend (`src/cloud/api_server.py`) is deployed on **Alibaba Cloud ECS**
via Docker, exposing endpoints for audio prediction, observation
processing, user profiles, and session management, with every LLM call
guarded against hallucination: multipliers clamped to `[0.1, 10.0]`, values
size-capped, class names checked against a whitelist before they ever reach
a prompt.

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

All figures above are measured on **simulated** rooms — see
[Known limitations](#known-limitations--said-plainly-not-buried). Full
empirical detail: [LIMITATIONS.md](LIMITATIONS.md).

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict` | Raw audio in → classification + distance + stair detection |
| `POST` | `/process` | Classification result → Memory Agent → navigation instruction |
| `GET` | `/profile/{user_id}` | User profile, episodic stats, semantic memory |
| `POST` | `/session` | Start new session for user |
| `GET` | `/health` | Health check |

All endpoints except `/health` require an `X-API-Token` header when
`REBOUND_API_TOKEN` is set (see [Security](#security)).

---

## Known limitations — said plainly, not buried

- **No spatial direction.** REBOUND can tell you there's a wall a meter
  away. It cannot yet tell you if that wall is to your left, your right, or
  straight ahead. That needs a stereo microphone array to triangulate
  direction — a hardware constraint, not a software one. A single phone
  microphone can't resolve it.
- **The training dataset is synthetic**, generated from acoustic
  simulation, not from real rooms with real people walking through them. It
  works well enough to demo convincingly. It is not yet trained on the
  messiness of real-world echoes, real hallways, real furniture.
- **It still needs a phone in your hand.** For someone who already
  navigates with a cane and possibly a guide dog, adding "also hold a phone
  up and tap it" is a real usability cost we haven't solved yet.

None of these are hidden from the judges because none of them are
embarrassing. They're the honest next steps for a project built in days,
not the finish line for a project that's done. Full empirical constraints,
measured rather than assumed: [LIMITATIONS.md](LIMITATIONS.md). The
adversarial security review we ran against our own system, findings and
fixes both: [AUDITORIA_ADVERSARIAL.md](AUDITORIA_ADVERSARIAL.md) and
[CHANGELOG_2026-07-01.md](CHANGELOG_2026-07-01.md).

---

## What's next

The most exciting fix for all three limitations above is the same one: **a
dedicated IoT device.** A small wearable with a stereo microphone pair
(solving spatial direction), an onboard low-power inference chip (cutting
round-trip latency further), and no phone required at all (solving the
"hands are busy" problem cane users already live with). Once the dataset
moves from synthetic to real recordings — ideally collected in partnership
with actual blind or low-vision users — the CNN classifier's confidence and
the Memory Agent's personalization both improve with it, because the whole
system is only as honest as the data it learned from.

- **Real-world dataset** — field RIR collection to close the sim-to-real
  gap documented above.
- **On-device personalization** — Elastic Weight Consolidation fine-tuning
  per user (`future/ewc.py`, already prototyped, deferred to v2).
- **Edge inference** — 296K parameters is small enough to run offline when
  there's no signal.

---

## Installation

```bash
pip install -r requirements.txt
```

Set your DashScope API key:
```bash
export DASHSCOPE_API_KEY=your_key_here
```

Recommended for production — a shared demo token (see
[Security](#security)):
```bash
export REBOUND_API_TOKEN=$(openssl rand -hex 16)
export REBOUND_ALLOWED_ORIGINS=https://rebound-olga.duckdns.org
```

CUDA-capable GPU required for training. CPU is sufficient for inference and
for the live demo.

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

## Repository structure

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
frontend/      React PWA (Vite) — sonar engine, earcons, speech, live view
future/
  ewc.py             Elastic Weight Consolidation — deferred to v2
  augmentation.py    noise augmentation (known bug, see header) — deferred to v2
tests/                     88 unit + integration tests
LIMITATIONS.md             empirically measured system constraints
AUDITORIA_ADVERSARIAL.md   adversarial security audit (Peircean abductive method)
CHANGELOG_2026-07-01.md    bug remediation changelog
```

---

## Security

REBOUND handles behavioral data about people with disabilities, so we hold
it to a real security bar, not a demo-day shrug:

- **Token authentication** on every data endpoint (`X-API-Token`,
  constant-time comparison) — the server logs loudly on startup if no
  token is configured, so "silently open" is never silent.
- **CORS scoped** to the deployed frontend origin — no wildcard +
  credentials in production.
- **Payload size caps and bounded concurrency** on `/predict`: oversized
  audio is rejected before decoding, and excess concurrent requests get a
  clean `429` instead of degrading every request in flight.
- **LRU eviction with a hard cap** on per-user state, so no amount of
  invented `user_id`s can grow server memory without bound.
- `user_id` validated against `[a-zA-Z0-9_\-]{1,64}` (path traversal
  prevention).
- `class_name` validated against a fixed whitelist before reaching any
  prompt (prompt-injection prevention).
- LLM-generated multipliers clamped to `[0.1, 10.0]` (hallucination
  containment); semantic memory values capped at 512 characters (unbounded
  growth prevention).
- Per-user `asyncio.Lock` (race-condition prevention) with heavy DSP/CNN
  work off the event loop, so concurrent users no longer block each other.
- `.dockerignore` excludes `data/profiles/` and `data/checkpoints/`.
- Deployed over HTTPS via Caddy with a real Let's Encrypt certificate —
  not self-signed, because iOS and Android both refuse microphone access
  otherwise.

We ran an adversarial audit against our own system before polishing the
demo and published every finding: see
[AUDITORIA_ADVERSARIAL.md](AUDITORIA_ADVERSARIAL.md).

---

## Built with

Python, FastAPI, PyTorch, librosa, Qwen Cloud (qwen-plus), Docker, Alibaba
Cloud ECS, Caddy, DuckDNS, React (Web Audio API, PWA)

---

## Team

Designed, architected, and developed in full by **Olga Vasilieva**, with
infrastructure and deployment support from the **VIGÍA AI Collective**.

The GitHub repository is hosted under the account of Anna Tchijova (her
daughter) for technical convenience. Anna Tchijova had no participation in
the code, design, or technical decisions of this project.

**License:** Apache License 2.0 — see [LICENSE](LICENSE)

---

*Qwen Cloud Hackathon · MemoryAgent Track*

*Every phone is a sonar.*

---

## Slides — Pitch

![Slide 1](visual/1/REBOUND_pitch-01.png)
![Slide 2](visual/1/REBOUND_pitch-02.png)
![Slide 3](visual/1/REBOUND_pitch-03.png)
![Slide 4](visual/1/REBOUND_pitch-04.png)
![Slide 5](visual/1/REBOUND_pitch-05.png)
![Slide 6](visual/1/REBOUND_pitch-06.png)
![Slide 7](visual/1/REBOUND_pitch-07.png)
![Slide 8](visual/1/REBOUND_pitch-08.png)
![Slide 9](visual/1/REBOUND_pitch-09.png)
![Slide 10](visual/1/REBOUND_pitch-10.png)
![Slide 11](visual/1/REBOUND_pitch-11.png)

---

## Slides — Biomimetic Sonar Navigation System

![Slide 1](visual/2/REBOUND_sonar-01.png)
![Slide 2](visual/2/REBOUND_sonar-02.png)
![Slide 3](visual/2/REBOUND_sonar-03.png)
![Slide 4](visual/2/REBOUND_sonar-04.png)
![Slide 5](visual/2/REBOUND_sonar-05.png)
![Slide 6](visual/2/REBOUND_sonar-06.png)
![Slide 7](visual/2/REBOUND_sonar-07.png)
![Slide 8](visual/2/REBOUND_sonar-08.png)
![Slide 9](visual/2/REBOUND_sonar-09.png)
![Slide 10](visual/2/REBOUND_sonar-10.png)
![Slide 11](visual/2/REBOUND_sonar-11.png)
![Slide 12](visual/2/REBOUND_sonar-12.png)
![Slide 13](visual/2/REBOUND_sonar-13.png)
![Slide 14](visual/2/REBOUND_sonar-14.png)
![Slide 15](visual/2/REBOUND_sonar-15.png)
![Slide 16](visual/2/REBOUND_sonar-16.png)
