"""
REBOUND adaptive system demo.

Simulates a complete navigation session:
1. Generate environments with pyroomacoustics
2. Process chirp -> echo -> RIR -> features
3. Classify with CNN (or mock if no trained model)
4. Memory Agent (Qwen or mock) decides instructions and memory operations
5. Visualize user profile evolution in real time

Usage:
    python3 -m src.demo.adaptive_demo
    python3 -m src.demo.adaptive_demo --use-qwen  # with real API
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import matplotlib
matplotlib.use(os.environ.get("MPLBACKEND", "TkAgg"))
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.feedback.haptic import get_pattern, pattern_to_waveform
from src.features.geometric import estimate_distance, estimate_echo_strength
from src.features.spectral import compute_mel_spectrogram, compute_rt60, compute_spectral_centroid
from src.memory.agent import MockQwenMemoryAgent, QwenMemoryAgent
from src.memory.episodic import Episode, EpisodicMemory
from src.memory.profile import UserProfile
from src.memory.semantic import SemanticMemory
from src.signal.capture import simulate_capture
from src.signal.chirp import CHIRP_PARAMS, generate_chirp
from src.signal.deconvolution import adaptive_wiener, estimate_snr
from src.signal.stairs import detect_stair_periodicity, estimate_stair_geometry, build_stair_message, synthesize_stair_rir
from src.simulation.room_generator import (
    GENERATORS,
    SPACE_CLASSES,
    generate_rir,
)


# Simulated navigation sequence (class_id, user_action)
NAVIGATION_SEQUENCE = [
    (0, "advance"),      # open space — advances
    (0, "advance"),      # open space — continues
    (4, "advance"),      # corridor — advances
    (4, "advance"),      # corridor — continues
    (1, "hesitate"),     # nearby wall — hesitates
    (1, "advance"),      # nearby wall — decides to advance
    (2, "hesitate"),     # doorway — hesitates
    (2, "advance"),      # doorway — crosses
    (3, "retreat"),      # corner — retreats
    (3, "hesitate"),     # corner — hesitates
    (5, "hesitate"),     # stairs — hesitates (new environment)
    (5, "advance"),      # stairs — advances carefully
    (4, "advance"),      # corridor — advances
    (1, "hesitate"),     # nearby wall — hesitates
    (2, "advance"),      # doorway — crosses (now with confidence)
    (0, "advance"),      # open space — advances
    (3, "advance"),      # corner — now advances (learned)
    (5, "advance"),      # stairs — confident now
    (1, "advance"),      # nearby wall — no longer hesitates
    (2, "advance"),      # doorway — confident
    (4, "advance"),      # corridor
    (3, "advance"),      # corner — fully adapted
    (0, "advance"),      # open space
]


def run_demo(use_qwen: bool = False, pause_s: float = 1.5) -> None:
    """Run the complete demo with visualization."""
    rng = np.random.default_rng(42)
    profile = UserProfile(user_id="demo_user")
    profile.start_session()
    episodic = EpisodicMemory()
    semantic = SemanticMemory()

    if use_qwen:
        agent = QwenMemoryAgent()
    else:
        agent = MockQwenMemoryAgent(api_key="mock")

    chirp = generate_chirp()
    sr = CHIRP_PARAMS["sample_rate"]

    # Visualization setup
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("REBOUND — Biomimetic Sonar Navigation System", fontsize=14, fontweight="bold")
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

    ax_waveform = fig.add_subplot(gs[0, 0])
    ax_spectrogram = fig.add_subplot(gs[0, 1])
    ax_classification = fig.add_subplot(gs[0, 2])
    ax_profile = fig.add_subplot(gs[1, :2])
    ax_distance = fig.add_subplot(gs[1, 2])
    ax_haptic = fig.add_subplot(gs[2, 0])
    ax_memory = fig.add_subplot(gs[2, 1:])

    plt.ion()
    plt.show()

    weight_history: list[list[float]] = []
    class_names = [SPACE_CLASSES[i] for i in range(6)]
    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#795548"]

    for step, (class_id, action) in enumerate(NAVIGATION_SEQUENCE):
        # 1. Generate environment
        config = GENERATORS[class_id](rng)

        # 2. Simulate capture
        if class_id == 5:
            # For stairs, use synthetic stair RIR
            n_steps = rng.integers(5, 12)
            rir = synthesize_stair_rir(n_steps=n_steps, sample_rate=sr)
        else:
            rir = generate_rir(config, sample_rate=sr, max_order=8)

        captured = simulate_capture(rir, chirp, noise_level=0.005, seed=step)

        # 3. Extract RIR via deconvolution
        snr = estimate_snr(captured)
        estimated_rir = adaptive_wiener(captured, chirp, snr_estimate_db=max(snr, 5))

        # 4. Features
        mel = compute_mel_spectrogram(estimated_rir, sr)
        rt60 = compute_rt60(estimated_rir, sr)
        centroid = compute_spectral_centroid(estimated_rir, sr)
        dist = estimate_distance(estimated_rir, sr)
        echo_str = estimate_echo_strength(estimated_rir, sr)

        # 5. Stair-specific analysis
        stair_info = None
        if class_id == 5:
            stair_result = detect_stair_periodicity(estimated_rir, sr)
            if stair_result["is_stair"]:
                stair_info = estimate_stair_geometry(
                    stair_result["echo_spacing_m"],
                    stair_result["n_steps_detected"],
                )

        # 6. Prediction (mock — in production would be CNN)
        confidence = 0.7 + rng.uniform(0, 0.25)
        prediction = {
            "class": SPACE_CLASSES[class_id],
            "confidence": float(confidence),
            "distance_m": float(dist) if dist > 0 else config.distance_m,
        }

        features_summary = {
            "rt60": float(rt60),
            "spectral_centroid": float(centroid),
            "echo_strength": float(echo_str),
        }

        # 7. Memory Agent
        response = agent.process_observation(
            user_profile=profile,
            episodic=episodic,
            semantic=semantic,
            prediction=prediction,
            features_summary=features_summary,
            user_action=action,
            session_id=1,
        )

        # Override instruction with stair message if applicable
        if stair_info:
            response.navigation_instruction = build_stair_message(stair_info, "undetermined")

        # 8. Apply memory operations (includes profile weight adjustments)
        agent.apply_memory_ops(response, profile, episodic, semantic, current_session=1)

        # 9. Store episode
        episodic.store(Episode(
            timestamp=time.time(),
            session_id=1,
            prediction_class=SPACE_CLASSES[class_id],
            prediction_confidence=confidence,
            distance_m=prediction["distance_m"],
            user_action=action,
            features_summary=features_summary,
        ))

        # Periodic semantic consolidation
        if step % 5 == 4:
            semantic.consolidate_from_episodic(episodic, min_observations=3)

        weight_history.append(list(profile.class_weights))

        # === VISUALIZATION ===
        _update_plots(
            fig, step, action,
            ax_waveform, ax_spectrogram, ax_classification,
            ax_profile, ax_distance, ax_haptic, ax_memory,
            chirp, captured, mel, sr,
            prediction, response, profile,
            weight_history, class_names, colors,
            episodic, semantic,
        )

        plt.pause(pause_s)

    # Final consolidation
    semantic.consolidate_from_episodic(episodic, min_observations=3)

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print(f"\nFinal user profile:")
    print(f"  Interactions: {profile.total_interactions}")
    print(f"  Weights per class:")
    for i, name in SPACE_CLASSES.items():
        print(f"    {name}: {profile.class_weights[i]:.3f}")
    print(f"\nSemantic memory ({len(semantic.entries)} entries):")
    for key, entry in semantic.entries.items():
        print(f"    {key}: {entry.value} (conf={entry.confidence:.2f})")
    print(f"\nEpisodic memory: {len(episodic)} active episodes")

    plt.ioff()
    plt.show()
    agent.close()


def _update_plots(
    fig, step, action,
    ax_waveform, ax_spectrogram, ax_classification,
    ax_profile, ax_distance, ax_haptic, ax_memory,
    chirp, captured, mel, sr,
    prediction, response, profile,
    weight_history, class_names, colors,
    episodic, semantic,
):
    """Update all visualization panels."""

    # 1. Waveform
    ax_waveform.clear()
    t_chirp = np.arange(len(chirp)) / sr * 1000
    t_cap = np.arange(min(len(captured), len(chirp) * 3)) / sr * 1000
    ax_waveform.plot(t_chirp, chirp, alpha=0.7, label="Chirp", color="#2196F3")
    ax_waveform.plot(t_cap, captured[:len(t_cap)], alpha=0.5, label="Echo", color="#F44336")
    ax_waveform.set_title("Waveform", fontsize=10)
    ax_waveform.set_xlabel("ms")
    ax_waveform.legend(fontsize=7)
    ax_waveform.set_ylim(-1, 1)

    # 2. Spectrogram
    ax_spectrogram.clear()
    ax_spectrogram.imshow(mel, aspect="auto", origin="lower", cmap="magma")
    ax_spectrogram.set_title("Mel Spectrogram", fontsize=10)
    ax_spectrogram.set_xlabel("Frame")
    ax_spectrogram.set_ylabel("Mel bin")

    # 3. Classification
    ax_classification.clear()
    weights = profile.get_prior_weights()
    conf = prediction["confidence"]
    probs = np.zeros(6)
    probs[list(SPACE_CLASSES.keys())[list(SPACE_CLASSES.values()).index(prediction["class"])]] = conf
    adjusted = probs * weights
    if adjusted.sum() > 0:
        adjusted /= adjusted.sum()
    else:
        adjusted = probs

    bars = ax_classification.barh(class_names, adjusted, color=colors)
    ax_classification.set_title(f"Classification (step {step+1})", fontsize=10)
    ax_classification.set_xlim(0, 1)
    for bar, val in zip(bars, adjusted):
        if val > 0.05:
            ax_classification.text(val - 0.02, bar.get_y() + bar.get_height()/2,
                                   f"{val:.2f}", va="center", ha="right", fontsize=8, color="white")

    # 4. Profile evolution
    ax_profile.clear()
    if weight_history:
        arr = np.array(weight_history)
        for i in range(6):
            ax_profile.plot(arr[:, i], label=class_names[i], color=colors[i], linewidth=2)
        ax_profile.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax_profile.legend(fontsize=7, loc="upper left")
    ax_profile.set_title("User Profile Evolution (Bayesian Priors)", fontsize=10)
    ax_profile.set_xlabel("Interaction")
    ax_profile.set_ylabel("Weight")

    # 5. Distance
    ax_distance.clear()
    dist = prediction["distance_m"]
    ax_distance.text(0.5, 0.5, f"{dist:.1f}m", transform=ax_distance.transAxes,
                     fontsize=36, ha="center", va="center", fontweight="bold",
                     color="#F44336" if dist < 1.0 else "#4CAF50")
    ax_distance.text(0.5, 0.15, prediction["class"].replace("_", " "),
                     transform=ax_distance.transAxes, fontsize=12, ha="center", va="center")
    ax_distance.text(0.5, 0.85, f"action: {action}",
                     transform=ax_distance.transAxes, fontsize=10, ha="center", va="center",
                     color="#FF9800")
    ax_distance.set_title("Distance", fontsize=10)
    ax_distance.set_xlim(0, 1)
    ax_distance.set_ylim(0, 1)
    ax_distance.axis("off")

    # 6. Haptic
    ax_haptic.clear()
    pattern = get_pattern(response.haptic_pattern)
    waveform = pattern_to_waveform(pattern)
    ax_haptic.fill_between(range(len(waveform)), waveform, alpha=0.7, color="#9C27B0")
    ax_haptic.set_title(f"Haptic: {pattern.name}", fontsize=10)
    ax_haptic.set_ylim(0, 1)
    ax_haptic.set_ylabel("Intensity")

    # 7. Memory status
    ax_memory.clear()
    ax_memory.axis("off")
    memory_text = f"Navigation: {response.navigation_instruction}\n\n"
    memory_text += f"Episodic: {len(episodic)} events | "
    memory_text += f"Semantic: {len(semantic.entries)} entries\n\n"
    if response.memory_ops:
        memory_text += "Memory ops:\n"
        for op in response.memory_ops[:3]:
            memory_text += f"  - {op.get('op', '?')}: {op.get('value', op.get('key', ''))[:50]}\n"
    memory_text += f"\nReasoning: {response.reasoning[:80]}"
    ax_memory.text(0.02, 0.95, memory_text, transform=ax_memory.transAxes,
                   fontsize=9, va="top", fontfamily="monospace",
                   bbox=dict(boxstyle="round", facecolor="#f5f5f5", alpha=0.8))
    ax_memory.set_title("Memory Agent", fontsize=10)

    fig.canvas.draw_idle()
    fig.canvas.flush_events()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REBOUND Adaptive Demo")
    parser.add_argument("--use-qwen", action="store_true", help="Use real Qwen API")
    parser.add_argument("--pause", type=float, default=1.5, help="Pause between steps (seconds)")
    args = parser.parse_args()

    run_demo(use_qwen=args.use_qwen, pause_s=args.pause)
