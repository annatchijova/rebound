"""
Room Impulse Response (RIR) generation with pyroomacoustics.

Simulates 5 classes of indoor spaces for classifier training:
- open_space: large room, no nearby obstacles
- nearby_wall: obstacle < 1.5 m in emission direction
- doorway: narrow opening with characteristic spectral signature
- corner: multiple reflections at angles
- corridor: parallel lateral reflections

Stairs are detected separately via detect_stair_periodicity (src/signal/stairs.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from src.signal.chirp import CHIRP_PARAMS

SPACE_CLASSES = {
    0: "open_space",
    1: "nearby_wall",
    2: "doorway",
    3: "corner",
    4: "corridor",
    5: "stairs",
}

CLASS_NAMES_TO_ID = {v: k for k, v in SPACE_CLASSES.items()}


@dataclass
class RoomConfig:
    """Room configuration for simulation."""
    room_dim: list[float]         # [x, y, z] in meters
    source_pos: list[float]       # emitter position [x, y, z]
    mic_pos: list[float]          # receiver position [x, y, z]
    materials: dict[str, float]   # absorption per wall (0=reflective, 1=absorptive)
    class_id: int
    class_name: str
    distance_m: float             # distance to nearest obstacle


def generate_rir(
    config: RoomConfig,
    sample_rate: int | None = None,
    max_order: int = 10,
    noise_level: float = 0.0,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Generate a RIR using pyroomacoustics or synthetic stair model.

    For class_id=5 (stairs), uses synthesize_stair_rir from the stairs
    module instead of pyroomacoustics ShoeBox.

    Args:
        config: room configuration
        sample_rate: sampling rate in Hz
        max_order: maximum reflection order (higher = more reverberation)
        noise_level: Gaussian noise level to add (0 = clean)
        seed: RNG seed for noise

    Returns:
        RIR — shape: (n_samples,) — float64
    """
    sr = sample_rate or CHIRP_PARAMS["sample_rate"]

    if config.class_id == 5:
        from src.signal.stairs import synthesize_stair_rir
        rir = synthesize_stair_rir(
            n_steps=int(config.materials.get("n_steps", 8)),
            tread_m=config.materials.get("tread_m", 0.29),
            attenuation_per_step=config.materials.get("attenuation", 0.85),
            sample_rate=sr,
            base_distance_m=config.materials.get("base_distance_m", 0.5),
            rir_length_s=0.15,
        )
        if noise_level > 0:
            rng = np.random.default_rng(seed)
            rir = rir + rng.standard_normal(len(rir)) * noise_level
        return rir

    import pyroomacoustics as pra

    abs_coeff = config.materials.get("absorption", 0.3)
    materials = pra.Material(abs_coeff)

    room = pra.ShoeBox(
        config.room_dim,
        fs=sr,
        materials=materials,
        max_order=max_order,
    )

    room.add_source(config.source_pos)
    mic_array = np.array([config.mic_pos]).T  # shape: (3, 1)
    room.add_microphone_array(mic_array)

    room.compute_rir()

    rir = room.rir[0][0]  # first mic, first source
    # rir: (n_samples,) — float64
    return rir


def random_open_space(rng: np.random.Generator) -> RoomConfig:
    """Generate config for open space (large room)."""
    width = rng.uniform(6.0, 15.0)
    depth = rng.uniform(6.0, 12.0)
    height = rng.uniform(2.5, 4.0)

    # User in center, far from walls
    src_x = width / 2 + rng.uniform(-1.0, 1.0)
    src_y = depth / 2 + rng.uniform(-1.0, 1.0)

    # Mic co-located with source (in practice)
    mic_x = src_x + rng.uniform(-0.05, 0.05)
    mic_y = src_y + rng.uniform(-0.05, 0.05)

    distance = min(src_x, width - src_x, src_y, depth - src_y)

    return RoomConfig(
        room_dim=[width, depth, height],
        source_pos=[src_x, src_y, 1.5],
        mic_pos=[mic_x, mic_y, 1.5],
        materials={"absorption": rng.uniform(0.2, 0.6)},
        class_id=0,
        class_name="open_space",
        distance_m=distance,
    )


def random_wall_nearby(rng: np.random.Generator) -> RoomConfig:
    """Generate config for nearby wall (< 1.5 m)."""
    width = rng.uniform(4.0, 10.0)
    depth = rng.uniform(4.0, 10.0)
    height = rng.uniform(2.5, 3.5)

    distance = rng.uniform(0.3, 1.5)
    wall = rng.choice(["north", "south", "east", "west"])

    if wall == "north":
        src_x = width / 2 + rng.uniform(-1.0, 1.0)
        src_y = depth - distance
    elif wall == "south":
        src_x = width / 2 + rng.uniform(-1.0, 1.0)
        src_y = distance
    elif wall == "east":
        src_x = width - distance
        src_y = depth / 2 + rng.uniform(-1.0, 1.0)
    else:
        src_x = distance
        src_y = depth / 2 + rng.uniform(-1.0, 1.0)

    mic_x = src_x + rng.uniform(-0.05, 0.05)
    mic_y = src_y + rng.uniform(-0.05, 0.05)

    return RoomConfig(
        room_dim=[width, depth, height],
        source_pos=[src_x, src_y, 1.5],
        mic_pos=[mic_x, mic_y, 1.5],
        materials={"absorption": rng.uniform(0.1, 0.5)},
        class_id=1,
        class_name="nearby_wall",
        distance_m=distance,
    )


def random_doorway(rng: np.random.Generator) -> RoomConfig:
    """Generate config for doorway.

    Simulated as a narrow room where the user stands in the opening.
    The spectral signature comes from close reflections off the frame.
    distance_m = distance to the far wall (forward direction), consistent
    with all other classes.
    """
    width = rng.uniform(0.8, 1.2)   # door width
    depth = rng.uniform(3.0, 6.0)
    height = rng.uniform(2.2, 2.8)

    src_x = width / 2
    src_y = rng.uniform(0.3, 0.8)
    # Forward distance: to the far wall through the opening
    distance = depth - src_y

    mic_x = src_x + rng.uniform(-0.03, 0.03)
    mic_y = src_y + rng.uniform(-0.03, 0.03)

    return RoomConfig(
        room_dim=[width, depth, height],
        source_pos=[src_x, src_y, 1.5],
        mic_pos=[mic_x, mic_y, 1.5],
        materials={"absorption": rng.uniform(0.1, 0.3)},
        class_id=2,
        class_name="doorway",
        distance_m=distance,
    )


def random_corner(rng: np.random.Generator) -> RoomConfig:
    """Generate config for corner (two nearby walls at angle)."""
    width = rng.uniform(4.0, 8.0)
    depth = rng.uniform(4.0, 8.0)
    height = rng.uniform(2.5, 3.5)

    corner = rng.choice(["sw", "se", "nw", "ne"])
    dist_x = rng.uniform(0.3, 1.5)
    dist_y = rng.uniform(0.3, 1.5)

    if corner == "sw":
        src_x, src_y = dist_x, dist_y
    elif corner == "se":
        src_x, src_y = width - dist_x, dist_y
    elif corner == "nw":
        src_x, src_y = dist_x, depth - dist_y
    else:
        src_x, src_y = width - dist_x, depth - dist_y

    mic_x = src_x + rng.uniform(-0.05, 0.05)
    mic_y = src_y + rng.uniform(-0.05, 0.05)

    distance = min(dist_x, dist_y)

    return RoomConfig(
        room_dim=[width, depth, height],
        source_pos=[src_x, src_y, 1.5],
        mic_pos=[mic_x, mic_y, 1.5],
        materials={"absorption": rng.uniform(0.1, 0.4)},
        class_id=3,
        class_name="corner",
        distance_m=distance,
    )


def random_corridor(rng: np.random.Generator) -> RoomConfig:
    """Generate config for corridor (parallel lateral walls)."""
    width = rng.uniform(1.0, 2.0)    # narrow
    depth = rng.uniform(5.0, 20.0)   # long
    height = rng.uniform(2.5, 3.0)

    src_x = width / 2 + rng.uniform(-0.2, 0.2)
    src_y = rng.uniform(2.0, depth - 2.0)

    mic_x = src_x + rng.uniform(-0.03, 0.03)
    mic_y = src_y + rng.uniform(-0.03, 0.03)

    distance = min(src_x, width - src_x)

    return RoomConfig(
        room_dim=[width, depth, height],
        source_pos=[src_x, src_y, 1.5],
        mic_pos=[mic_x, mic_y, 1.5],
        materials={"absorption": rng.uniform(0.1, 0.3)},
        class_id=4,
        class_name="corridor",
        distance_m=distance,
    )


def random_stairs(rng: np.random.Generator) -> RoomConfig:
    """Generate config for staircase.

    Uses synthetic stair RIR parameters. The actual RIR is generated
    by generate_rir_or_stairs() which dispatches to synthesize_stair_rir
    for class_id=5.
    """
    n_steps = rng.integers(5, 15)
    tread_m = rng.uniform(0.27, 0.31)  # within standard tolerance
    base_distance = rng.uniform(0.3, 1.5)
    distance = base_distance  # distance to first step

    # Room dimensions are placeholders — RIR comes from synthesize_stair_rir
    return RoomConfig(
        room_dim=[2.0, float(n_steps) * tread_m + 2.0, 3.0],
        source_pos=[1.0, 0.5, 1.5],
        mic_pos=[1.0, 0.5, 1.5],
        materials={
            "absorption": rng.uniform(0.1, 0.4),
            "n_steps": float(n_steps),
            "tread_m": tread_m,
            "base_distance_m": base_distance,
            "attenuation": rng.uniform(0.75, 0.92),
        },
        class_id=5,
        class_name="stairs",
        distance_m=distance,
    )


# Generator map by class
GENERATORS = {
    0: random_open_space,
    1: random_wall_nearby,
    2: random_doorway,
    3: random_corner,
    4: random_corridor,
    5: random_stairs,
}


def generate_dataset_configs(
    n_per_class: int = 500,
    seed: int = 42,
) -> list[RoomConfig]:
    """Generate room configurations for the full dataset.

    Args:
        n_per_class: configurations per class
        seed: seed for reproducibility

    Returns:
        List of RoomConfig, n_per_class * 6 elements
    """
    rng = np.random.default_rng(seed)
    configs: list[RoomConfig] = []

    for class_id, generator in GENERATORS.items():
        for _ in range(n_per_class):
            configs.append(generator(rng))

    return configs
