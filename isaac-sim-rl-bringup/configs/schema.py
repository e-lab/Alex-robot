"""Dataclass schemas for cam_room_explore_isaac Hydra configs.

Registered with Hydra's ConfigStore so the yaml files can be validated against
typed groups and defaults can be set in one place. See Hydra's "Structured
Configs" docs for background.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple
from hydra.core.config_store import ConfigStore


# ── Scene ─────────────────────────────────────────────────────────────────────
@dataclass
class SceneConfig:
    """Which USD world to load."""
    name: str = "room"                 # room | hallway | groundplane
    usd_path: Optional[str] = None     # resolved relative to repo_root
    # Hallway-specific:
    doors: str = "open"                # open | closed


# ── Spawn ─────────────────────────────────────────────────────────────────────
@dataclass
class SpawnConfig:
    x: float = 0.0
    y: float = 0.0
    z: float = 1.5
    yaw_deg: float = 0.0


# ── Motion (keyboard sensitivities) ───────────────────────────────────────────
@dataclass
class MotionConfig:
    move_speed: float = 0.6            # m/s for translate keys
    turn_speed_deg: float = 60.0       # deg/s for yaw keys


# ── Detector (SAM3) ───────────────────────────────────────────────────────────
@dataclass
class DetectorConfig:
    enabled: bool = False
    # Comma-separated prompt list passed as one string (matches arg shape)
    prompts: str = ""
    conf: float = 0.3


# ── Rerun streaming ───────────────────────────────────────────────────────────
@dataclass
class RerunConfig:
    enabled: bool = False
    # Point cloud (optional, depth-unprojected to world)
    pointcloud: bool = False
    pointcloud_stride: int = 8
    pointcloud_max_depth: float = 8.0


# ── Autonomy ──────────────────────────────────────────────────────────────────
@dataclass
class AutonomyConfig:
    """Either 'manual' (keyboard only) or 'approach' (FSM that walks toward target)."""
    mode: str = "manual"               # manual | approach
    # When mode=approach:
    target: Optional[str] = None       # prompt label to search for
    stop_dist: float = 1.0
    walk_speed: float = 0.4
    search_yaw: float = 0.6
    heading_kp: float = 1.5
    stale_s: float = 5.0
    lock_conf: float = 0.6
    tilt_deg: float = 20.0
    tilt_period_s: float = 10.0


# ── Output / scene graph save path ────────────────────────────────────────────
@dataclass
class OutputConfig:
    scene_graph_path: str = "${repo_root}/isaac-sim-rl-bringup/scene_graph.json"
    save_every_s: float = 30.0


# ── Policy (Alex walking ONNX) ───────────────────────────────────────────────
@dataclass
class PolicyConfig:
    """Walking ONNX policy file + initial velocity command."""
    onnx_path: str = (
        "${repo_root}/isaac-sim-rl-bringup/models/"
        "2026-03-17_23-20-27_flatfeet/policy.onnx"
    )
    vx: float = 0.3                    # initial forward velocity (m/s)
    vy: float = 0.0                    # initial lateral velocity (m/s)
    yaw: float = 0.0                   # initial yaw rate (rad/s)
    standing: bool = False             # start in standing mode (standing_flag=1)


# ── Perception (Alex script variant with YOLO + SAM3 + head camera) ──────────
@dataclass
class YoloConfig:
    """YOLO ONNX detector on head camera. Alternative to SAM3."""
    enabled: bool = False
    weights: str = (
        "/home/sravani/nadia/repository-group/ihmc-open-robotics-software/"
        "ihmc-perception/src/main/resources/yolo/best_multi_02_17_2026/"
        "best_multi_02_17_2026.onnx"
    )
    conf: float = 0.35


@dataclass
class AlexAppConfig:
    """Top-level config for alex_onnx_walking_policy.py (the walking script)."""
    repo_root: str = "???"
    scene:     SceneConfig    = field(default_factory=SceneConfig)
    policy:    PolicyConfig   = field(default_factory=PolicyConfig)
    detector:  DetectorConfig = field(default_factory=DetectorConfig)   # SAM3
    yolo:      YoloConfig     = field(default_factory=YoloConfig)       # YOLO (mutex w/ SAM3)
    rerun:     RerunConfig    = field(default_factory=RerunConfig)


# ── Root (cam_room_explore) ──────────────────────────────────────────────────
@dataclass
class Config:
    repo_root: str = "???"             # set in root.yaml, used for path interpolation
    scene:     SceneConfig = field(default_factory=SceneConfig)
    spawn:     SpawnConfig = field(default_factory=SpawnConfig)
    motion:    MotionConfig = field(default_factory=MotionConfig)
    detector:  DetectorConfig = field(default_factory=DetectorConfig)
    rerun:     RerunConfig = field(default_factory=RerunConfig)
    autonomy:  AutonomyConfig = field(default_factory=AutonomyConfig)
    output:    OutputConfig = field(default_factory=OutputConfig)


def register() -> None:
    """Register schemas with Hydra's ConfigStore. Call before @hydra.main."""
    cs = ConfigStore.instance()
    cs.store(name="base_config",   node=Config)
    cs.store(name="base_alex_app", node=AlexAppConfig)
    cs.store(group="scene",    name="base_scene",    node=SceneConfig)
    cs.store(group="detector", name="base_detector", node=DetectorConfig)
    cs.store(group="rerun",    name="base_rerun",    node=RerunConfig)
    cs.store(group="autonomy", name="base_autonomy", node=AutonomyConfig)
    cs.store(group="policy",   name="base_policy",   node=PolicyConfig)
    cs.store(group="yolo",     name="base_yolo",     node=YoloConfig)
