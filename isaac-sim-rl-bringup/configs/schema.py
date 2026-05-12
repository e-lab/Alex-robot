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
    """Modes:
      - manual:    keyboard only, FSM disabled (default).
      - approach:  SAM3-driven FSM that walks toward `target` (Phase 2+).
      - fixed_xyz: FSM with hardcoded goal (Phase 1 acceptance — locomotion only).
    """
    mode: str = "manual"               # manual | approach | fixed_xyz

    # --- Goal source -----------------------------------------------------------
    target: Optional[str] = None                       # used when mode=approach
    fixed_xyz: Tuple[float, float, float] = (3.0, 0.0, 0.0)  # used when mode=fixed_xyz

    # --- Geometry --------------------------------------------------------------
    stop_dist: float = 1.0

    # --- Velocities (Alex gait limits — tighter than the cam robot's) ----------
    walk_speed: float = 0.30           # m/s   (cam used 0.4 — too aggressive)
    search_yaw: float = 0.30           # rad/s (cam used 0.6 — exceeded gait limit)
    yaw_max:    float = 0.40           # rad/s hard cap on APPROACH yaw_rate
    heading_kp: float = 0.8            # rad/s per rad (cam used 1.5)
    heading_walk_deg: float = 30.0     # only walk forward when |err|< this many deg

    # --- Goal state machine ---------------------------------------------------
    stale_s:    float = 5.0
    lock_conf:  float = 0.6
    # Phase-2 lock-on: require this many distinct SAM3 sightings of the same
    # object before allowing the goal to latch. Mitigates single-frame mis-
    # projections (e.g. mask leaking onto the wall behind a glass oven door).
    min_observations: int = 3
    tilt_deg:   float = 20.0
    tilt_period_s: float = 10.0

    # --- Obstacle avoidance (Phase 3) -----------------------------------------
    obstacle_stop_dist: float = 1.5
    obstacle_cone_h_deg: float = 20.0
    obstacle_cone_v_deg: float = 10.0

    # --- Recovery / fall detection (Phase 1 stub, Phase 4 full) ---------------
    fall_height_m:  float = 0.5
    fall_tilt_norm: float = 0.7
    stuck_window_s: float = 5.0
    stuck_dist_m:   float = 0.2
    # Phase-4 recovery agent tunables.
    recovery_stand_s:      float = 3.0   # standing-flag hold per attempt
    recovery_max_attempts: int   = 2     # hard giveup → autonomy disabled
    recovery_rotation_yaw: float = 0.4   # rad/s during 90° unstuck rotation


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
class LocoXConfig:
    """LA-6: Loco-X agent integration. ``enabled=False`` (default)
    means Phase 1-4 behaviour is unchanged. When enabled, the
    autonomy script builds an AsyncRunner + TaskDispatcher and the
    LLM agent drives goto / face / stop / peek / survey skills."""
    enabled: bool = False
    tick_hz: float = 2.0
    max_turns: int = 20
    exec_timeout_s: float = 5.0


@dataclass
class AlexAppConfig:
    """Top-level config for alex_onnx_walking_policy.py (the walking script)."""
    repo_root: str = "???"
    scene:     SceneConfig    = field(default_factory=SceneConfig)
    policy:    PolicyConfig   = field(default_factory=PolicyConfig)
    detector:  DetectorConfig = field(default_factory=DetectorConfig)   # SAM3
    yolo:      YoloConfig     = field(default_factory=YoloConfig)       # YOLO (mutex w/ SAM3)
    rerun:     RerunConfig    = field(default_factory=RerunConfig)
    autonomy:  AutonomyConfig = field(default_factory=AutonomyConfig)
    output:    OutputConfig   = field(default_factory=OutputConfig)
    loco_x:    LocoXConfig    = field(default_factory=LocoXConfig)


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
    cs.store(group="loco_x",   name="base_loco_x",   node=LocoXConfig)
