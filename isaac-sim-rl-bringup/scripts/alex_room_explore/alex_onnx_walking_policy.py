"""
alex_onnx_walking_policy.py
----------------------------
Standalone Isaac Sim: run the ONNX walking policy in Python (no Java, no DDS).
Keyboard control lets you steer the robot in real time.

Run with:
    cd ~/pathtoFolder/IsaacLab
    ./isaaclab.sh -p ~/pathtoFolder/Alex-robot/isaac-sim-rl-bringup/scripts/alex_onnx_walking_policy.py

Keyboard controls (focus the Isaac Sim viewport first):
    Arrow Up    / Numpad 8  — walk forward
    Arrow Down  / Numpad 2  — walk backward
    Arrow Left  / Z         — turn left  (yaw +)
    Arrow Right / X         — turn right (yaw -)
    Q / Numpad 4            — strafe left
    E / Numpad 6            — strafe right
    L                       — stop / reset velocity to zero
    S                       — toggle standing mode (velocity = 0, standing_flag = 1)
    F                       — force a synthetic fall (test Phase 4 recovery)

CLI args (override defaults):
    --vx FLOAT      initial forward velocity  (default 0.3 m/s)
    --vy FLOAT      initial lateral velocity  (default 0.0 m/s)
    --yaw FLOAT     initial yaw rate          (default 0.0 rad/s)
    --standing      start in standing mode    (standing_flag = 1)
    --scene STR     scene to load: 'groundplane' (default) or 'room'
                    'room' loads scenes/ithor/FloorPlan1_physics_simple.xml,
                    converting to USD on first run (~30s) and caching in
                    isaac-sim-rl-bringup/scenes/.

Close the viewer window to exit.
"""

# ── Hydra + Isaac AppLauncher sequencing ─────────────────────────────────────
# Isaac's AppLauncher parses --headless / --enable_cameras / --device via argparse,
# while all experiment config comes from Hydra. We split sys.argv: everything
# that looks like a Hydra override (key=value) goes to Hydra's compose API;
# everything with a leading dash stays for argparse.
import sys as _sys
import pathlib as _pathlib

_argparse_argv = [_sys.argv[0]]
_hydra_overrides = []
for _arg in _sys.argv[1:]:
    if "=" in _arg and not _arg.startswith("-"):
        _hydra_overrides.append(_arg)
    else:
        _argparse_argv.append(_arg)
_sys.argv = _argparse_argv

import argparse
from isaaclab.app import AppLauncher

_iparser = argparse.ArgumentParser(description="Alex ONNX walking policy "
                                               "(Hydra-configured — see configs/)")
AppLauncher.add_app_launcher_args(_iparser)
_isaac_args = _iparser.parse_args()
app_launcher = AppLauncher(_isaac_args)
simulation_app = app_launcher.app

# Compose Hydra config after AppLauncher.
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
# __file__ = .../isaac-sim-rl-bringup/scripts/alex_room_explore/alex_onnx_walking_policy.py
# parents: [0] alex_room_explore  [1] scripts  [2] isaac-sim-rl-bringup  [3] Alex-robot
_CONFIGS_DIR = str(_pathlib.Path(__file__).resolve().parents[2] / "configs")
if _CONFIGS_DIR not in _sys.path:
    _sys.path.insert(0, _CONFIGS_DIR)
import schema as _schema   # noqa: E402
_schema.register()

_REPO_ROOT = str(_pathlib.Path(__file__).resolve().parents[3])
_hydra_overrides.insert(0, f"repo_root={_REPO_ROOT}")

with initialize_config_dir(version_base=None, config_dir=_CONFIGS_DIR):
    cfg = compose(config_name="alex_room_explore", overrides=_hydra_overrides)

print("[hydra] Composed config:")
print(OmegaConf.to_yaml(cfg))


# ── Config × AppLauncher consistency checks ──────────────────────────────────
# Rerun + head-camera sensors only work when Isaac's RTX offscreen renderer is
# active, which requires `--enable_cameras`. The two CLI layers (Hydra and
# AppLauncher argparse) don't know about each other, so we cross-check here
# and fail fast with a clear message instead of deep inside sim.reset().
_errs = []
_needs_cameras = (
    cfg.rerun.enabled
    or cfg.detector.enabled
    or cfg.yolo.enabled
    or cfg.rerun.pointcloud
)
if _needs_cameras and not getattr(_isaac_args, "enable_cameras", False):
    _reasons = []
    if cfg.rerun.enabled:    _reasons.append("rerun.enabled=true")
    if cfg.detector.enabled: _reasons.append("detector.enabled=true")
    if cfg.yolo.enabled:     _reasons.append("yolo.enabled=true")
    if cfg.rerun.pointcloud: _reasons.append("rerun.pointcloud=true")
    _errs.append(
        f"Config requires head camera ({', '.join(_reasons)}) but Isaac was "
        f"started without --enable_cameras. Add --enable_cameras to the "
        f"command line."
    )

if (cfg.detector.enabled or cfg.yolo.enabled) and not cfg.rerun.enabled:
    _errs.append(
        "detector.enabled / yolo.enabled require rerun.enabled=true "
        "(detections log to Rerun). Add rerun=full or rerun.enabled=true."
    )

if _errs:
    print("\n[config error] Config / flag mismatch:")
    for e in _errs:
        print(f"  - {e}")
    print("Aborting before Isaac starts. Fix the command and rerun.\n")
    simulation_app.close()
    import sys
    sys.exit(2)


# ── Shim: populate an args_cli namespace so the body of this script (~40
# references) keeps working without renaming. Every CLI flag in the old
# argparse version maps to a cfg leaf here.
class _Shim:
    pass
args_cli = _Shim()

# Scene
args_cli.scene = cfg.scene.name
args_cli.doors = cfg.scene.doors

# Policy / initial velocity command
args_cli.vx       = cfg.policy.vx
args_cli.vy       = cfg.policy.vy
args_cli.yaw      = cfg.policy.yaw
args_cli.standing = cfg.policy.standing

# Rerun + point cloud
args_cli.rerun                 = cfg.rerun.enabled
args_cli.pointcloud            = cfg.rerun.pointcloud
args_cli.pointcloud_stride     = cfg.rerun.pointcloud_stride
args_cli.pointcloud_max_depth  = cfg.rerun.pointcloud_max_depth

# YOLO (old `--yolo` was weights path, None when disabled — mirror that)
args_cli.yolo      = cfg.yolo.weights if cfg.yolo.enabled else None
args_cli.yolo_conf = cfg.yolo.conf

# SAM3 (old `--sam3` was prompt string, None when disabled)
args_cli.sam3      = cfg.detector.prompts if cfg.detector.enabled else None
args_cli.sam3_conf = cfg.detector.conf

# Autonomy (Phase 1: manual | fixed_xyz; Phase 2 will wire `approach`)
args_cli.autonomy_mode      = cfg.autonomy.mode
args_cli.autonomy_target    = cfg.autonomy.target
args_cli.autonomy_fixed_xyz = tuple(cfg.autonomy.fixed_xyz)
args_cli.stop_dist          = cfg.autonomy.stop_dist
args_cli.walk_speed         = cfg.autonomy.walk_speed
args_cli.search_yaw         = cfg.autonomy.search_yaw
args_cli.yaw_max            = cfg.autonomy.yaw_max
args_cli.heading_kp         = cfg.autonomy.heading_kp
args_cli.heading_walk_deg   = cfg.autonomy.heading_walk_deg
args_cli.fall_height_m      = cfg.autonomy.fall_height_m
args_cli.fall_tilt_norm     = cfg.autonomy.fall_tilt_norm
# Phase-4 recovery + stuck-detection tunables.
args_cli.stuck_window_s        = cfg.autonomy.stuck_window_s
args_cli.stuck_dist_m          = cfg.autonomy.stuck_dist_m
args_cli.recovery_stand_s      = cfg.autonomy.recovery_stand_s
args_cli.recovery_max_attempts = cfg.autonomy.recovery_max_attempts
args_cli.recovery_rotation_yaw = cfg.autonomy.recovery_rotation_yaw
args_cli.lock_conf          = cfg.autonomy.lock_conf
args_cli.min_observations   = cfg.autonomy.min_observations
# Forward-cone telemetry for the emergency brake. The cone half-angles are
# still configurable; the brake threshold is a constant in the bundle.
args_cli.obstacle_cone_h_deg = cfg.autonomy.obstacle_cone_h_deg
args_cli.obstacle_cone_v_deg = cfg.autonomy.obstacle_cone_v_deg
# Phase-2 perception goal: scene_graph save path (already exposed in schema)
args_cli.scene_graph_path   = cfg.output.scene_graph_path

# LA-6 — Loco-X agent integration. Soft feature flag, default off so
# Phase 1-4 runs are unchanged. When enabled the autonomy loop builds
# an AsyncRunner + TaskDispatcher into the bundle and calls them once
# per tick. The runner queries an LLM client (Hydra-selected:
# scripted / stdin / anthropic / openrouter); the dispatcher drains
# skill-emitted tasks (goto / face / stop / peek / survey) into the
# existing FSM control surface.
args_cli.use_loco_x = bool(getattr(cfg, "loco_x", {}).get("enabled", False))

# `autonomy=approach` requires SAM3 (perception path). Fail fast on misconfig.
if args_cli.autonomy_mode == "approach":
    if not cfg.detector.enabled:
        print("\n[config error] autonomy=approach requires detector=sam3 "
              "(SAM3 supplies the goal XYZ). Add detector=sam3 to the command.\n")
        simulation_app.close()
        import sys as _sysabort
        _sysabort.exit(2)
    if cfg.autonomy.target in (None, ""):
        print("\n[config error] autonomy=approach requires autonomy.target=<label>.\n"
              "Example: autonomy=approach autonomy.target=oven\n")
        simulation_app.close()
        import sys as _sysabort
        _sysabort.exit(2)

# ── Isaac imports (after AppLauncher) ────────────────────────────────────────
import copy
import math
import pathlib
from pathlib import Path
import time

import numpy as np
import torch
import onnxruntime as ort

if args_cli.rerun:
    import rerun as rr

_yolo_model = None
_yolo_imgsz = None
if args_cli.yolo:
    assert args_cli.rerun, "--yolo requires --rerun"
    from ultralytics import YOLO
    _yolo_task = "detect"
    try:
        import onnx as _onnx
        _m = _onnx.load(args_cli.yolo)
        _meta = {p.key: p.value for p in _m.metadata_props}
        _yolo_task = _meta.get("task", "detect")
        _imgsz_raw = _meta.get("imgsz", None)
        _yolo_imgsz = eval(_imgsz_raw) if _imgsz_raw else None
    except Exception:
        pass
    _yolo_model = YOLO(args_cli.yolo, task=_yolo_task)
    print(f"[yolo] Loaded {args_cli.yolo}  task={_yolo_task}  imgsz={_yolo_imgsz}  conf≥{args_cli.yolo_conf}")

_sam3_model = None
_sam3_processor = None
_sam3_prompts: list = []
if args_cli.sam3:
    assert args_cli.rerun, "--sam3 requires --rerun"
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    print(f"[sam3] Loading image model (checkpoint from HuggingFace cache) …")
    _sam3_model = build_sam3_image_model()
    _sam3_model = _sam3_model.to("cuda").eval()
    _sam3_processor = Sam3Processor(_sam3_model)
    _sam3_prompts = [p.strip() for p in args_cli.sam3.split(",") if p.strip()]
    print(f"[sam3] Ready. Prompts: {_sam3_prompts}  conf≥{args_cli.sam3_conf}")

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import quat_apply_inverse

from isaaclab_assets.ihmc.robots.alex import alex as alex_cfg

# ── Autonomy package (sibling dir to this script) ────────────────────────────
# Phase 1: FSM controller + _cmd translator + fall monitor. Pure logic; no
# Isaac imports inside the package, so it stays unit-testable.
# Phase 2: target_picker + goal.update_from_object + perception adapter.
_SCRIPT_DIR = str(_pathlib.Path(__file__).resolve().parent)
if _SCRIPT_DIR not in _sys.path:
    _sys.path.insert(0, _SCRIPT_DIR)
from autonomy import (   # noqa: E402
    FallMonitor,
    FSMController,
    FSMMode,
    GaitLimits,
    GoalState,
    RecoveryAgent,
    RecoveryState,
    StuckMonitor,
    YawTracker,
    forward_cone_distance,
    fsm_mode_to_cmd,
    heading_error,
    pick_goal_for_target,
    yaw_from_quat,
)
from autonomy.fsm import FSMParams                                # noqa: E402
from autonomy.planner import plan_path                            # noqa: E402
from autonomy.timing import (                                     # noqa: E402
    Timings,
    format_memory_report,
    format_timing_report,
)
from autonomy.usd_occupancy import (                              # noqa: E402
    GridFrame,
    load_occupancy_npz,
    occupancy_from_usd,
    save_occupancy_npz,
    save_topdown_png,
)
from autonomy.perception import (  # noqa: E402
    get_cam_pose_K,
    read_depth,
    read_rgb_depth,
)

# Vendored scene-graph package (see isaac-sim-rl-bringup/scene_graph/VENDORED.md).
# Provides the Phase-2 perception substrate: SAM3 → mask → unproject → dedup.
_BRINGUP = str(_pathlib.Path(__file__).resolve().parents[2])
if _BRINGUP not in _sys.path:
    _sys.path.insert(0, _BRINGUP)
from scene_graph import SceneGraph, serialize                     # noqa: E402
from scene_graph.pipeline.frame_loop import process_one_frame     # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────
_BRINGUP_ROOT = pathlib.Path(__file__).resolve().parents[2]   # .../isaac-sim-rl-bringup
_ALEX_ROBOT   = pathlib.Path(__file__).resolve().parents[3]   # .../Alex-robot

# ONNX policy — path comes from Hydra config (policy.onnx_path)
ONNX_PATH = pathlib.Path(cfg.policy.onnx_path)

# URDF — alex_models/ already in the Alex-robot repo
ISAACDATA = _ALEX_ROBOT / "alex_models" / "alex_V1_description"

# ── Policy config (from 2026-03-17_23-20-27_flatfeet/policy_cfg.yaml) ────────
# Joint order matches YAML jointParameters order = obs joint_pos_rel / last_action order.
YAML_JOINT_ORDER = [
    "LEFT_HIP_X",    "RIGHT_HIP_X",   "SPINE_Z",
    "LEFT_HIP_Z",    "RIGHT_HIP_Z",
    "LEFT_SHOULDER_Y", "NECK_Z",      "RIGHT_SHOULDER_Y",
    "LEFT_HIP_Y",    "RIGHT_HIP_Y",
    "LEFT_SHOULDER_X", "NECK_Y",      "RIGHT_SHOULDER_X",
    "LEFT_KNEE_Y",   "RIGHT_KNEE_Y",
    "LEFT_SHOULDER_Z", "RIGHT_SHOULDER_Z",
    "LEFT_ANKLE_Y",  "RIGHT_ANKLE_Y",
    "LEFT_ELBOW_Y",  "RIGHT_ELBOW_Y",
    "LEFT_ANKLE_X",  "RIGHT_ANKLE_X",
]

# homePositions from YAML
HOME_POS = {
    "LEFT_HIP_X":      0.0,   "RIGHT_HIP_X":     0.0,   "SPINE_Z":         0.0,
    "LEFT_HIP_Z":      0.0,   "RIGHT_HIP_Z":     0.0,
    "LEFT_SHOULDER_Y": 0.15,  "NECK_Z":          0.0,   "RIGHT_SHOULDER_Y":0.15,
    "LEFT_HIP_Y":     -0.35,  "RIGHT_HIP_Y":    -0.35,
    "LEFT_SHOULDER_X": 0.05,  "NECK_Y":          0.0,   "RIGHT_SHOULDER_X":-0.05,
    "LEFT_KNEE_Y":     0.7,   "RIGHT_KNEE_Y":    0.7,
    "LEFT_SHOULDER_Z": 0.05,  "RIGHT_SHOULDER_Z":-0.05,
    "LEFT_ANKLE_Y":   -0.35,  "RIGHT_ANKLE_Y":  -0.35,
    "LEFT_ELBOW_Y":   -0.5,   "RIGHT_ELBOW_Y":  -0.5,
    "LEFT_ANKLE_X":    0.0,   "RIGHT_ANKLE_X":   0.0,
}

# kp / kd from YAML
KP = {
    "LEFT_HIP_X":     80.35,  "RIGHT_HIP_X":    80.35,  "SPINE_Z":         80.35,
    "LEFT_HIP_Z":     70.5,   "RIGHT_HIP_Z":    70.5,
    "LEFT_SHOULDER_Y":26.783, "NECK_Z":          5.0,   "RIGHT_SHOULDER_Y":26.783,
    "LEFT_HIP_Y":    108.6,   "RIGHT_HIP_Y":   108.6,
    "LEFT_SHOULDER_X":26.783, "NECK_Y":          5.0,   "RIGHT_SHOULDER_X":26.783,
    "LEFT_KNEE_Y":   108.6,   "RIGHT_KNEE_Y":  108.6,
    "LEFT_SHOULDER_Z":23.5,   "RIGHT_SHOULDER_Z":23.5,
    "LEFT_ANKLE_Y":   96.8,   "RIGHT_ANKLE_Y":  96.8,
    "LEFT_ELBOW_Y":   23.5,   "RIGHT_ELBOW_Y":  23.5,
    "LEFT_ANKLE_X":   72.6,   "RIGHT_ANKLE_X":  72.6,
}
KD = {
    "LEFT_HIP_X":     8.035,  "RIGHT_HIP_X":   8.035,   "SPINE_Z":        8.035,
    "LEFT_HIP_Z":     7.05,   "RIGHT_HIP_Z":   7.05,
    "LEFT_SHOULDER_Y":8.0,    "NECK_Z":        1.0,      "RIGHT_SHOULDER_Y":8.0,
    "LEFT_HIP_Y":    10.86,   "RIGHT_HIP_Y":  10.86,
    "LEFT_SHOULDER_X":8.0,    "NECK_Y":        1.0,      "RIGHT_SHOULDER_X":8.0,
    "LEFT_KNEE_Y":   10.86,   "RIGHT_KNEE_Y": 10.86,
    "LEFT_SHOULDER_Z":4.0,    "RIGHT_SHOULDER_Z":4.0,
    "LEFT_ANKLE_Y":   9.68,   "RIGHT_ANKLE_Y": 9.68,
    "LEFT_ELBOW_Y":   4.0,    "RIGHT_ELBOW_Y": 4.0,
    "LEFT_ANKLE_X":   7.26,   "RIGHT_ANKLE_X": 7.26,
}

ACTION_SCALE = 0.3
OBS_SIZE     = 80

# Initial spawn height: bent-knee home pose (HIP_Y=-0.35, KNEE_Y=0.7, ANKLE_Y=-0.35)
SPAWN_HEIGHT = 0.93   # metres — adjust if robot spawns with feet above/below ground

SIM_DT     = 0.005
DECIMATION = 4         # 4 × 5 ms = 20 ms per policy tick = 50 Hz

# ── Scene paths ───────────────────────────────────────────────────────────────
# Pre-built USD from molmospaces (ms-download --type usd --scenes ithor).
# Symlinked to assets/usd/scenes/ithor/FloorPlan1_physics/ by ms-download.
_ROOM_USD    = _ALEX_ROBOT / "assets" / "usd" / "scenes" / "ithor" / "FloorPlan1_physics" / "scene.usda"
_HALLWAY_USD = _ALEX_ROBOT / "scenes" / "HallwayScene" / "Hallway.usdc"

# Robot spawn position inside the room scene.
# Floor of FloorPlan1_physics is at z=0 → spawn CoM at z=0.93 m.
SPAWN_POS_ROOM = (+1.47, -0.24, 0.93)
# SPAWN_POS_ROOM = (1.2, -0.56, 0.93)

# Hallway spawn — centre corridor at standing CoM height. Hallway floor is z=0.
SPAWN_POS_HALLWAY = (0.0, 0.0, 0.93)

# Room camera: inside room, near far wall corner, looking toward robot spawn.
ROOM_CAM_EYE    = (2.2, 2.2, 2.0)
ROOM_CAM_TARGET = (0.5, -0.5, 0.8)

# Hallway camera: 3rd-person perspective above spawn.
HALLWAY_CAM_EYE    = (3.0, 3.0, 3.0)
HALLWAY_CAM_TARGET = (0.0, 0.0, 0.9)

# Head camera: attached to HEAD_LINK, looks along +X (forward in body frame).
HEAD_CAM_OFFSET = (0.1, 0.0, 0.0)
HEAD_CAM_W, HEAD_CAM_H = 640, 480
CAMERA_DECIMATION = 4  # camera update every 4 policy ticks (~12.5 Hz)

# OpenGL-convention quat (wxyz) that points the camera along HEAD_LINK's +X.
# Cached at import so _step_perception can compose live world pose without
# recomputing each frame; matches what's passed to CameraCfg.OffsetCfg.rot.
# (Filled in below right after _lookat_quat_wxyz is defined.)
_HEAD_CAM_OFFSET_QUAT: tuple = (1.0, 0.0, 0.0, 0.0)  # placeholder

# Chest camera: attached to TORSO_LINK, pitched ~30° down so it sees waist-
# height obstacles (counters, tables) the head cam misses until very close.
# Used only for forward-cone obstacle distance — SAM3 still runs on head_cam.
# Offset is in TORSO_LINK frame: 10 cm forward, 0 cm side, 5 cm down from
# torso center. TORSO_LINK is at ~1.0 m world height with the home pose, so
# the chest cam sits at ~0.95 m looking 30° down — its cone covers the
# 0.6–1.5 m height band from 1–3 m away.
CHEST_CAM_OFFSET = (0.10, 0.0, -0.05)
CHEST_CAM_W, CHEST_CAM_H = 320, 240   # smaller than head cam — only depth, no SAM3
CHEST_CAM_PITCH_DEG = 30.0
_CHEST_CAM_OFFSET_QUAT: tuple = (1.0, 0.0, 0.0, 0.0)  # placeholder, filled below


def _lookat_quat_wxyz(eye, target) -> tuple:
    """(w,x,y,z) quat for a camera at eye looking at target (Z-up, opengl convention)."""
    eye    = np.array(eye,    dtype=float)
    target = np.array(target, dtype=float)
    up = np.array([0.0, 0.0, 1.0])
    z = -(target - eye); z /= np.linalg.norm(z)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-5:
        up = np.array([0.0, 1.0, 0.0]); x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x); y /= np.linalg.norm(y)
    R = np.stack([x, y, z], axis=1)
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0); w = 0.25 / s
        qx = (R[2,1]-R[1,2])*s; qy = (R[0,2]-R[2,0])*s; qz = (R[1,0]-R[0,1])*s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1]-R[1,2])/s; qx = 0.25*s; qy = (R[0,1]+R[1,0])/s; qz = (R[0,2]+R[2,0])/s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2]-R[2,0])/s; qx = (R[0,1]+R[1,0])/s; qy = 0.25*s; qz = (R[1,2]+R[2,1])/s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0]-R[0,1])/s; qx = (R[0,2]+R[2,0])/s; qy = (R[1,2]+R[2,1])/s; qz = 0.25*s
    return (float(w), float(qx), float(qy), float(qz))

# Resolve the head-cam offset quat now that _lookat_quat_wxyz is defined.
# Same call CameraCfg uses below — keeping them identical means the live world
# pose we compute matches what the renderer actually uses.
_HEAD_CAM_OFFSET_QUAT = _lookat_quat_wxyz(HEAD_CAM_OFFSET, (1.0, 0.0, 0.0))

# Chest cam: lookat target is forward + below, encoding the 30° downward pitch.
# tan(30°) ≈ 0.577 → for a 1 m forward target, drop 0.577 m below the cam.
import math as _math  # local alias; module already imports math elsewhere
_CHEST_PITCH_TAN = _math.tan(_math.radians(CHEST_CAM_PITCH_DEG))
_CHEST_CAM_OFFSET_QUAT = _lookat_quat_wxyz(
    CHEST_CAM_OFFSET,
    (CHEST_CAM_OFFSET[0] + 1.0, CHEST_CAM_OFFSET[1], CHEST_CAM_OFFSET[2] - _CHEST_PITCH_TAN),
)

# ── Mutable command state (updated by keyboard at runtime) ────────────────────
# [vx, vy, yaw_rate, standing_flag]  — populated in main() from CLI args
_cmd = np.zeros(4, dtype=np.float32)   # [vx, vy, yaw, standing]

# ── Per-run wall-clock + call-count accumulator ───────────────────────────────
# Wraps every hot path (ONNX, physics, SAM3, scene_graph, planner, occupancy
# build). The atexit hook at the bottom of this file prints the summary.
_TIMINGS = Timings()

# ── Phase-4 testing aid: F-key forces a synthetic fall ─────────────────────────
# Set by the "F" keyboard callback (see _make_keyboard); consumed at the top
# of _step_autonomy by latching the FallMonitor. Lets us test recovery
# deterministically without fighting Isaac's viewport interaction model.
_FORCE_FALL = {"requested": False}


# ── URDF resolution (same logic as alex_onnx_policy_test.py) ─────────────────
def resolve_urdf() -> str:
    pkg_prefix = "package://alex_V1_description/"
    abs_prefix = str(ISAACDATA) + "/"

    def _rewrite(src: pathlib.Path) -> pathlib.Path:
        dst = src.with_name(src.stem + "_abs_paths.urdf")
        if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
            text = src.read_text()
            if pkg_prefix in text:
                dst.write_text(text.replace(pkg_prefix, abs_prefix))
        return dst if dst.exists() else src

    candidate = ISAACDATA / "rl_urdf" / (
        "alex_v1.rlModel_nubForearms_robotAccurate_torsoFootCollisions.urdf"
    )
    if candidate.exists():
        result = _rewrite(candidate)
        print(f"[URDF] {result}")
        return str(result)

    isaaclab_root = pathlib.Path.home() 
    candidate2 = (isaaclab_root / alex_cfg.ALEX_V1_NUBFOREARMS_MINIMALCOLLISIONS_URDF).resolve()
    if candidate2.exists():
        result = _rewrite(candidate2)
        print(f"[URDF] {result}")
        return str(result)

    raise FileNotFoundError(f"Alex URDF not found. Tried:\n  {candidate}\n  {candidate2}")


# ── Scene setup ───────────────────────────────────────────────────────────────
def _occupancy_npz_path(usd_path) -> "Path":
    """Where the cached occupancy NPZ for ``usd_path`` lives.

    Sits next to the USD so it's findable but stays out of the source
    tree's tracked assets via ``.gitignore`` patterns
    (``*.occupancy.npz`` / ``*.topdown.png``).
    """
    p = Path(str(usd_path))
    return p.parent / "room.occupancy.npz"


def _build_or_load_room_occupancy(
    *,
    usd_path,
    z_band=(0.10, 1.50),
    resolution_m: float = 0.05,
):
    """Return ``(occ, gf)`` for the room USD.

    Uses a sidecar NPZ as a cache: if the NPZ exists and is newer than
    the USD, load it; otherwise rebuild from the USD and save the NPZ +
    PNG sidecars. The build is deterministic for a fixed USD, so caching
    is safe.

    Pure-Python — no Isaac dependency. Called from the autonomy bundle
    setup before the live Isaac stage is opened.
    """
    from pxr import Usd  # local import: keeps top-of-module pxr ref optional
    npz_path = _occupancy_npz_path(usd_path)
    png_path = npz_path.with_name("room.topdown.png")
    usd_mtime = Path(str(usd_path)).stat().st_mtime
    if npz_path.exists() and npz_path.stat().st_mtime >= usd_mtime:
        try:
            occ, gf = load_occupancy_npz(str(npz_path))
            print(f"[autonomy] occupancy: loaded cache {npz_path}  "
                  f"({gf.width}x{gf.height} cells, "
                  f"{100.0 * occ.sum() / occ.size:.1f}% occupied)")
            return occ, gf
        except Exception as e:
            print(f"[autonomy] occupancy: cache load failed ({e}); rebuilding")

    print(f"[autonomy] occupancy: rebuilding from {usd_path}")
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"could not open USD: {usd_path}")
    with _TIMINGS.time("occupancy.build"):
        occ, gf = occupancy_from_usd(
            stage,
            z_band=z_band,
            resolution_m=resolution_m,
            bounds_xy=None,
            skip_prim_paths=(),
            use_collision_api=True,
        )
    print(f"  grid:    {gf.width}x{gf.height} cells, "
          f"{gf.width * gf.resolution_m:.2f}x{gf.height * gf.resolution_m:.2f} m  "
          f"origin=({gf.origin_x:+.2f},{gf.origin_y:+.2f})  "
          f"{100.0 * occ.sum() / occ.size:.1f}% occupied")
    try:
        save_occupancy_npz(str(npz_path), occ, gf)
        save_topdown_png(str(png_path), occ, gf)
        print(f"  cached → {npz_path}")
        print(f"  png    → {png_path}")
    except Exception as e:
        print(f"  cache write failed: {e} (non-fatal)")
    return occ, gf


def setup_scene(scene: str):
    sim_cfg = sim_utils.SimulationCfg(dt=SIM_DT, device="cpu")
    sim = SimulationContext(sim_cfg)

    if scene == "room":
        sim.set_camera_view(eye=ROOM_CAM_EYE, target=ROOM_CAM_TARGET)
        assert _ROOM_USD.exists(), (
            f"Room USD not found: {_ROOM_USD}\n"
            "Run: ms-download --type usd --install-dir assets/usd --scenes ithor"
        )
        sim_utils.UsdFileCfg(usd_path=str(_ROOM_USD)).func(
            "/World/Room", sim_utils.UsdFileCfg(usd_path=str(_ROOM_USD))
        )
        spawn_pos = SPAWN_POS_ROOM
        print(f"[scene] Loaded room scene (FloorPlan1). Robot spawn {spawn_pos}")
    elif scene == "hallway":
        sim.set_camera_view(eye=HALLWAY_CAM_EYE, target=HALLWAY_CAM_TARGET)
        assert _HALLWAY_USD.exists(), f"Hallway USD not found: {_HALLWAY_USD}"
        # Load as sublayer — the USD has multiple root prims and no defaultPrim,
        # so UsdFileCfg references won't work. This puts every root prim
        # (Layout, GroundPlane, furniture, DoorObject_*) directly onto the stage.
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        stage.GetRootLayer().subLayerPaths.append(str(_HALLWAY_USD))
        print(f"[scene] Loaded hallway as sublayer from {_HALLWAY_USD}")

        # Drive door revolute joints to the requested target angle.
        from pxr import UsdPhysics
        target_angle = -90.0 if args_cli.doors == "open" else 0.0
        for door_name in ("DoorObject", "DoorObject_01", "DoorObject_02", "DoorObject_03"):
            joint_path = f"/{door_name}/DoorObject/Doorframe_001/RevoluteJoint"
            joint_prim = stage.GetPrimAtPath(joint_path)
            if not joint_prim.IsValid():
                continue
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
            drive.CreateTargetPositionAttr(target_angle)
            drive.CreateDampingAttr(1e4)
            drive.CreateStiffnessAttr(1e6)
        print(f"[scene]   all doors: angular drive → {target_angle}° ({args_cli.doors})")

        spawn_pos = SPAWN_POS_HALLWAY
        print(f"[scene] Robot spawn {spawn_pos}")
    else:
        sim.set_camera_view(eye=[10.0, 8.0, 2.5], target=[4.0, 0.0, 0.5])
        sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
        spawn_pos = (0.0, 0.0, SPAWN_HEIGHT)
        print(f"[scene] Loaded ground plane. Robot spawn {spawn_pos}")

    sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=2000.0)
    )

    robot_cfg = copy.deepcopy(alex_cfg.ALEX_V1_NUBS_DEFAULT_CFG)
    robot_cfg.spawn.asset_path = resolve_urdf()
    robot_cfg.init_state.joint_pos = HOME_POS   # non-zero home pose (bent-knee standing)
    robot_cfg.init_state.pos = spawn_pos
    robot = Articulation(robot_cfg.replace(prim_path="/World/Alex"))

    left_contact  = ContactSensor(ContactSensorCfg(prim_path="/World/Alex/LEFT_FOOT",  update_period=0.0, history_length=1))
    right_contact = ContactSensor(ContactSensorCfg(prim_path="/World/Alex/RIGHT_FOOT", update_period=0.0, history_length=1))

    head_cam = None
    chest_cam = None
    if args_cli.rerun:
        # Head camera on HEAD_LINK. Must be created BEFORE sim.reset() so its
        # render product gets initialised by sim.reset().
        head_rot = _lookat_quat_wxyz(HEAD_CAM_OFFSET, (1.0, 0.0, 0.0))
        head_cam = Camera(CameraCfg(
            prim_path="/World/Alex/HEAD_LINK/HeadCamera",
            update_period=SIM_DT * CAMERA_DECIMATION,
            height=HEAD_CAM_H,
            width=HEAD_CAM_W,
            data_types=["rgb", "distance_to_image_plane"],
            update_latest_camera_pose=True,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=3.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 20.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=HEAD_CAM_OFFSET,
                rot=head_rot,
                convention="opengl",
            ),
        ))
        print(f"[cameras] Created head camera at /World/Alex/HEAD_LINK/HeadCamera")

        # Chest camera on TORSO_LINK — depth only, pitched ~30° down. Feeds
        # the forward-cone obstacle check; head cam keeps doing SAM3.
        # Same render-product timing constraint — created before sim.reset().
        chest_cam = Camera(CameraCfg(
            prim_path="/World/Alex/TORSO_LINK/ChestCamera",
            update_period=SIM_DT * CAMERA_DECIMATION,
            height=CHEST_CAM_H,
            width=CHEST_CAM_W,
            data_types=["distance_to_image_plane"],
            update_latest_camera_pose=True,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=2.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 10.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=CHEST_CAM_OFFSET,
                rot=_CHEST_CAM_OFFSET_QUAT,
                convention="opengl",
            ),
        ))
        print(f"[cameras] Created chest camera at /World/Alex/TORSO_LINK/ChestCamera "
              f"(pitch={CHEST_CAM_PITCH_DEG:.0f}° down)")

    return sim, robot, left_contact, right_contact, head_cam, chest_cam


# ── Joint map: name → Isaac DOF index ────────────────────────────────────────
def build_joint_map(robot: Articulation) -> dict:
    jmap = {}
    for name in YAML_JOINT_ORDER:
        idx_list, _ = robot.find_joints(name)
        if idx_list:
            jmap[name] = idx_list[0]
        else:
            print(f"[WARNING] Joint not found in Isaac: {name}")
    return jmap


# ── Observation builder ───────────────────────────────────────────────────────
def build_obs(robot: Articulation, joint_map: dict, last_action: np.ndarray) -> np.ndarray:
    """
    Build the 80-dim observation vector for the 2026-03-17_23-20-27_flatfeet policy:
      [0:3]   base_ang_vel      — root angular velocity in body frame
      [3:6]   projected_gravity — gravity unit vector in body frame
      [6:9]   base_velocity     — commanded velocity [vx, vy, yaw_rate]
      [9]     standing_flag     — 0.0=walking, 1.0=stand still (CMD_STANDING_FLAG)
      [10]    base_height       — root z height above ground
      [11:34] joint_pos_rel     — q - homePos, YAML order
      [34:57] joint_vel_rel     — qd, YAML order
      [57:80] last_action       — previous raw policy output, YAML order
    """
    dev = robot.data.root_quat_w.device
    obs = np.zeros(OBS_SIZE, dtype=np.float32)

    # base_ang_vel [0:3]: root angular velocity in body frame
    obs[0:3] = robot.data.root_ang_vel_b[0].cpu().numpy()

    # projected_gravity [3:6]: gravity unit vector in body frame
    grav_world = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=dev)
    pg = quat_apply_inverse(robot.data.root_quat_w[0:1], grav_world)[0].cpu().numpy()
    obs[3:6] = pg

    # base_velocity_plus_standing [6:10]: [vx, vy, yaw_rate, standing_flag]
    obs[6]  = _cmd[0]   # vx
    obs[7]  = _cmd[1]   # vy
    obs[8]  = _cmd[2]   # yaw
    obs[9]  = _cmd[3]   # standing_flag

    # base_height [10]: commanded desired height (matches training — UniformHeightCommandCfg default)
    obs[10] = 0.93

    # joint_pos_rel [11:34] and joint_vel_rel [34:57]
    joint_pos = robot.data.joint_pos[0].cpu().numpy()
    joint_vel = robot.data.joint_vel[0].cpu().numpy()
    for i, name in enumerate(YAML_JOINT_ORDER):
        idx = joint_map.get(name)
        if idx is None:
            continue
        obs[11 + i] = joint_pos[idx] - HOME_POS[name]
        obs[34 + i] = joint_vel[idx]

    # last_action [57:80]
    obs[57:80] = last_action

    return obs


# ── Apply policy output to Isaac articulation ─────────────────────────────────
def apply_policy(robot: Articulation, joint_map: dict, action: np.ndarray) -> None:
    """
    action: 23-dim residual from policy (before actionScale).
    q_des = homePos + actionScale * action  (homePos=0.0 for all joints)
    Applies as PD position target with kp/kd from YAML.
    """
    num_dofs    = robot.num_joints
    pos_targets = torch.zeros(1, num_dofs)
    stiffness   = torch.zeros(1, num_dofs)
    damping     = torch.zeros(1, num_dofs)

    for i, name in enumerate(YAML_JOINT_ORDER):
        idx = joint_map.get(name)
        if idx is None:
            continue
        q_des = HOME_POS[name] + ACTION_SCALE * float(action[i])
        pos_targets[0, idx] = q_des
        stiffness[0, idx]   = KP[name]
        damping[0, idx]     = KD[name]

    robot.set_joint_position_target(pos_targets)
    robot.set_joint_effort_target(torch.zeros(1, num_dofs))
    robot.write_joint_stiffness_to_sim(stiffness)
    robot.write_joint_damping_to_sim(damping)
    robot.write_data_to_sim()


# ── Keyboard command update ───────────────────────────────────────────────────
def _make_keyboard(sim_device: str) -> Se2Keyboard:
    """Create and return an Se2Keyboard; also bind S to toggle standing mode."""
    kb = Se2Keyboard(Se2KeyboardCfg(
        v_x_sensitivity=0.6,
        v_y_sensitivity=0.6,
        omega_z_sensitivity=0.8,
        sim_device=sim_device,
    ))
    # Remap so Q/E handle strafe (LEFT arrow turns by default in Se2Keyboard)
    kb._INPUT_KEY_MAPPING.update({
        "Q": np.asarray([0.0,  1.0, 0.0]) * kb.v_y_sensitivity,
        "E": np.asarray([0.0, -1.0, 0.0]) * kb.v_y_sensitivity,
    })

    def _toggle_standing():
        _cmd[3] = 0.0 if _cmd[3] > 0.5 else 1.0
        mode = "STANDING" if _cmd[3] > 0.5 else "WALKING"
        print(f"[keyboard] mode → {mode}")

    def _force_fall():
        """Phase-4 testing hotkey. Sets a flag that ``_step_autonomy``
        translates into a forced FallMonitor latch on the next tick.
        Exercises the recovery code path without needing Isaac's
        viewport interaction (Shift+drag is finicky)."""
        _FORCE_FALL["requested"] = True
        print("[keyboard] F pressed → forcing synthetic fall on next autonomy tick")

    kb.add_callback("S", _toggle_standing)
    kb.add_callback("F", _force_fall)
    return kb


# ── Loco-X client factory (LA-7+) ────────────────────────────────────────────
def _make_loco_x_client(cfg):
    """Build the LLM client the AsyncRunner will use, based on
    ``cfg.loco_x.client``. Three backends today:

    * ``scripted`` — one-response placeholder; runs a single
      ``finish()`` and stops. Useful for the LA-6 end-to-end pipeline
      smoke test.
    * ``stdin``    — interactive REPL or a piped file. Reads one
      multi-line response per turn, bounded by an ``EOF`` line. No
      network. Drives LA-7 sim acceptance.
    * ``anthropic`` — Claude Messages API. Requires
      ``ANTHROPIC_API_KEY`` to be set in the environment. Drives
      LA-8 / LA-9 / LA-10 sim acceptance.

    On misconfiguration (e.g. anthropic selected with no API key) we
    fall back to ``scripted`` rather than crashing the sim — the
    autonomy loop keeps running and the agent reports a clear stop
    reason on its first tick.
    """
    import os
    from loco_x.llm.client import (
        AnthropicClient,
        ScriptedClient,
        StdinClient,
    )

    name = str(getattr(cfg.loco_x, "client", "scripted")).lower()

    if name == "scripted":
        return ScriptedClient(responses=[
            "```python\nfinish('loco_x=scripted placeholder — pick stdin or anthropic for real runs')\n```",
        ])

    if name == "stdin":
        path = getattr(cfg.loco_x, "stdin_path", None)
        stream = None
        if path:
            try:
                stream = open(path, "r")
            except OSError as e:
                print(f"[loco_x] stdin_path open failed ({e}); falling back to real stdin")
        return StdinClient(stream=stream)

    if name == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            print(
                "[loco_x] client=anthropic but ANTHROPIC_API_KEY is unset — "
                "falling back to a scripted placeholder. Set the env var "
                "and relaunch to enable Claude."
            )
            return ScriptedClient(responses=[
                "```python\nfail('ANTHROPIC_API_KEY not set')\n```",
            ])
        return AnthropicClient(
            api_key=key,
            model=str(getattr(cfg.loco_x, "anthropic_model", "claude-opus-4-7")),
            system_prompt=str(getattr(cfg.loco_x, "anthropic_system_prompt", "")) or None,
        )

    print(f"[loco_x] unknown client={name!r}; falling back to scripted")
    return ScriptedClient(responses=[
        f"```python\nfail('unknown loco_x.client={name!r}')\n```",
    ])


# ── Autonomy bundle ──────────────────────────────────────────────────────────
def _build_autonomy_bundle():
    """Construct FSM controller + GoalState + FallMonitor from the Hydra config.

    Returns ``None`` if autonomy is disabled (mode='manual'). Returns a dict
    otherwise, with keys: fsm, goal, fall_monitor.
    """
    if args_cli.autonomy_mode == "manual":
        return None

    params = FSMParams(
        stop_dist=args_cli.stop_dist,
        walk_speed=args_cli.walk_speed,
        search_yaw=args_cli.search_yaw,
        heading_kp=args_cli.heading_kp,
        yaw_max=args_cli.yaw_max,
        heading_walk_deg=args_cli.heading_walk_deg,
        limits=GaitLimits(),  # plan-defined hard caps
    )

    def _on_transition(old: str, new: str, info: dict) -> None:
        dist = info.get("forward_dist")
        deg  = info.get("heading_err_deg")
        dist_s = f"{dist:.2f}" if dist is not None else "—"
        deg_s  = f"{deg:+.1f}" if deg  is not None else "—"
        print(f"[autonomy] state {old} → {new}  dist={dist_s}m  heading_err={deg_s}°")

    fsm = FSMController(params, on_transition=_on_transition)
    goal = GoalState()
    fall = FallMonitor(
        fall_height_m=args_cli.fall_height_m,
        fall_tilt_norm=args_cli.fall_tilt_norm,
    )
    # Phase-4: recovery + stuck. Gait yaw_rate cap clamps the rotation
    # speed defensively so a typo in YAML can't push us above stable
    # walking-policy limits.
    recovery = RecoveryAgent(
        stand_duration_s=args_cli.recovery_stand_s,
        rotation_yaw_rate=min(args_cli.recovery_rotation_yaw,
                              GaitLimits().yaw_rate_max),
        max_attempts=args_cli.recovery_max_attempts,
    )
    stuck = StuckMonitor(
        window_s=args_cli.stuck_window_s,
        min_disp_m=args_cli.stuck_dist_m,
    )
    yaw_tracker = YawTracker()
    bundle = {
        "fsm": fsm, "goal": goal, "fall": fall,
        "recovery": recovery, "stuck": stuck, "yaw_tracker": yaw_tracker,
    }

    if args_cli.autonomy_mode == "fixed_xyz":
        goal.set_fixed(args_cli.autonomy_fixed_xyz)
        print(f"[autonomy] mode=fixed_xyz  goal_xyz={args_cli.autonomy_fixed_xyz}  "
              f"stop_dist={args_cli.stop_dist}m  walk_speed={args_cli.walk_speed}m/s")
    elif args_cli.autonomy_mode == "approach":
        # Phase 2: live SAM3 perception → SceneGraph → goal XYZ.
        sg = SceneGraph(scene=args_cli.scene, vocabulary=list(_sam3_prompts))
        bundle["scene_graph"] = sg
        bundle["target"] = args_cli.autonomy_target
        bundle["lock_conf"] = args_cli.lock_conf
        bundle["min_observations"] = args_cli.min_observations
        bundle["sam3_conf"] = args_cli.sam3_conf
        # Cache for the rerun SAM3 overlay — populated each camera tick by
        # ``_step_perception``. ``_log_sam3`` reads from here instead of
        # invoking SAM3 a second time.
        bundle["sam3_dets"] = None
        bundle["sam3_rgb_shape"] = None
        # Phase 3.5a: USD-derived 2D occupancy grid for the deliberative
        # planner. Built once at startup; reused for every plan call.
        # Only meaningful for the room scene (groundplane has no obstacles);
        # other scenes can opt in by adding a USD-path mapping below.
        if args_cli.scene == "room":
            try:
                occ, gf = _build_or_load_room_occupancy(usd_path=_ROOM_USD)
                bundle["occ"] = occ
                bundle["grid_frame"] = gf
                # Inflation = robot footprint clearance for A*. 0.45 m
                # is a small bump from the original 0.40 to add a bit of
                # swing-leg margin in the FloorPlan1 kitchen's marginal
                # corridors. 0.55 was tried first and proved too tight —
                # A* found no path at all from spawn to stove. 0.45
                # threads the needle: still finds a path, slightly less
                # likely to clip on near-misses than 0.40.
                bundle["plan_inflation_m"] = 0.45
                bundle["plan_waypoint_radius_m"] = 0.40
                bundle["path"] = None        # filled in on first goal-lock
                bundle["path_index"] = 0
            except Exception as e:
                print(f"[autonomy] occupancy build failed: {e}  — planner disabled")
        # Phase 3: forward-cone obstacle distance, refreshed each camera tick.
        # +inf until the camera produces its first depth frame — caller treats
        # infinite as "clear" so APPROACH starts walking immediately.
        bundle["obstacle_dist"] = float("inf")
        # Emergency-brake threshold. Last-resort safety only — Phase 3.5's
        # planner is the primary obstacle-handling path.
        bundle["emergency_dist"]      = 0.5
        bundle["obstacle_cone_h_deg"] = args_cli.obstacle_cone_h_deg
        bundle["obstacle_cone_v_deg"] = args_cli.obstacle_cone_v_deg
        bundle["_emergency_prev"] = False
        planner_state = (
            f"planner=on(inflation={bundle.get('plan_inflation_m', 0.40):.2f}m, "
            f"waypoint_radius={bundle.get('plan_waypoint_radius_m', 0.40):.2f}m)"
            if "occ" in bundle else "planner=off"
        )
        recovery_state = (
            f"recovery=on(stand={args_cli.recovery_stand_s}s, "
            f"max={args_cli.recovery_max_attempts}, "
            f"stuck_window={args_cli.stuck_window_s}s, "
            f"stuck_min={args_cli.stuck_dist_m}m)"
        )
        print(f"[autonomy] mode=approach  target='{args_cli.autonomy_target}'  "
              f"lock_conf={args_cli.lock_conf}  min_obs={args_cli.min_observations}  "
              f"stop_dist={args_cli.stop_dist}m  walk_speed={args_cli.walk_speed}m/s  "
              f"emergency_dist={bundle['emergency_dist']}m  "
              f"cone=±{args_cli.obstacle_cone_h_deg:.0f}°h/±{args_cli.obstacle_cone_v_deg:.0f}°v  "
              f"chest_cam=on(pitch={CHEST_CAM_PITCH_DEG:.0f}°)  "
              f"{planner_state}  {recovery_state}")
        print(f"[autonomy] SAM3 prompts: {_sam3_prompts}")

    # ── LA-6: Loco-X agent integration (opt-in via cfg.loco_x.enabled) ─
    if args_cli.use_loco_x:
        # Bundle fields the LA-1 skills + LA-6 dispatcher read/write.
        bundle.setdefault("task_queue", [])
        bundle.setdefault("task_history", [])
        bundle.setdefault("scene_nodes", [])
        bundle.setdefault("agent_should_stop", False)
        bundle.setdefault("task_result_status", None)
        bundle.setdefault("task_result_reason", None)
        bundle.setdefault("last_action", None)
        bundle.setdefault("fsm_mode", "IDLE")
        bundle.setdefault("robot_pose", {"xy": (0.0, 0.0), "yaw_rad": 0.0})
        bundle.setdefault("goal_lock_xyz", None)
        bundle.setdefault("face_yaw_rad", None)
        bundle.setdefault("safe_stop_requested", False)
        bundle.setdefault("head_yaw_request", None)
        bundle.setdefault("head_sweep_queue", None)

        # Import lazily so a Phase-1-4-only run doesn't require the
        # loco_x package to be on sys.path.
        import sys as _sys, pathlib as _pl
        _BRINGUP = _pl.Path(__file__).resolve().parents[2]
        if str(_BRINGUP) not in _sys.path:
            _sys.path.insert(0, str(_BRINGUP))
        from loco_x.agent import AsyncRunner, RunnerConfig, TaskDispatcher

        # LA-7+: Hydra-selected client. Three backends covered today:
        # scripted / stdin / anthropic. See configs/loco_x/*.yaml.
        client = _make_loco_x_client(cfg)
        runner_cfg = RunnerConfig(
            enabled=True,
            tick_hz=float(getattr(cfg, "loco_x", {}).get("tick_hz", 2.0)),
            max_turns=int(getattr(cfg, "loco_x", {}).get("max_turns", 20)),
            exec_timeout_s=float(getattr(cfg, "loco_x", {}).get("exec_timeout_s", 5.0)),
        )
        bundle["agent"] = AsyncRunner(
            bundle=bundle, client=client, config=runner_cfg,
        )
        bundle["dispatcher"] = TaskDispatcher()
        print(f"[loco_x] agent enabled  tick_hz={runner_cfg.tick_hz} "
              f"max_turns={runner_cfg.max_turns}")

    return bundle


def _step_perception(bundle: dict, head_cam, chest_cam, tick: int) -> None:
    """One perception tick (camera-rate): SAM3 → SceneGraph → goal update.

    No-op unless we're in `approach` mode and a head camera is attached.
    Obstacle distance is computed from ``chest_cam`` if provided (it sees
    waist-height obstacles the head cam misses), otherwise from ``head_cam``.
    """
    if bundle is None or "scene_graph" not in bundle:
        return
    if head_cam is None or _sam3_processor is None:
        return

    rgb_depth = read_rgb_depth(head_cam)
    if rgb_depth is None:
        return  # camera not warm yet
    rgb, depth = rgb_depth

    try:
        cam_pos, cam_quat_wxyz, K = get_cam_pose_K(
            head_cam,
            robot=bundle.get("_robot_for_debug"),
            body_name="HEAD_LINK",
            cam_offset_pos=HEAD_CAM_OFFSET,
            cam_offset_quat_wxyz=_HEAD_CAM_OFFSET_QUAT,
        )
    except Exception as e:
        if tick == 0:
            print(f"[autonomy] perception: head-cam pose unavailable yet: {e}")
        return

    sg: SceneGraph = bundle["scene_graph"]
    with _TIMINGS.time("scene_graph.process_one_frame"):
        dets = process_one_frame(
            sg, rgb=rgb, depth=depth, K=K,
            cam_pos=cam_pos, cam_quat_wxyz=cam_quat_wxyz,
            tick=tick,
            sam3_processor=_sam3_processor,
            prompts=list(_sam3_prompts),
            conf_threshold=bundle["sam3_conf"],
        )
    # Cache detections + frame shape for the rerun overlay so ``_log_sam3``
    # can reuse them instead of re-running SAM3 on the same RGB frame.
    # Halves SAM3 wall-clock per camera tick.
    bundle["sam3_dets"] = dets
    bundle["sam3_rgb_shape"] = rgb.shape[:2]

    # Phase 3: forward-cone obstacle distance. Prefer the chest cam — its
    # 30°-downward pitch sees counter-tops and tables from much farther away
    # than the head cam, which from 1.6 m looks too high to register a 0.9 m
    # surface until the robot is < 0.5 m from it. Fall back to head cam if
    # the chest cam isn't warm yet so the autonomy bundle always has a value.
    obstacle_depth = None
    obstacle_K = None
    if chest_cam is not None:
        chest_depth = read_depth(chest_cam)
        if chest_depth is not None:
            obstacle_depth = chest_depth
            obstacle_K = chest_cam.data.intrinsic_matrices[0].cpu().numpy().astype(np.float32)
    if obstacle_depth is None:
        obstacle_depth = depth
        obstacle_K = K
    # Single-zone forward cone for the emergency brake. Steering uses the
    # Phase 3.5 USD planner; this is purely "stop if something pops up
    # very close in the cone". Center-zone semantics are equivalent to
    # the original ``forward_cone_distance`` over the full cone.
    bundle["obstacle_dist"] = forward_cone_distance(
        obstacle_depth, obstacle_K,
        h_deg=bundle["obstacle_cone_h_deg"],
        v_deg=bundle["obstacle_cone_v_deg"],
    )

    # Goal selection: highest-confidence object whose label matches `target`
    # and which has been observed >= min_observations times.
    obj = pick_goal_for_target(
        sg, bundle["target"],
        lock_conf=bundle["lock_conf"],
        min_observations=bundle["min_observations"],
    )
    if obj is not None:
        goal: GoalState = bundle["goal"]
        was_locked = goal.locked
        goal.update_from_object(obj, lock_conf=bundle["lock_conf"])
        if goal.locked and not was_locked:
            xyz_s = ", ".join(f"{v:+.2f}" for v in obj.position_xyz)
            print(f"[autonomy] goal LOCKED on '{obj.label}' id={obj.id}  "
                  f"xyz=({xyz_s})  score={obj.confidence:.2f}  "
                  f"n_obs={obj.n_observations}")
            # Phase 3.5c: plan a path through the occupancy grid the
            # moment the goal latches. The waypoint follower in
            # ``_step_autonomy`` walks the FSM along the resulting
            # waypoints; the planner is run once here and not on every
            # tick (the map is static; the goal doesn't move once locked).
            _maybe_plan_path_on_lock(bundle)


def _maybe_plan_path_on_lock(bundle: dict) -> None:
    """Phase 3.5c: plan a path from the robot's current XY to the just-
    locked goal XY, on the occupancy grid built at startup. No-op if the
    bundle has no occupancy (e.g. groundplane scene) or no robot ref.
    """
    if "occ" not in bundle:
        return
    robot = bundle.get("_robot_for_debug")
    goal: GoalState = bundle["goal"]
    if robot is None or goal.xyz is None:
        return
    pos = robot.data.root_pos_w[0].cpu().numpy()
    start_xy = (float(pos[0]), float(pos[1]))
    goal_xy  = (float(goal.xyz[0]), float(goal.xyz[1]))
    occ      = bundle["occ"]
    gf: GridFrame = bundle["grid_frame"]
    inflation_m  = bundle["plan_inflation_m"]

    with _TIMINGS.time("planner.plan_path"):
        path = plan_path(start_xy, goal_xy, occ, gf, inflation_m=inflation_m)
    bundle["path"] = path
    bundle["path_index"] = 0

    if path is None:
        print(f"[autonomy] planner: NO PATH from "
              f"({start_xy[0]:+.2f},{start_xy[1]:+.2f}) → "
              f"({goal_xy[0]:+.2f},{goal_xy[1]:+.2f})  "
              f"— falling back to direct heading control")
        return
    seg_len = sum(
        ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
        for a, b in zip(path[:-1], path[1:])
    )
    print(f"[autonomy] planner: {len(path)} waypoints, length {seg_len:.2f}m, "
          f"inflation {inflation_m:.2f}m")
    for i, (x, y) in enumerate(path):
        print(f"  wp[{i:2d}] ({x:+.2f}, {y:+.2f})")

    # Rerun: draw the path as a polyline at floor height (z=0.05) so it
    # sits just above the ground plane. Logged once per plan; subsequent
    # waypoint advances re-use this entity unchanged.
    if args_cli.rerun:
        try:
            import rerun as rr  # local import — same as existing usage
            pts = np.asarray(
                [[x, y, 0.05] for (x, y) in path], dtype=np.float32,
            )
            rr.log("world/scene/path",
                   rr.LineStrips3D([pts], colors=[[220, 30, 30]], radii=[0.04]))
            rr.log("world/scene/path/waypoints",
                   rr.Points3D(pts, colors=[[255, 140, 0]] * len(pts), radii=0.06))
            # Goal (literal) marker.
            rr.log("world/scene/goal",
                   rr.Points3D([[goal_xy[0], goal_xy[1], 0.05]],
                               colors=[[40, 80, 230]], radii=0.10))
        except Exception as e:
            print(f"[autonomy] planner: rerun log failed (non-fatal): {e}")


def _replan_from_current_pose(bundle: dict) -> None:
    """Phase 4: re-plan after a recovery (fall stand-up or stuck rotation).

    Clears the cached path and re-invokes the lock-time planner so it
    re-runs A* from the robot's *current* XY rather than the original
    lock-on XY. The goal lock is preserved (``goal.locked`` stays True),
    so this is a pure path refresh — no SAM3 re-discovery needed.

    No-op for scenes without an occupancy grid (e.g. groundplane).
    """
    if "occ" not in bundle:
        return
    bundle["path"] = None
    bundle["path_index"] = 0
    _maybe_plan_path_on_lock(bundle)


def _step_autonomy(bundle: dict, robot) -> None:
    """One autonomy tick: read robot pose, run FSM, write _cmd in place."""
    fsm: FSMController = bundle["fsm"]
    goal: GoalState = bundle["goal"]
    fall: FallMonitor = bundle["fall"]
    recovery: RecoveryAgent = bundle["recovery"]
    stuck: StuckMonitor = bundle["stuck"]
    yaw_track: YawTracker = bundle["yaw_tracker"]

    pos  = robot.data.root_pos_w[0].cpu().numpy()
    quat = robot.data.root_quat_w[0].cpu().numpy()   # (w, x, y, z)

    # Projected gravity in base frame — used to detect tilt > ~45°.
    dev = robot.data.root_quat_w.device
    grav_world = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=dev)
    pg = quat_apply_inverse(robot.data.root_quat_w[0:1], grav_world)[0].cpu().numpy()
    yaw = yaw_from_quat(quat)
    now = time.time()

    # ── LA-6: keep the Loco-X agent's view of the robot pose fresh ──
    # The agent observation builder reads bundle["robot_pose"]; the
    # dispatcher's writes (goal_lock_xyz, face_yaw_rad, ...) must
    # land *before* the rest of the FSM logic this tick consumes
    # them, so we drain the queue here at the top.
    if "agent" in bundle:
        try:
            bundle["robot_pose"] = {
                "xy": (float(pos[0]), float(pos[1])),
                "yaw_rad": float(yaw),
            }
            # FSMController.mode is a plain string (FSMMode.SEARCH ==
            # "search"); the runner gate accepts "idle" / "arrived" so
            # the agent will tick whenever the FSM isn't actively
            # walking or searching. We also map "arrived" → "IDLE" so
            # one-shot tasks complete naturally and the next observation
            # surfaces ARRIVED in last_action.
            fsm_str = str(getattr(bundle["fsm"], "mode", "search")).lower()
            if fsm_str in ("search", "idle", "arrived"):
                bundle["fsm_mode"] = "IDLE"
            else:
                bundle["fsm_mode"] = fsm_str.upper()
            # Bridge SAM3's SceneGraph objects → the agent's
            # bundle["scene_nodes"] shape. The agent observation
            # builder reads ``label / world_xy / last_seen /
            # confidence`` from this list. We rebuild it each tick
            # so newly-detected objects appear next agent turn (D13:
            # observation is a snapshot at build time, perception
            # keeps running between snapshots).
            sg = bundle.get("scene_graph")
            if sg is not None and hasattr(sg, "objects"):
                nodes: list = []
                for obj in sg.objects.values():
                    pos_xyz = getattr(obj, "position_xyz", None)
                    if pos_xyz is None or len(pos_xyz) < 2:
                        continue
                    nodes.append({
                        "label": getattr(obj, "label", "?"),
                        "world_xy": (float(pos_xyz[0]), float(pos_xyz[1])),
                        "last_seen": float(getattr(obj, "last_seen_tick", 0)),
                        "confidence": float(getattr(obj, "confidence", 0.0)),
                    })
                bundle["scene_nodes"] = nodes
            # LA-7 diagnostic: print the gate state once per ~2 s so we
            # can see *why* the agent isn't firing when Phase 1-4
            # already grabbed the goal. Reads the runner's internal
            # ``_inner`` to call reason_for_skip without polluting the
            # public AsyncRunner surface.
            _last_diag_t = bundle.get("_loco_x_last_diag_t", 0.0)
            if now - _last_diag_t > 2.0:
                bundle["_loco_x_last_diag_t"] = now
                inner = getattr(bundle["agent"], "_inner", None)
                reason = inner.reason_for_skip(now) if inner else None
                if reason is None:
                    print(f"[loco_x] gate: PERMITS this tick "
                          f"(fsm={bundle['fsm_mode']}, queue_len="
                          f"{len(bundle.get('task_queue') or [])}, "
                          f"scene_nodes={len(bundle.get('scene_nodes') or [])})")
                else:
                    print(f"[loco_x] gate: skip — {reason}")
            bundle["agent"].poll(now)
            if "dispatcher" in bundle:
                try:
                    bundle["dispatcher"].drain(bundle)
                except Exception as e:
                    print(f"[loco_x] dispatcher error: {e}")
            # If the agent has decided to stop, force the safe-stop _cmd
            # via the same path Phase 4's RECOVERY-FAILED case uses.
            if bundle.get("agent_should_stop") and bundle.get("safe_stop_requested"):
                _cmd[:] = (0.0, 0.0, 0.0, 1.0)
                return
            # LA-7: `face(yaw_rad)` skill. Dispatcher writes the target
            # yaw into bundle["face_yaw_rad"]; we close the loop here
            # by driving _cmd[2] until the current yaw reaches the
            # target. When SAM3 hasn't grounded the goal yet, the
            # agent rotates to scan a new field of view. While face
            # is active we short-circuit the FSM (return early) so
            # the policy follows our yaw-rate command instead.
            target_yaw = bundle.get("face_yaw_rad")
            if target_yaw is not None:
                err = math.atan2(
                    math.sin(target_yaw - yaw),
                    math.cos(target_yaw - yaw),
                )
                if abs(err) < math.radians(8.0):
                    # Reached the target; clear and let Phase 1-4 take over.
                    bundle["face_yaw_rad"] = None
                    print(f"[loco_x] face: reached target yaw "
                          f"{math.degrees(target_yaw):+.1f}° (err {math.degrees(err):+.1f}°)")
                else:
                    yaw_rate = 0.4 if err > 0 else -0.4
                    _cmd[0] = 0.0
                    _cmd[1] = 0.0
                    _cmd[2] = yaw_rate
                    _cmd[3] = 0.0
                    return
        except Exception as e:
            # Never let the agent integration crash the autonomy loop
            # — print and continue with the existing Phase 1-4 path.
            print(f"[loco_x] tick error (continuing without agent): {e}")
            import traceback as _tb
            _tb.print_exc()

    # ── Phase 4: recovery state machine (top of tick) ──────────────────
    #
    # Five mutually-exclusive sub-paths, in priority order:
    #   1. FAILED — recovery already gave up; safe-stop forever.
    #   2. STANDING — currently holding (0,0,0,1) for the stand window.
    #      When the window ends, re-check fall status: success → reset
    #      everything and re-plan; failure → bump attempt count, retry
    #      or transition to FAILED.
    #   3. ROTATING — closed-loop yaw rotation. Done when YawTracker
    #      reports |delta| ≥ target. On done: reset and re-plan.
    #   4. Fresh fall — FallMonitor latched and recovery is IDLE → enter
    #      STANDING for attempt 1.
    #   5. Fall-through — none of the above; run normal autonomy.
    if recovery.state is RecoveryState.FAILED:
        with _TIMINGS.time("recovery.failed_hold"):
            _cmd[0], _cmd[1], _cmd[2], _cmd[3] = recovery.cmd()
        return

    if recovery.state is RecoveryState.STANDING:
        if recovery.is_standing_done(now=now):
            # Re-evaluate fall status with a fresh latch.
            fall.reset()
            re_fallen = fall.update(
                float(pos[2]),
                proj_grav_xy=(float(pg[0]), float(pg[1])),
            )
            if not re_fallen:
                print(f"[autonomy] recovery succeeded "
                      f"(attempt {recovery.attempts_used + 1}/"
                      f"{recovery.max_attempts}) — re-planning from current pose")
                recovery.succeed()
                fsm.reset()
                stuck.reset()
                _replan_from_current_pose(bundle)
                # Fall through to normal logic this tick.
            else:
                recovery.fail_attempt()
                if recovery.state is RecoveryState.FAILED:
                    print(f"[autonomy] recovery FAILED after "
                          f"{recovery.max_attempts} attempts — autonomy disabled")
                    _cmd[0], _cmd[1], _cmd[2], _cmd[3] = recovery.cmd()
                    return
                print(f"[autonomy] still fallen — retrying "
                      f"(attempt {recovery.attempts_used + 1}/"
                      f"{recovery.max_attempts})")
                recovery.begin_standing(now=now)
                _cmd[0], _cmd[1], _cmd[2], _cmd[3] = recovery.cmd()
                return
        else:
            with _TIMINGS.time("recovery.standing"):
                _cmd[0], _cmd[1], _cmd[2], _cmd[3] = recovery.cmd()
            return

    if recovery.state is RecoveryState.ROTATING:
        done = yaw_track.update(yaw_now=yaw)
        if done:
            print(f"[autonomy] rotation complete (delta={math.degrees(yaw_track.delta):+.1f}°) "
                  f"— re-planning from current pose")
            recovery.succeed()
            stuck.reset()
            _replan_from_current_pose(bundle)
            # Fall through to normal logic this tick.
        else:
            sign = 1.0 if yaw_track.target_rad >= 0.0 else -1.0
            with _TIMINGS.time("recovery.rotating"):
                _cmd[0], _cmd[1], _cmd[2], _cmd[3] = recovery.cmd(rotation_sign=sign)
            return

    # Phase-4 testing aid: if F was pressed, force the FallMonitor latch
    # so the recovery code path engages even when the robot is upright.
    # Cleared the moment we consume it; recovery's own re-check at the
    # end of the standing window will see the (real, upright) pose and
    # transition to "succeeded" cleanly.
    if _FORCE_FALL["requested"]:
        _FORCE_FALL["requested"] = False
        fall.fallen = True

    is_fallen = fall.update(float(pos[2]), proj_grav_xy=(float(pg[0]), float(pg[1])))
    if is_fallen and recovery.state is RecoveryState.IDLE:
        print(f"[autonomy] [FALL] root_z={pos[2]:.2f}m  "
              f"proj_grav_xy=({pg[0]:+.2f},{pg[1]:+.2f}) — entering recovery "
              f"(attempt 1/{recovery.max_attempts})")
        recovery.begin_standing(now=now)
        _cmd[0], _cmd[1], _cmd[2], _cmd[3] = recovery.cmd()
        return

    # Phase 3.5c — waypoint follower.
    #
    # When a planner path is loaded, intermediate waypoints are driven
    # by the **translator directly**, *bypassing* the FSM's stop_dist /
    # arrival logic. Reason: smoothed path waypoints are typically
    # spaced 0.3–1.0 m apart — well inside ``stop_dist=1.0 m``. If we
    # fed each intermediate waypoint to ``fsm.step`` it would declare
    # ARRIVED instantly on every one, gait would freeze and re-engage
    # tens of times per second, and the policy would destabilise (saw
    # this in the first integration test: ARRIVED↔APPROACH flapping →
    # fall).
    #
    # Only the **last waypoint** goes through ``fsm.step`` with the
    # original goal, so the FSM arrival check fires on the real target.
    using_path = False
    if bundle.get("path") is not None:
        path = bundle["path"]
        idx = bundle["path_index"]
        # Advance through every waypoint we're already inside the
        # waypoint radius of.
        while idx < len(path) - 1:
            wx, wy = path[idx]
            d_wp = math.hypot(pos[0] - wx, pos[1] - wy)
            if d_wp >= bundle["plan_waypoint_radius_m"]:
                break
            idx += 1
            print(f"[autonomy] planner: advance → wp[{idx}]=({path[idx][0]:+.2f},"
                  f"{path[idx][1]:+.2f})  ({len(path) - 1 - idx} remaining)")
        bundle["path_index"] = idx

        # Rerun: highlight the current waypoint each tick. Cheap.
        if args_cli.rerun:
            try:
                import rerun as rr
                wx, wy = path[idx]
                rr.log("world/scene/current_waypoint",
                       rr.Points3D([[wx, wy, 0.05]],
                                   colors=[[40, 200, 40]], radii=0.10))
            except Exception:
                pass

        if idx < len(path) - 1:
            # Drive an intermediate waypoint via the translator only.
            using_path = True
            wx, wy = path[idx]
            herr = heading_error(float(pos[0]), float(pos[1]), float(yaw),
                                 float(wx), float(wy))
            vx, vy, yawr, standing = fsm_mode_to_cmd(
                FSMMode.APPROACH,
                heading_err_rad=herr,
                walk_speed=fsm.params.walk_speed,
                search_yaw=fsm.params.search_yaw,
                heading_kp=fsm.params.heading_kp,
                yaw_max=fsm.params.yaw_max,
                heading_walk_deg=fsm.params.heading_walk_deg,
                limits=fsm.params.limits,
            )
            # Maintain FSM bookkeeping so the rest of the script (and
            # the FALL transition) sees consistent state. We're walking
            # toward a sub-goal, so APPROACH is the right declared mode.
            if fsm.mode != FSMMode.APPROACH:
                old = fsm.mode
                fsm.mode = FSMMode.APPROACH
                fsm.last_dist = math.hypot(pos[0] - wx, pos[1] - wy)
                fsm.last_heading_err = herr
            else:
                fsm.last_dist = math.hypot(pos[0] - wx, pos[1] - wy)
                fsm.last_heading_err = herr

    if not using_path:
        # Either no path, or we're at the last waypoint. Before handing
        # off to the FSM (which would declare ARRIVED and stop with the
        # robot still facing whatever direction the last segment had),
        # check if we should first **face** the goal.
        #
        # Rotate-in-place band: dist < stop_dist AND |heading_err| > tol.
        # Pure-yaw command (no vx). Once heading is within tolerance,
        # fall through to ``fsm.step`` which will enter ARRIVED.
        face_tol_deg = 10.0
        if (goal.xyz is not None
                and not is_fallen
                and fsm.mode != FSMMode.FALLEN):
            tx, ty, _ = goal.xyz
            d = math.hypot(pos[0] - tx, pos[1] - ty)
            herr = heading_error(float(pos[0]), float(pos[1]), float(yaw),
                                 float(tx), float(ty))
            if d < fsm.params.stop_dist and abs(math.degrees(herr)) > face_tol_deg:
                # Rotate toward goal — same yaw P-controller as APPROACH,
                # but with vx forced to zero. ``arrived_facing`` is a
                # display-only state name; the FSM mode itself stays
                # APPROACH so the arrival print fires correctly once we
                # finish facing.
                if fsm.mode != FSMMode.APPROACH:
                    fsm.mode = FSMMode.APPROACH
                yawr = max(-fsm.params.yaw_max,
                           min(fsm.params.yaw_max,
                               fsm.params.heading_kp * herr))
                vx = 0.0
                vy = 0.0
                standing = 0.0
                fsm.last_dist = d
                fsm.last_heading_err = herr
                # Edge-trigger log so we can see it engage.
                if not bundle.get("_facing_logged", False):
                    print(f"[autonomy] facing goal  dist={d:.2f}m  "
                          f"heading_err={math.degrees(herr):+.1f}° → rotating in place")
                    bundle["_facing_logged"] = True
            else:
                bundle["_facing_logged"] = False
                vx, vy, yawr, standing = fsm.step(
                    float(pos[0]), float(pos[1]), float(yaw),
                    goal=goal,
                    fallen=is_fallen,
                )
        else:
            vx, vy, yawr, standing = fsm.step(
                float(pos[0]), float(pos[1]), float(yaw),
                goal=goal,
                fallen=is_fallen,
            )

    # ── Phase 4: stuck monitor ──────────────────────────────────────────
    # Only meaningful while APPROACH commands forward velocity. The
    # monitor's buffer is dropped on inactive samples, so stand-still
    # (ARRIVED, SEARCH, emergency brake holds) and recovery (handled
    # above with early returns) cannot accumulate stuck-time.
    approach_active = (fsm.mode == FSMMode.APPROACH and vx > 0.0)
    if stuck.update(float(pos[0]), float(pos[1]),
                    active=approach_active, now=now):
        print(f"[autonomy] STUCK at ({pos[0]:+.2f},{pos[1]:+.2f}) — "
              f"rotating 90° in place")
        # Pick rotation direction based on heading toward goal: if the
        # goal is on our left (heading_err > 0), rotate left; else right.
        rot_target = math.pi / 2.0
        if goal.xyz is not None:
            tx, ty, _ = goal.xyz
            herr = heading_error(float(pos[0]), float(pos[1]), float(yaw),
                                 float(tx), float(ty))
            rot_target = math.pi / 2.0 if herr >= 0.0 else -math.pi / 2.0
        yaw_track.target_rad = rot_target
        yaw_track.start(yaw_now=yaw)
        recovery.begin_rotation()
        # Invalidate the path — the wedged spot is presumed compromised.
        bundle["path"] = None
        bundle["path_index"] = 0
        sign = 1.0 if rot_target >= 0.0 else -1.0
        _cmd[0], _cmd[1], _cmd[2], _cmd[3] = recovery.cmd(rotation_sign=sign)
        return

    # Emergency brake (chest-cam depth). Last-resort safety: if something
    # very close shows up in the forward cone (planner missed it, scene
    # changed mid-run, scan was incomplete), zero ``_cmd`` and hold until
    # it clears. No steering — the planner (Phase 3.5) is responsible for
    # going around obstacles. Reactive cone steering was tried in the
    # deprecated Phase 3 and failed against wall-shaped obstacles; see
    # ``isaac-sim-rl-bringup/docs/phase3_retrospective.md``.
    if "emergency_dist" in bundle and fsm.mode == FSMMode.APPROACH:
        cone_d = bundle.get("obstacle_dist", float("inf"))
        emerg_d = bundle["emergency_dist"]
        emergency = cone_d < emerg_d
        if emergency:
            vx = 0.0
            vy = 0.0
        prev_em = bundle.get("_emergency_prev", False)
        if emergency and not prev_em:
            print(f"[autonomy] EMERGENCY STOP  cone_dist={cone_d:.2f}m < "
                  f"emergency_dist={emerg_d:.2f}m — holding")
        elif prev_em and not emergency:
            print(f"[autonomy] emergency CLEARED  cone_dist={cone_d:.2f}m")
        bundle["_emergency_prev"] = emergency

    _cmd[0] = vx
    _cmd[1] = vy
    _cmd[2] = yawr
    _cmd[3] = standing


_PC_LOG_ONCE = {"done": False}


_SAM3_WARN_ONCE = {"done": False}

def _log_sam3(dets, rgb_shape) -> None:
    """Render the latest SAM3 detections as a Rerun overlay (segmentation
    mask + bounding boxes). Consumes the cached ``RawDetection`` list
    produced by ``scene_graph.process_one_frame`` in ``_step_perception``
    — no second SAM3 inference. Cuts ~50% of per-frame SAM3 wall-clock.

    Parameters
    ----------
    dets
        ``List[RawDetection]`` from the most recent perception tick.
        ``None`` or empty → mask is cleared and bbox layer is cleared.
    rgb_shape
        ``(H, W)`` of the source RGB frame; needed to size the mask
        layer. Pulled from the same cache so the call site doesn't have
        to keep the RGB array around.
    """
    if dets is None or rgb_shape is None:
        return
    try:
        with _TIMINGS.time("sam3.postprocess"):
            mask_layer = np.zeros(rgb_shape, dtype=np.uint16)
            all_boxes = []
            all_labels = []
            all_class_ids = []
            # Stable label → class_id map so the same prompt renders the
            # same colour across frames.
            label_to_id = {p: i + 1 for i, p in enumerate(_sam3_prompts)}
            for det in dets:
                cls_id = label_to_id.get(det.label, 0)
                if cls_id == 0:
                    continue   # unknown label — shouldn't happen, but be safe
                m = det.mask
                if m is not None and m.shape == rgb_shape:
                    mask_layer[m] = cls_id
                all_boxes.append(np.asarray(det.bbox_xyxy, dtype=np.float32).reshape(-1)[:4])
                all_labels.append(f"{det.label} {det.score:.2f}")
                all_class_ids.append(cls_id)

        rr.log("camera/head/rgb/sam3_mask", rr.SegmentationImage(mask_layer))
        if all_boxes:
            rr.log("camera/head/rgb/sam3", rr.Boxes2D(
                array=np.stack(all_boxes),
                array_format=rr.Box2DFormat.XYXY,
                labels=all_labels,
                class_ids=all_class_ids,
            ))
        else:
            rr.log("camera/head/rgb/sam3", rr.Clear(recursive=False))
    except Exception as e:
        if not _SAM3_WARN_ONCE["done"]:
            print(f"[sam3] overlay failed: {e}")
            _SAM3_WARN_ONCE["done"] = True


def _log_head_pointcloud(head_cam, depth: np.ndarray, rgb: np.ndarray,
                         robot, stride: int, max_depth: float) -> None:
    """Unproject head-camera depth into world-frame coloured points and log to Rerun.

    Pose source: ``get_cam_pose_K`` composes from ``robot.data.body_link_pos_w``
    + the static CameraCfg offset. We can't use the simpler
    ``head_cam.data.pos_w`` / ``data.quat_w_opengl`` path here because Isaac
    Lab returns the **USD-authored** prim pose for cameras attached to
    articulated links, not the live pose — so the point cloud would be
    painted at the spawn frame forever even as the robot walks. Same bug
    we hit in Phase 2 with SAM3 unprojection.
    """
    try:
        pos, quat_wxyz, K = get_cam_pose_K(
            head_cam,
            robot=robot,
            body_name="HEAD_LINK",
            cam_offset_pos=HEAD_CAM_OFFSET,
            cam_offset_quat_wxyz=_HEAD_CAM_OFFSET_QUAT,
        )
    except Exception as e:
        if not _PC_LOG_ONCE["done"]:
            print(f"[pointcloud] failed to read camera pose/intrinsics: {e}")
            _PC_LOG_ONCE["done"] = True
        return

    # Subsample for speed
    if depth.ndim == 3:
        depth = depth[..., 0]
    H, W = depth.shape
    vs = np.arange(0, H, stride)
    us = np.arange(0, W, stride)
    vv, uu = np.meshgrid(vs, us, indexing="ij")
    z = depth[vv, uu]

    mask = (z > 0.05) & (z < max_depth)
    if not _PC_LOG_ONCE["done"]:
        print(f"[pointcloud] first call: depth shape={depth.shape} K=\n{K}\n"
              f"  pos_w={pos}  quat_wxyz={quat_wxyz}  "
              f"z range=[{z.min():.3f},{z.max():.3f}]  n_valid={int(mask.sum())}/{mask.size}")
        _PC_LOG_ONCE["done"] = True
    if not np.any(mask):
        rr.log("world/head_cam/points", rr.Clear(recursive=False))
        return

    uu = uu[mask].astype(np.float32)
    vv = vv[mask].astype(np.float32)
    z  = z[mask].astype(np.float32)

    # OpenGL camera frame (USD default): +X right, +Y up, -Z forward.
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam =  (uu - cx) * z / fx
    y_cam = -(vv - cy) * z / fy
    z_cam = -z
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

    # Rotate into world via the camera's world quaternion (wxyz)
    w, qx, qy, qz = quat_wxyz
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*w),     2*(qx*qz + qy*w)],
        [2*(qx*qy + qz*w),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*w)],
        [2*(qx*qz - qy*w),     2*(qy*qz + qx*w),     1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float32)
    pts_world = pts_cam @ R.T + pos.astype(np.float32)

    colours = rgb[vv.astype(int), uu.astype(int)]  # (N, 3) uint8
    rr.log("world/head_cam/points", rr.Points3D(pts_world, colors=colours, radii=0.02))


# ── Rerun telemetry ───────────────────────────────────────────────────────────
def _rerun_init() -> None:
    """Spawn the Rerun viewer and set up the world coordinate frame (Z-up)."""
    rr.init("alex_onnx_walking_policy", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)


def _rerun_log(tick: int, robot, joint_map: dict, left_contact, right_contact,
               raw_action: np.ndarray, head_cam=None, chest_cam=None,
               obstacle_dist: float = float("inf"),
               emergency_dist: float = float("inf"),
               sam3_dets=None, sam3_rgb_shape=None) -> None:
    """Log one frame of telemetry to Rerun."""
    rr.set_time("tick", sequence=tick)
    rr.set_time("sim_time", duration=tick * SIM_DT * DECIMATION)

    pos  = robot.data.root_pos_w[0].cpu().numpy()   # (x, y, z)
    quat = robot.data.root_quat_w[0].cpu().numpy()  # (w, x, y, z)
    # Rerun transforms take (x, y, z, w) quaternion order
    rr.log("world/robot", rr.Transform3D(
        translation=pos,
        rotation=rr.Quaternion(xyzw=[quat[1], quat[2], quat[3], quat[0]]),
    ))
    # Draw a 0.3 m axis triad at the robot root (rerun 0.31 dropped Transform3D.axis_length)
    rr.log("world/robot/axes", rr.Arrows3D(
        vectors=[[0.3, 0, 0], [0, 0.3, 0], [0, 0, 0.3]],
        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    ))

    # Scalar plots: velocity command + robot Z height
    rr.log("cmd/vx",       rr.Scalars(float(_cmd[0])))
    rr.log("cmd/vy",       rr.Scalars(float(_cmd[1])))
    rr.log("cmd/yaw_rate", rr.Scalars(float(_cmd[2])))
    rr.log("cmd/standing", rr.Scalars(float(_cmd[3])))
    rr.log("robot/z",      rr.Scalars(float(pos[2])))

    # Foot contact forces (z component)
    l_fz = float(left_contact.data.net_forces_w[0, 0, 2])  if left_contact.data.net_forces_w  is not None else 0.0
    r_fz = float(right_contact.data.net_forces_w[0, 0, 2]) if right_contact.data.net_forces_w is not None else 0.0
    rr.log("contacts/left_fz",  rr.Scalars(l_fz))
    rr.log("contacts/right_fz", rr.Scalars(r_fz))

    # Joint positions: one scalar per joint, relative to home pose
    joint_pos = robot.data.joint_pos[0].cpu().numpy()
    for name, idx in joint_map.items():
        rr.log(f"joints/{name}", rr.Scalars(float(joint_pos[idx] - HOME_POS[name])))

    # Raw policy output magnitude
    rr.log("policy/action_abs_max", rr.Scalars(float(np.abs(raw_action).max())))

    # Head camera RGB + depth — only when available and freshly updated this tick
    if head_cam is not None and tick % CAMERA_DECIMATION == 0:
        try:
            output = head_cam.data.output
            if output is not None:
                if "rgb" in output:
                    rgb = output["rgb"][0].cpu().numpy()
                    if rgb.shape[-1] == 4:
                        rgb = rgb[..., :3]
                    rr.log("camera/head/rgb", rr.Image(rgb))

                    if _yolo_model is not None:
                        _kw = {"conf": args_cli.yolo_conf, "verbose": False}
                        if _yolo_imgsz is not None:
                            _kw["imgsz"] = _yolo_imgsz
                        res = _yolo_model.predict(rgb, **_kw)[0]
                        if res.boxes is not None and len(res.boxes) > 0:
                            xyxy   = res.boxes.xyxy.cpu().numpy()
                            cls    = res.boxes.cls.cpu().numpy().astype(int)
                            conf   = res.boxes.conf.cpu().numpy()
                            labels = [f"{res.names[c]} {p:.2f}" for c, p in zip(cls, conf)]
                            rr.log("camera/head/rgb/yolo", rr.Boxes2D(
                                array=xyxy,
                                array_format=rr.Box2DFormat.XYXY,
                                labels=labels,
                                class_ids=cls.tolist(),
                            ))
                        else:
                            rr.log("camera/head/rgb/yolo", rr.Clear(recursive=False))

                    if _sam3_model is not None:
                        # Reuse the detections produced by the autonomy
                        # perception step instead of re-running SAM3 on
                        # the same frame. ``sam3_dets`` is None on ticks
                        # before the first perception update; the
                        # overlay is silently skipped in that window.
                        _log_sam3(sam3_dets, sam3_rgb_shape)

                if "distance_to_image_plane" in output:
                    depth = output["distance_to_image_plane"][0].cpu().numpy().astype(np.float32)
                    if depth.ndim == 3:
                        depth = depth[..., 0]
                    # Isaac returns inf for pixels beyond clip range — clamp to clip far (20 m)
                    depth = np.where(np.isfinite(depth), depth, 20.0)
                    rr.log("camera/head/depth", rr.DepthImage(depth, meter=1.0))
        except Exception as e:
            if tick == 0:
                print(f"[rerun] head camera not ready yet: {e}")

        # Chest camera depth — Phase-3 obstacle source. Same camera-rate gate
        # as the head cam (CAMERA_DECIMATION). Logged with the obstacle_dist
        # scalar so the cone reading is visible alongside what it sees.
        if chest_cam is not None:
            try:
                c_out = chest_cam.data.output
                if c_out is not None and "distance_to_image_plane" in c_out:
                    c_depth = c_out["distance_to_image_plane"][0].cpu().numpy().astype(np.float32)
                    if c_depth.ndim == 3:
                        c_depth = c_depth[..., 0]
                    c_depth = np.where(np.isfinite(c_depth), c_depth, 10.0)
                    rr.log("camera/chest/depth", rr.DepthImage(c_depth, meter=1.0))
            except Exception as e:
                if tick == 0:
                    print(f"[rerun] chest camera not ready yet: {e}")

        # Forward-cone distance for the emergency brake. Plot against
        # the brake threshold so it's obvious when/why the brake fires.
        if np.isfinite(obstacle_dist):
            rr.log("autonomy/obstacle_dist", rr.Scalars(float(obstacle_dist)))
        if np.isfinite(emergency_dist):
            rr.log("autonomy/emergency_dist", rr.Scalars(float(emergency_dist)))

        # Point cloud log — outside the outer try/except so errors surface.
        if args_cli.pointcloud:
            out2 = head_cam.data.output if head_cam is not None else None
            if out2 is not None and "rgb" in out2 and "distance_to_image_plane" in out2:
                _rgb = out2["rgb"][0].cpu().numpy()
                if _rgb.shape[-1] == 4:
                    _rgb = _rgb[..., :3]
                _depth = out2["distance_to_image_plane"][0].cpu().numpy().astype(np.float32)
                if _depth.ndim == 3:
                    _depth = _depth[..., 0]
                _depth = np.where(np.isfinite(_depth), _depth, 20.0)
                _log_head_pointcloud(head_cam, _depth, _rgb, robot,
                                     stride=args_cli.pointcloud_stride,
                                     max_depth=args_cli.pointcloud_max_depth)
            elif tick == 0:
                print(f"[pointcloud] camera output missing. head_cam={head_cam}, keys={list(out2.keys()) if out2 else None}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global _cmd

    # Initialise command from CLI args
    _cmd[0] = args_cli.vx
    _cmd[1] = args_cli.vy
    _cmd[2] = args_cli.yaw
    _cmd[3] = 1.0 if args_cli.standing else 0.0

    print(f"Loading ONNX model: {ONNX_PATH}")
    assert ONNX_PATH.exists(), f"ONNX not found: {ONNX_PATH}"
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    print(f"  ONNX input:  {input_name}  shape: {sess.get_inputs()[0].shape}")
    print(f"  ONNX output: {sess.get_outputs()[0].name}  shape: {sess.get_outputs()[0].shape}")
    print(f"  obs_size={OBS_SIZE}  action_scale={ACTION_SCALE}")
    print(f"  initial cmd: vx={_cmd[0]:.2f} vy={_cmd[1]:.2f} yaw={_cmd[2]:.2f} standing={int(_cmd[3])}")

    sim, robot, left_contact, right_contact, head_cam, chest_cam = setup_scene(args_cli.scene)
    sim.reset()

    # Cap render FPS to 60 at runtime — prevents "fast forward" visual effect.
    # Physics runs at fixed 50 Hz regardless; this only affects the display update rate.
    import carb
    carb.settings.get_settings().set("/app/runLoops/main/rateLimitFrequency", 60)
    carb.settings.get_settings().set("/app/runLoops/rendering/rateLimitFrequency", 60)
    carb.settings.get_settings().set("/app/runLoops/main/syncToPresent", False)

    left_contact.reset()
    right_contact.reset()

    joint_map = build_joint_map(robot)
    print(f"Joint map built: {len(joint_map)}/23 joints found")

    kb = _make_keyboard(sim_device="cpu")
    print(kb)   # prints key binding summary

    if args_cli.rerun:
        _rerun_init()
        print("[rerun] Viewer spawned — streaming telemetry")

    autonomy_bundle = _build_autonomy_bundle()
    # Stash for atexit-style scene-graph save (Phase 2). Works even on Ctrl+C
    # or simulation crash — the atexit hook below picks up whatever's been
    # accumulated.
    global _autonomy_bundle_for_save
    _autonomy_bundle_for_save = autonomy_bundle
    # Phase-2 debug: let _step_perception print robot pose alongside the
    # locked goal so we can sanity-check geometry without re-instrumenting.
    if autonomy_bundle is not None:
        autonomy_bundle["_robot_for_debug"] = robot

    # Manual override: any keyboard press pauses autonomy for 1 s. Lets the
    # user wrest control without fighting the FSM.
    manual_override_until = 0.0

    last_action = np.zeros(23, dtype=np.float32)  # zero-init, matches training episode start
    tick = 0
    last_print = time.time()

    print("\nStarting policy loop. Close the viewer window to exit.")

    while simulation_app.is_running():
        robot.update(SIM_DT)

        # Choose command source: autonomy FSM or keyboard.
        # Manual override: peek the keyboard *before* deciding so any human
        # input pauses autonomy for ~1 s.
        kb_vel = kb.advance().numpy()       # (vx, vy, yaw)
        kb_active = bool(np.any(np.abs(kb_vel) > 1e-6))
        if kb_active:
            manual_override_until = time.time() + 1.0

        use_auto = (
            autonomy_bundle is not None
            and time.time() > manual_override_until
        )

        if use_auto:
            _step_autonomy(autonomy_bundle, robot)
        else:
            # Fall back to keyboard. Reuse the keyboard reading we just took
            # rather than calling kb.advance() again.
            _cmd[0] = float(kb_vel[0])
            _cmd[1] = float(kb_vel[1])
            _cmd[2] = float(kb_vel[2])
            # _cmd[3] (standing_flag) is toggled by the S callback only.

        # Build observation
        obs = build_obs(robot, joint_map, last_action)

        # Run ONNX policy
        with _TIMINGS.time("onnx.policy_run"):
            raw_action = sess.run(None, {input_name: obs[np.newaxis, :]})[0][0]  # (23,)
        last_action = raw_action.copy()

        # Apply to Isaac
        apply_policy(robot, joint_map, raw_action)

        # Step physics (4 substeps); only render on the last substep to reduce
        # display compositor load on high-resolution monitors.
        with _TIMINGS.time("physics.step"):
            for i in range(DECIMATION):
                sim.step(render=(i == DECIMATION - 1))
                robot.update(SIM_DT)
                left_contact.update(SIM_DT)
                right_contact.update(SIM_DT)

        if head_cam is not None and tick % CAMERA_DECIMATION == 0:
            with _TIMINGS.time("camera.update"):
                head_cam.update(SIM_DT)
                if chest_cam is not None:
                    chest_cam.update(SIM_DT)
            # Phase-2 perception: SAM3 → SceneGraph → goal update. Runs at
            # camera rate (~12.5 Hz). The next tick's _step_autonomy reads
            # the freshly-updated GoalState; one-tick lag is acceptable.
            # Phase-3 chest-cam depth feeds forward_cone_distance.
            with _TIMINGS.time("autonomy.step_perception"):
                _step_perception(autonomy_bundle, head_cam, chest_cam, tick)

        if args_cli.rerun:
            _ob_d = autonomy_bundle.get("obstacle_dist", float("inf")) if autonomy_bundle else float("inf")
            _em_d = autonomy_bundle.get("emergency_dist", float("inf")) if autonomy_bundle else float("inf")
            _dets = autonomy_bundle.get("sam3_dets") if autonomy_bundle else None
            _rgb_shape = autonomy_bundle.get("sam3_rgb_shape") if autonomy_bundle else None
            _rerun_log(tick, robot, joint_map, left_contact, right_contact, raw_action,
                       head_cam, chest_cam, obstacle_dist=_ob_d, emergency_dist=_em_d,
                       sam3_dets=_dets, sam3_rgb_shape=_rgb_shape)

        # Per-second diagnostic print
        now = time.time()
        if now - last_print >= 1.0:
            root_x   = float(robot.data.root_pos_w[0, 0])
            root_y   = float(robot.data.root_pos_w[0, 1])
            root_z   = float(robot.data.root_pos_w[0, 2])
            l_hip_y  = float(robot.data.joint_pos[0, joint_map["LEFT_HIP_Y"]])
            r_hip_y  = float(robot.data.joint_pos[0, joint_map["RIGHT_HIP_Y"]])
            l_knee_y = float(robot.data.joint_pos[0, joint_map["LEFT_KNEE_Y"]])

            dev = robot.data.root_quat_w.device
            grav_world = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=dev)
            pg = quat_apply_inverse(robot.data.root_quat_w[0:1], grav_world)[0].cpu().numpy()

            l_fz = float(left_contact.data.net_forces_w[0, 0, 2]) if left_contact.data.net_forces_w is not None else 0.0
            r_fz = float(right_contact.data.net_forces_w[0, 0, 2]) if right_contact.data.net_forces_w is not None else 0.0

            standing_str = "STAND" if _cmd[3] > 0.5 else "WALK "
            # print(
            #     f"[tick {tick:5d}] [{standing_str}] cmd=({_cmd[0]:+.2f},{_cmd[1]:+.2f},{_cmd[2]:+.2f})  "
            #     f"pos=({root_x:+.2f},{root_y:+.2f},{root_z:.3f})m  "
            #     f"hipY L{l_hip_y:+.3f}/R{r_hip_y:+.3f}  "
            #     f"kneeY L{l_knee_y:+.3f}  "
            #     f"projGrav=({pg[0]:+.3f},{pg[1]:+.3f},{pg[2]:+.3f})  "
            #     f"Fz L{l_fz:+.0f}/R{r_fz:+.0f}N  "
            #     f"action_max={np.abs(raw_action).max():.4f}"
            # )

            if tick == 0:
                print(f"  obs[0:3]  base_ang_vel  = {obs[0:3]}")
                print(f"  obs[3:6]  proj_grav     = {obs[3:6]}")
                print(f"  obs[6:9]  base_velocity = {obs[6:9]}")
                print(f"  obs[9]    standing_flag = {obs[9]:.1f}")
                print(f"  obs[10]   base_height   = {obs[10]:.4f}")
                print(f"  obs[11:34] joint_pos_rel = {obs[11:34]}")
                print(f"  raw_action = {raw_action}")

            last_print = now

        tick += 1


# ── atexit fallback — save the scene graph on Ctrl+C / kill / clean exit ────
# Mirrors cam_room_explore_isaac.py pattern. Catches the common case where
# the user closes the Isaac viewer mid-run; we still get a JSON snapshot of
# whatever objects SAM3 had detected.
_autonomy_bundle_for_save: "dict | None" = None


def _save_scene_graph_on_exit() -> None:
    bundle = _autonomy_bundle_for_save
    if bundle is None or "scene_graph" not in bundle:
        return
    sg = bundle["scene_graph"]
    if not sg.objects:
        return  # nothing detected — skip the empty file
    try:
        sg.scan_complete = True
        serialize.save(sg, args_cli.scene_graph_path)
        print(f"[autonomy] scene graph saved → {args_cli.scene_graph_path}  "
              f"({len(sg.objects)} objects)")
    except Exception as e:
        print(f"[autonomy] scene graph save failed: {e}")


import atexit as _atexit   # noqa: E402


# ── End-of-run performance report ────────────────────────────────────────────
# atexit hooks fire **last-registered-first**, so we register the perf
# report *before* the scene-graph save and it lands at the bottom of the
# terminal output (most useful position).
_RUN_START_T = time.time()


_PERF_REPORT_PRINTED = {"done": False}


def _print_perf_report_on_exit() -> None:
    if _PERF_REPORT_PRINTED["done"]:
        return
    _PERF_REPORT_PRINTED["done"] = True
    elapsed = time.time() - _RUN_START_T
    print()
    print("=" * 64)
    print(f"  Run duration: {elapsed:.1f} s")
    print("=" * 64)
    print("  Timing breakdown:")
    print()
    print(format_timing_report(_TIMINGS.snapshot()))
    print("  Memory peaks:")
    print(format_memory_report())
    print("=" * 64)


_atexit.register(_print_perf_report_on_exit)
_atexit.register(_save_scene_graph_on_exit)


# ── Robust shutdown ──────────────────────────────────────────────────────────
# Isaac Sim's app launcher swallows the normal Python shutdown path on
# Ctrl+C (SIGINT) — atexit hooks don't always fire. We install three
# defenses:
#
#   1. SIGINT handler that prints the perf report and saves the scene
#      graph before the process exits.
#   2. ``try/finally`` around ``main()`` so a clean viewer close still
#      fires the report (the ``simulation_app.close()`` call below would
#      otherwise short-circuit ``atexit``).
#   3. The original ``atexit.register`` calls above stay as a final
#      safety net.
#
# All three call the same handlers; ``_REPORT_PRINTED`` makes them
# idempotent so you don't see the table three times.
_REPORT_PRINTED = {"done": False}


def _final_shutdown_print() -> None:
    """Idempotent wrapper: print the perf report at most once per run."""
    if _REPORT_PRINTED["done"]:
        return
    _REPORT_PRINTED["done"] = True
    try:
        _print_perf_report_on_exit()
    except Exception as e:                           # pragma: no cover
        print(f"[autonomy] perf report failed: {e}")
    try:
        _save_scene_graph_on_exit()
    except Exception as e:                           # pragma: no cover
        print(f"[autonomy] scene graph save failed: {e}")


def _sigint_handler(signum, frame) -> None:          # pragma: no cover
    """Catch Ctrl+C, dump the report, and exit cleanly."""
    print("\n[autonomy] caught SIGINT — flushing perf report")
    _final_shutdown_print()
    # ``os._exit`` so we don't tangle with Isaac's own SIGINT handling.
    import os as _os
    _os._exit(0)


import signal as _signal                             # noqa: E402
_signal.signal(_signal.SIGINT, _sigint_handler)


if __name__ == "__main__":
    try:
        main()
    finally:
        _final_shutdown_print()
        try:
            simulation_app.close()
        except Exception:
            pass