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

# ── Isaac imports (after AppLauncher) ────────────────────────────────────────
import copy
import math
import pathlib
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
SPAWN_POS_ROOM = (1.0, -0.8, 0.93)

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

# ── Mutable command state (updated by keyboard at runtime) ────────────────────
# [vx, vy, yaw_rate, standing_flag]  — populated in main() from CLI args
_cmd = np.zeros(4, dtype=np.float32)   # [vx, vy, yaw, standing]


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

    return sim, robot, left_contact, right_contact, head_cam


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

    kb.add_callback("S", _toggle_standing)
    return kb


def _update_cmd_from_keyboard(kb: Se2Keyboard) -> None:
    """Pull latest velocity from keyboard and write into _cmd."""
    vel = kb.advance().numpy()   # [vx, vy, yaw]
    _cmd[0] = float(vel[0])
    _cmd[1] = float(vel[1])
    _cmd[2] = float(vel[2])
    # _cmd[3] (standing_flag) is toggled by the S callback — leave it alone here


_PC_LOG_ONCE = {"done": False}

def _get_head_cam_world_pose(head_cam) -> "tuple[np.ndarray, np.ndarray]":
    """Read the head camera's world transform directly from the USD prim.

    Returns (pos_xyz, quat_wxyz). The quaternion is in USD/OpenGL convention:
    camera +X right, +Y up, -Z forward (same convention CameraCfg was created with).
    """
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(head_cam.cfg.prim_path)
    xform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    t = xform.ExtractTranslation()
    r = xform.ExtractRotationQuat()
    img = r.GetImaginary()
    pos  = np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float32)
    quat = np.array([float(r.GetReal()), float(img[0]), float(img[1]), float(img[2])],
                    dtype=np.float32)
    return pos, quat


_SAM3_WARN_ONCE = {"done": False}

def _log_sam3(rgb: np.ndarray) -> None:
    """Run SAM3 text-prompted segmentation on the head camera RGB and log to Rerun.

    Logs one set of Boxes2D + SegmentationImage per prompt (merged into a single
    mask layer with per-prompt class ids so Rerun draws distinct colours).
    """
    from PIL import Image as _PIL

    try:
        img = _PIL.fromarray(rgb.astype(np.uint8))
        state = _sam3_processor.set_image(img)

        all_boxes  = []
        all_labels = []
        all_scores = []
        all_class_ids = []
        mask_layer = np.zeros(rgb.shape[:2], dtype=np.uint16)

        for cls_id, prompt in enumerate(_sam3_prompts, start=1):
            out = _sam3_processor.set_text_prompt(state=state, prompt=prompt)
            masks  = out.get("masks")
            boxes  = out.get("boxes")
            scores = out.get("scores")
            if masks is None or len(masks) == 0:
                continue
            for i, s in enumerate(scores):
                s = float(s)
                if s < args_cli.sam3_conf:
                    continue
                m = masks[i]
                if hasattr(m, "cpu"):
                    m = m.cpu().numpy()
                m = np.asarray(m)
                if m.ndim == 3:
                    m = m[0]
                mask_layer[m > 0.5] = cls_id

                b = boxes[i]
                if hasattr(b, "cpu"):
                    b = b.cpu().numpy()
                all_boxes.append(np.asarray(b).reshape(-1)[:4])
                all_labels.append(f"{prompt} {s:.2f}")
                all_scores.append(s)
                all_class_ids.append(cls_id)

        rr.log("camera/head/rgb/sam3_mask",
               rr.SegmentationImage(mask_layer))
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
            print(f"[sam3] inference failed: {e}")
            _SAM3_WARN_ONCE["done"] = True


def _log_head_pointcloud(head_cam, depth: np.ndarray, rgb: np.ndarray,
                         robot, stride: int, max_depth: float) -> None:
    """Unproject head-camera depth into world-frame coloured points and log to Rerun."""
    try:
        K = head_cam.data.intrinsic_matrices[0].cpu().numpy()  # (3,3)
        pos, quat_wxyz = _get_head_cam_world_pose(head_cam)    # opengl-convention
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
               raw_action: np.ndarray, head_cam=None) -> None:
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
                        _log_sam3(rgb)

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

    sim, robot, left_contact, right_contact, head_cam = setup_scene(args_cli.scene)
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

    last_action = np.zeros(23, dtype=np.float32)  # zero-init, matches training episode start
    tick = 0
    last_print = time.time()

    print("\nStarting policy loop. Close the viewer window to exit.")

    while simulation_app.is_running():
        robot.update(SIM_DT)

        # Update velocity command from keyboard
        _update_cmd_from_keyboard(kb)

        # Build observation
        obs = build_obs(robot, joint_map, last_action)

        # Run ONNX policy
        raw_action = sess.run(None, {input_name: obs[np.newaxis, :]})[0][0]  # (23,)
        last_action = raw_action.copy()

        # Apply to Isaac
        apply_policy(robot, joint_map, raw_action)

        # Step physics (4 substeps); only render on the last substep to reduce
        # display compositor load on high-resolution monitors.
        for i in range(DECIMATION):
            sim.step(render=(i == DECIMATION - 1))
            robot.update(SIM_DT)
            left_contact.update(SIM_DT)
            right_contact.update(SIM_DT)

        if head_cam is not None and tick % CAMERA_DECIMATION == 0:
            head_cam.update(SIM_DT)

        if args_cli.rerun:
            _rerun_log(tick, robot, joint_map, left_contact, right_contact, raw_action, head_cam)

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


if __name__ == "__main__":
    main()
    simulation_app.close()