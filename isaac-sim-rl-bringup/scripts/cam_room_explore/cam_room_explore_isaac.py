"""
cam_room_explore_isaac.py
-------------------------
Isaac Sim equivalent of demos/cam_room_explore: drive a simple free-flying
camera through the ithor FloorPlan1 scene with keyboard controls, stream head
camera RGB/depth + SAM3 detections to Rerun.

No walking policy, no articulation — just a moveable camera prim.  This is the
"camera robot" baseline that prompt-driven autonomy can be built on top of
before we add it to the Alex ONNX policy flow.

Run with:
    cd ~/alex/repository-group/IsaacLab
    ./isaaclab.sh -p .../cam_room_explore_isaac.py \
        --rerun --enable_cameras --sam3 "oven, door, chair, bottle"

Keyboard:
    Arrow Up / Down         — translate forward / back (cam-local)
    Arrow Left / Right      — yaw left / right
    Q / E                   — strafe left / right (cam-local)
    R / F                   — rise / lower (world Z)
    L                       — reset pose to spawn

CLI:
    --scene               ground | room           (default: room)
    --spawn-x / -y / -z   camera spawn (m)
    --spawn-yaw-deg       initial yaw (degrees, 0 = +X world)
    --move-speed          m/s for translate       (default 0.6)
    --turn-speed-deg      deg/s for yaw           (default 60)
    --rerun               stream telemetry + camera to Rerun viewer
    --sam3 "a, b, c"      SAM3 text prompts
    --sam3-conf FLOAT     SAM3 score threshold    (default 0.3)
    --pointcloud          project depth to world-frame point cloud
"""

# ── Hydra + Isaac AppLauncher sequencing ─────────────────────────────────────
# Isaac's AppLauncher parses --headless / --enable_cameras / --device via argparse,
# while all experiment config comes from Hydra. We split sys.argv: everything
# that looks like a Hydra override (key=value, @group=choice, +group.key=value)
# goes to Hydra's compose API; everything with a leading dash stays for argparse.
import sys as _sys
import pathlib as _pathlib

_HYDRA_TOKEN_CHARS = ("=",)  # Hydra overrides always contain '='
_argparse_argv = [_sys.argv[0]]
_hydra_overrides = []
for _arg in _sys.argv[1:]:
    if any(c in _arg for c in _HYDRA_TOKEN_CHARS) and not _arg.startswith("-"):
        _hydra_overrides.append(_arg)
    else:
        _argparse_argv.append(_arg)
_sys.argv = _argparse_argv  # AppLauncher argparse now sees only Isaac flags

import argparse
from isaaclab.app import AppLauncher

_iparser = argparse.ArgumentParser(description="Isaac Sim camera-robot room explorer "
                                               "(Hydra-configured — see configs/)")
AppLauncher.add_app_launcher_args(_iparser)
_isaac_args = _iparser.parse_args()
app_launcher = AppLauncher(_isaac_args)
simulation_app = app_launcher.app

# Now compose Hydra config (must happen AFTER AppLauncher to avoid conflicts).
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import sys as _sys2
# Make configs/ importable so `schema.register()` works.
# __file__ = .../isaac-sim-rl-bringup/scripts/cam_room_explore/cam_room_explore_isaac.py
# parents: [0] cam_room_explore  [1] scripts  [2] isaac-sim-rl-bringup  [3] Alex-robot
_CONFIGS_DIR = str(_pathlib.Path(__file__).resolve().parents[2] / "configs")
if _CONFIGS_DIR not in _sys2.path:
    _sys2.path.insert(0, _CONFIGS_DIR)
import schema as _schema   # noqa: E402 — structured-config dataclasses
_schema.register()

# Resolve repo_root from this file's location so ${repo_root} interpolations work.
_REPO_ROOT = str(_pathlib.Path(__file__).resolve().parents[3])
_hydra_overrides.insert(0, f"repo_root={_REPO_ROOT}")

with initialize_config_dir(version_base=None, config_dir=_CONFIGS_DIR):
    cfg = compose(config_name="cam_room_explore", overrides=_hydra_overrides)

print("[hydra] Composed config:")
print(OmegaConf.to_yaml(cfg))


# ── Config × AppLauncher consistency checks ──────────────────────────────────
# Rerun + head-camera sensors require Isaac's RTX offscreen renderer, which
# requires `--enable_cameras`. The two CLI layers don't know about each other,
# so we fail fast here instead of deep inside sim.reset().
_errs = []
_needs_cameras = (
    cfg.rerun.enabled or cfg.detector.enabled or cfg.rerun.pointcloud
)
if _needs_cameras and not getattr(_isaac_args, "enable_cameras", False):
    _reasons = []
    if cfg.rerun.enabled:    _reasons.append("rerun.enabled=true")
    if cfg.detector.enabled: _reasons.append("detector.enabled=true")
    if cfg.rerun.pointcloud: _reasons.append("rerun.pointcloud=true")
    _errs.append(
        f"Config requires head camera ({', '.join(_reasons)}) but Isaac was "
        f"started without --enable_cameras. Add --enable_cameras to the "
        f"command line."
    )

if cfg.detector.enabled and not cfg.rerun.enabled:
    _errs.append(
        "detector.enabled=true requires rerun.enabled=true (SAM3 masks log "
        "to Rerun). Add rerun=full or rerun.enabled=true."
    )

if cfg.autonomy.mode == "approach" and cfg.autonomy.target in (None, ""):
    _errs.append(
        "autonomy=approach requires autonomy.target=<label>. "
        "Example: autonomy=approach autonomy.target=sofa"
    )

if _errs:
    print("\n[config error] Config / flag mismatch:")
    for e in _errs:
        print(f"  - {e}")
    print("Aborting before Isaac starts. Fix the command and rerun.\n")
    simulation_app.close()
    import sys
    sys.exit(2)


# ── Shim: keep the existing `args_cli.X` downstream interface working ────────
# The body of this script was written against an argparse Namespace with flat
# keys (`args_cli.prompt_stop_dist`, `args_cli.sam3`, etc.). Rather than rename
# ~40 references, we build an equivalent Namespace from the Hydra config here.
class _Shim:
    pass
args_cli = _Shim()

# Scene + doors
args_cli.scene          = cfg.scene.name
args_cli.doors          = cfg.scene.doors

# Spawn
args_cli.spawn_x        = cfg.spawn.x
args_cli.spawn_y        = cfg.spawn.y
args_cli.spawn_z        = cfg.spawn.z
args_cli.spawn_yaw_deg  = cfg.spawn.yaw_deg

# Motion
args_cli.move_speed     = cfg.motion.move_speed
args_cli.turn_speed_deg = cfg.motion.turn_speed_deg

# Rerun + point cloud
args_cli.rerun                  = cfg.rerun.enabled
args_cli.pointcloud             = cfg.rerun.pointcloud
args_cli.pointcloud_stride      = cfg.rerun.pointcloud_stride
args_cli.pointcloud_max_depth   = cfg.rerun.pointcloud_max_depth

# Detector (SAM3). The old code used `--sam3` as the prompts string and
# required `--rerun`; map both.
args_cli.sam3      = cfg.detector.prompts if cfg.detector.enabled else None
args_cli.sam3_conf = cfg.detector.conf

# Autonomy
_auto = cfg.autonomy
args_cli.prompt               = _auto.target if _auto.mode == "approach" else None
args_cli.prompt_stop_dist     = _auto.stop_dist
args_cli.prompt_walk_speed    = _auto.walk_speed
args_cli.prompt_search_yaw    = _auto.search_yaw
args_cli.prompt_heading_kp    = _auto.heading_kp
args_cli.prompt_stale_s       = _auto.stale_s
args_cli.prompt_lock_conf     = _auto.lock_conf
args_cli.prompt_tilt_deg      = _auto.tilt_deg
args_cli.prompt_tilt_period_s = _auto.tilt_period_s

# Scene graph output path
args_cli.scene_graph_path = cfg.output.scene_graph_path
args_cli.save_every_s     = cfg.output.save_every_s

# ── Imports after AppLauncher ────────────────────────────────────────────────
import math
import pathlib
import time

import numpy as np

if args_cli.rerun:
    import rerun as rr

# SAM3 (optional)
_sam3_model = None
_sam3_processor = None
_sam3_prompts: list = []
if args_cli.prompt and not args_cli.sam3:
    # Approach needs SAM3 to detect the target; auto-enable with prompt as prompts.
    args_cli.sam3 = args_cli.prompt

if args_cli.sam3:
    assert args_cli.rerun, "detector=sam3 requires rerun=full"
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    print(f"[sam3] Loading image model …")
    _sam3_model = build_sam3_image_model().to("cuda").eval()
    _sam3_processor = Sam3Processor(_sam3_model)
    _sam3_prompts = [p.strip() for p in args_cli.sam3.split(",") if p.strip()]
    # Ensure --prompt is in the SAM3 list so we can detect it.
    if args_cli.prompt and args_cli.prompt not in _sam3_prompts:
        _sam3_prompts.append(args_cli.prompt)
    print(f"[sam3] Ready. Prompts: {_sam3_prompts}  conf≥{args_cli.sam3_conf}")

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim import SimulationContext

# ── Paths ─────────────────────────────────────────────────────────────────────
_ALEX_ROBOT = pathlib.Path(__file__).resolve().parents[3]
_ROOM_USD = _ALEX_ROBOT / "assets" / "usd" / "scenes" / "ithor" / "FloorPlan1_physics" / "scene.usda"
_HALLWAY_USD = _ALEX_ROBOT / "scenes" / "HallwayScene" / "Hallway.usdc"

SIM_DT = 0.01  # 100 Hz — no policy, just move the camera
CAMERA_DECIMATION = 4   # camera update every 4 ticks = 25 Hz
CAM_W, CAM_H = 640, 480


# ── Camera + world pose helpers ──────────────────────────────────────────────
def _yaw_to_quat_wxyz(yaw_rad: float) -> tuple:
    """Rotation about world Z by yaw_rad, returned as wxyz (no conversion)."""
    c, s = math.cos(yaw_rad / 2.0), math.sin(yaw_rad / 2.0)
    return (c, 0.0, 0.0, s)


def _yaw_pitch_to_quat_wxyz(yaw_rad: float, pitch_rad: float) -> tuple:
    """ZY-intrinsic rotation in world convention (+X forward, +Y left, +Z up).
    First yaw about world Z, then pitch about local Y. Sign chosen so
    `pitch_rad > 0` tilts the camera UP (aircraft nose-up convention), matching
    the chase-camera target_z = z + sin(pitch). Returned as wxyz."""
    cy, sy = math.cos(yaw_rad / 2.0), math.sin(yaw_rad / 2.0)
    p = -pitch_rad   # negate so +pitch = look up in world convention
    cp, sp = math.cos(p / 2.0), math.sin(p / 2.0)
    # q = q_yaw * q_pitch  with q_yaw = (cy, 0, 0, sy), q_pitch = (cp, 0, sp, 0)
    w = cy * cp
    x = -sy * sp
    y = cy * sp
    z = sy * cp
    return (w, x, y, z)


def _get_cam_world_pose(cam) -> "tuple[np.ndarray, np.ndarray]":
    """Read head camera world transform from the USD prim. Returns (pos, quat_wxyz)."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(cam.cfg.prim_path)
    xform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    t = xform.ExtractTranslation()
    r = xform.ExtractRotationQuat()
    img = r.GetImaginary()
    pos  = np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float32)
    quat = np.array([float(r.GetReal()), float(img[0]), float(img[1]), float(img[2])],
                    dtype=np.float32)
    return pos, quat


# ── Scene setup ───────────────────────────────────────────────────────────────
def setup_scene():
    sim_cfg = sim_utils.SimulationCfg(dt=SIM_DT, device="cpu")
    sim = SimulationContext(sim_cfg)

    # World lighting
    sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=2000.0)
    )

    if args_cli.scene == "room":
        assert _ROOM_USD.exists(), (
            f"Room USD not found: {_ROOM_USD}\n"
            "Run: ms-download --type usd --install-dir assets/usd --scenes ithor"
        )
        sim_utils.UsdFileCfg(usd_path=str(_ROOM_USD)).func(
            "/World/Room", sim_utils.UsdFileCfg(usd_path=str(_ROOM_USD))
        )
        print(f"[scene] Loaded FloorPlan1 room from {_ROOM_USD}")
    elif args_cli.scene == "hallway":
        assert _HALLWAY_USD.exists(), f"Hallway USD not found: {_HALLWAY_USD}"
        # Hallway.usdc has no defaultPrim (multiple root prims), so UsdFileCfg
        # reference won't work. Add it as a sublayer instead — all root prims
        # appear directly on the stage.
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        root_layer = stage.GetRootLayer()
        root_layer.subLayerPaths.append(str(_HALLWAY_USD))
        print(f"[scene] Loaded Hallway as sublayer from {_HALLWAY_USD}")
        # Configure all doors per --doors flag.
        #   open   → angular drive target = -90° (PhysX motors them open)
        #   closed → angular drive target =   0° (closed, must be pushed open)
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
    else:
        sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
        print("[scene] Loaded groundplane")

    sim.set_camera_view(eye=[args_cli.spawn_x + 2.0,
                             args_cli.spawn_y + 2.0,
                             args_cli.spawn_z + 0.5],
                        target=[args_cli.spawn_x, args_cli.spawn_y, args_cli.spawn_z])

    # Camera prim — attached to /World/Robot (a plain Xform) so we can move it
    # by writing the xform transform each frame.
    import isaacsim.core.utils.prims as prims_utils
    prims_utils.create_prim("/World/Robot", "Xform",
                            translation=(args_cli.spawn_x, args_cli.spawn_y, args_cli.spawn_z),
                            orientation=_yaw_to_quat_wxyz(math.radians(args_cli.spawn_yaw_deg)))

    cam_cfg = CameraCfg(
        prim_path="/World/Robot/Camera",
        update_period=SIM_DT * CAMERA_DECIMATION,
        height=CAM_H,
        width=CAM_W,
        data_types=["rgb", "distance_to_image_plane"],
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=3.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 20.0),
        ),
        # Use "world" convention so the camera's local forward = parent Xform's +X.
        # Identity rotation keeps the camera upright and looking along robot yaw.
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
    )
    cam = Camera(cam_cfg)
    print("[cameras] Created head camera at /World/Robot/Camera")
    return sim, cam


# ── Camera (prim) pose writer ────────────────────────────────────────────────
class CameraRobot:
    """Tiny state wrapper: holds (x, y, z, yaw, pitch) and writes the xform each tick.
    Pitch > 0 tilts the camera up."""

    def __init__(self, x, y, z, yaw_rad, pitch_rad=0.0):
        self.x, self.y, self.z = x, y, z
        self.yaw, self.pitch = yaw_rad, pitch_rad
        self._init = (x, y, z, yaw_rad, pitch_rad)

    def translate_local(self, dx_forward: float, dy_left: float, dz_world: float):
        """Move in camera-local axes (forward/left) plus world Z."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.x += dx_forward * c - dy_left * s
        self.y += dx_forward * s + dy_left * c
        self.z += dz_world

    def rotate_yaw(self, dyaw_rad: float):
        self.yaw += dyaw_rad

    def set_pitch(self, pitch_rad: float):
        self.pitch = pitch_rad

    def reset(self):
        self.x, self.y, self.z, self.yaw, self.pitch = self._init

    def write_to_stage(self, prim_path: str = "/World/Robot"):
        """Update the translate + orient xform ops. Creates them on first call
        (using double precision) and reuses them afterwards."""
        import omni.usd
        from pxr import UsdGeom, Gf
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        xform = UsdGeom.Xformable(prim)

        t_op, r_op = None, None
        for op in xform.GetOrderedXformOps():
            n = op.GetOpName()
            if n.endswith("translate"): t_op = op
            elif n.endswith("orient"):  r_op = op
        if t_op is None:
            t_op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        if r_op is None:
            r_op = xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)

        t_op.Set(Gf.Vec3d(self.x, self.y, self.z))
        w, qx, qy, qz = _yaw_pitch_to_quat_wxyz(self.yaw, self.pitch)
        r_op.Set(Gf.Quatd(w, qx, qy, qz))


# ── Keyboard handling (carb raw input) ───────────────────────────────────────
class KeyboardState:
    """Polls carb's keyboard subscription. Independent of Se2Keyboard (which
    assumes an articulation). Uses raw key down/up events."""

    def __init__(self):
        import carb.input
        import omni.appwindow
        self._keys_down: set = set()
        self._reset_requested = False

        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_key)

    def _on_key(self, event):
        import carb.input
        name = event.input.name
        if event.type == carb.input.KeyboardEventType.KEY_PRESS or \
           event.type == carb.input.KeyboardEventType.KEY_REPEAT:
            self._keys_down.add(name)
            if name == "L":
                self._reset_requested = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._keys_down.discard(name)
        return True

    def held(self, name: str) -> bool:
        return name in self._keys_down

    def consume_reset(self) -> bool:
        v = self._reset_requested
        self._reset_requested = False
        return v


def _apply_keyboard(kb: "KeyboardState", robot: CameraRobot, dt: float):
    v = args_cli.move_speed * dt
    w = math.radians(args_cli.turn_speed_deg) * dt
    if kb.held("UP"):         robot.translate_local(+v, 0, 0)
    if kb.held("DOWN"):       robot.translate_local(-v, 0, 0)
    if kb.held("LEFT"):       robot.rotate_yaw(+w)
    if kb.held("RIGHT"):      robot.rotate_yaw(-w)
    if kb.held("Q"):          robot.translate_local(0, +v, 0)
    if kb.held("E"):          robot.translate_local(0, -v, 0)
    if kb.held("R"):          robot.translate_local(0, 0, +v)
    if kb.held("F"):          robot.translate_local(0, 0, -v)
    if kb.consume_reset():
        robot.reset()
        _ctrl_state["tilt_phase_rad"] = 0.0
        _goal_state["xyz"] = None
        _goal_state["locked"] = False
        print("[keyboard] reset → spawn pose (goal unlatched)")


# ── Rerun logging ────────────────────────────────────────────────────────────
def _rerun_init():
    rr.init("cam_room_explore_isaac", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)


_SAM3_WARN_ONCE = {"done": False}
# Shared goal state updated by _log_sam3 when it detects args_cli.prompt.
# (x, y, z, wall_time_seconds). None = not yet seen / stale.
_goal_state: dict = {"xyz": None, "t": 0.0, "score": 0.0, "locked": False}


# ── Scene graph ───────────────────────────────────────────────────────────────
import json as _json

_MERGE_DIST = 0.5  # metres — merge detections within this radius


class SceneGraph:
    """Persistent, deduplicated record of every object detected during the run.

    Each entry is identified by `label_N` (e.g. `oven_1`, `door_2`). Detections
    within `_MERGE_DIST` of an existing entry with the same label update that
    entry (averaging the position, keeping the best score). New detections
    beyond that radius create a new entry.

    Logged to Rerun as persistent labelled spheres at `world/scene/<id>` and
    saved to JSON on demand.
    """

    def __init__(self):
        self.objects: dict[str, dict] = {}
        self._label_counts: dict[str, int] = {}
        self.robot_path: list = []
        self.scan_complete: bool = False

    def _next_id(self, label: str) -> str:
        n = self._label_counts.get(label, 0) + 1
        self._label_counts[label] = n
        return f"{label}_{n}"

    def _find_nearby(self, label: str, xyz: np.ndarray) -> "str | None":
        for obj_id, obj in self.objects.items():
            if not obj_id.rsplit("_", 1)[0] == label:
                continue
            dist = np.linalg.norm(np.array(obj["position_xyz"]) - xyz)
            if dist < _MERGE_DIST:
                return obj_id
        return None

    def update(self, label: str, xyz: np.ndarray, score: float, tick: int,
               bbox_area_px: float = 0.0) -> str:
        """Insert or merge a detection. Returns the object id."""
        existing = self._find_nearby(label, xyz)
        if existing is not None:
            obj = self.objects[existing]
            # Running average of position, keep best score
            n = obj.get("_n_obs", 1)
            old_pos = np.array(obj["position_xyz"])
            obj["position_xyz"] = ((old_pos * n + xyz) / (n + 1)).tolist()
            obj["_n_obs"] = n + 1
            obj["confidence"] = max(obj["confidence"], score)
            obj["last_seen_tick"] = tick
            obj["bbox_area_px"] = max(obj["bbox_area_px"], bbox_area_px)
            return existing

        obj_id = self._next_id(label)
        self.objects[obj_id] = {
            "label": label,
            "position_xyz": xyz.tolist(),
            "confidence": score,
            "first_seen_tick": tick,
            "last_seen_tick": tick,
            "bbox_area_px": bbox_area_px,
            "_n_obs": 1,
        }
        return obj_id

    def record_pose(self, x: float, y: float, yaw_deg: float):
        self.robot_path.append([round(x, 3), round(y, 3), round(yaw_deg, 1)])

    def log_to_rerun(self):
        """Log every object as a persistent labelled sphere in Rerun."""
        if not args_cli.rerun:
            return
        for obj_id, obj in self.objects.items():
            pos = obj["position_xyz"]
            rr.log(f"world/scene/{obj_id}", rr.Points3D(
                [pos],
                colors=[[255, 200, 0]],
                radii=0.08,
                labels=[f"{obj_id} ({obj['confidence']:.2f})"],
            ))

    def to_dict(self) -> dict:
        """Clean JSON-serialisable dict (strip internal keys)."""
        objs = {}
        for obj_id, obj in self.objects.items():
            objs[obj_id] = {k: v for k, v in obj.items() if not k.startswith("_")}
        return {
            "room_id": "FloorPlan1",
            "scan_complete": self.scan_complete,
            "objects": objs,
            "robot_path": self.robot_path,
            "n_objects": len(objs),
            "object_labels": sorted(set(o["label"] for o in objs.values())),
        }

    def save(self, path: str = "scene_graph.json"):
        p = pathlib.Path(path)
        p.write_text(_json.dumps(self.to_dict(), indent=2))
        print(f"[scene_graph] saved {len(self.objects)} objects → {p}")

    def summary(self) -> str:
        labels = {}
        for obj in self.objects.values():
            labels[obj["label"]] = labels.get(obj["label"], 0) + 1
        parts = [f"{v}× {k}" for k, v in sorted(labels.items())]
        return f"{len(self.objects)} objects: {', '.join(parts) if parts else '(none)'}"


_scene_graph = SceneGraph()


def _pixel_to_world(cam, depth: np.ndarray, u: int, v: int) -> "np.ndarray | None":
    """Unproject a single pixel (u, v) with depth[v, u] to world XYZ. Returns None
    if depth is invalid at that pixel."""
    if depth.ndim == 3:
        depth = depth[..., 0]
    H, W = depth.shape
    if not (0 <= u < W and 0 <= v < H):
        return None
    z = float(depth[v, u])
    if not np.isfinite(z) or z <= 0.05 or z > 20.0:
        return None
    try:
        K = cam.data.intrinsic_matrices[0].cpu().numpy()
        pos, quat_wxyz = _get_cam_world_pose(cam)
    except Exception:
        return None
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_c =  (u - cx) * z / fx
    y_c = -(v - cy) * z / fy
    z_c = -z
    w, qx, qy, qz = quat_wxyz
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*w),     2*(qx*qz + qy*w)],
        [2*(qx*qy + qz*w),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*w)],
        [2*(qx*qz - qy*w),     2*(qy*qz + qx*w),     1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float32)
    pt_world = R @ np.array([x_c, y_c, z_c], dtype=np.float32) + pos.astype(np.float32)
    return pt_world


def _log_sam3(rgb: np.ndarray, depth: "np.ndarray | None" = None, cam=None, tick: int = 0):
    """Run SAM3 and log masks + boxes. Updates scene graph with every detection
    that has a valid world position. Also updates goal for --prompt target."""
    from PIL import Image as _PIL
    try:
        img = _PIL.fromarray(rgb.astype(np.uint8))
        state = _sam3_processor.set_image(img)
        all_boxes, all_labels, all_class_ids = [], [], []
        mask_layer = np.zeros(rgb.shape[:2], dtype=np.uint16)
        best_goal = None   # (score, pixel_uv, mask) for args_cli.prompt

        for cls_id, prompt in enumerate(_sam3_prompts, start=1):
            out = _sam3_processor.set_text_prompt(state=state, prompt=prompt)
            masks, boxes, scores = out.get("masks"), out.get("boxes"), out.get("scores")
            if masks is None or len(masks) == 0:
                continue
            is_target = (args_cli.prompt is not None and prompt == args_cli.prompt)

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
                m_bool = m > 0.5
                mask_layer[m_bool] = cls_id

                b = boxes[i]
                if hasattr(b, "cpu"):
                    b = b.cpu().numpy()
                b_arr = np.asarray(b).reshape(-1)[:4]
                all_boxes.append(b_arr)
                all_labels.append(f"{prompt} {s:.2f}")
                all_class_ids.append(cls_id)

                # Feed scene graph: unproject mask centroid to world XYZ
                if depth is not None and cam is not None:
                    ys_sg, xs_sg = np.where(m_bool)
                    if len(xs_sg) > 0:
                        u_sg = int(np.median(xs_sg))
                        v_sg = int(np.median(ys_sg))
                        xyz_sg = _pixel_to_world(cam, depth, u_sg, v_sg)
                        if xyz_sg is not None:
                            bbox_area = float((b_arr[2] - b_arr[0]) * (b_arr[3] - b_arr[1]))
                            _scene_graph.update(prompt, xyz_sg, s, tick, bbox_area)

                if is_target and (best_goal is None or s > best_goal[0]):
                    ys, xs = np.where(m_bool)
                    if len(xs) > 0:
                        u = int(np.median(xs))
                        v = int(np.median(ys))
                        best_goal = (s, u, v)

        rr.log("camera/rgb/sam3_mask", rr.SegmentationImage(mask_layer))
        if all_boxes:
            rr.log("camera/rgb/sam3", rr.Boxes2D(
                array=np.stack(all_boxes),
                array_format=rr.Box2DFormat.XYXY,
                labels=all_labels,
                class_ids=all_class_ids,
            ))
        else:
            rr.log("camera/rgb/sam3", rr.Clear(recursive=False))

        # Update goal from best target-matching detection — unless goal is
        # already locked (latched on a high-confidence earlier frame).
        if (best_goal is not None and depth is not None and cam is not None
                and not _goal_state["locked"]):
            s, u, v = best_goal
            xyz = _pixel_to_world(cam, depth, u, v)
            if xyz is not None:
                _goal_state["xyz"] = xyz
                _goal_state["t"] = time.time()
                _goal_state["score"] = s
                # Latch once we get a sufficiently confident detection. After
                # this, the goal XYZ is frozen until reset (L key) or arrived.
                if (args_cli.prompt_lock_conf >= 0
                        and s >= args_cli.prompt_lock_conf):
                    _goal_state["locked"] = True
                    print(f"[autonomous] goal LOCKED at "
                          f"({xyz[0]:+.2f},{xyz[1]:+.2f},{xyz[2]:+.2f}) m  "
                          f"score={s:.2f}")
                label = f"{args_cli.prompt} {s:.2f}" + (" [LOCKED]" if _goal_state["locked"] else "")
                rr.log("world/goal", rr.Points3D(
                    [xyz], colors=[[0, 255, 0]], radii=0.1,
                    labels=[label],
                ))
    except Exception as e:
        if not _SAM3_WARN_ONCE["done"]:
            print(f"[sam3] inference failed: {e}")
            _SAM3_WARN_ONCE["done"] = True


def _log_pointcloud(cam, depth, rgb, stride, max_depth):
    """Unproject head-camera depth to a world-frame coloured point cloud."""
    try:
        K = cam.data.intrinsic_matrices[0].cpu().numpy()
        pos, quat_wxyz = _get_cam_world_pose(cam)
    except Exception:
        return
    if depth.ndim == 3:
        depth = depth[..., 0]
    H, W = depth.shape
    vs = np.arange(0, H, stride)
    us = np.arange(0, W, stride)
    vv, uu = np.meshgrid(vs, us, indexing="ij")
    z = depth[vv, uu]
    mask = (z > 0.05) & (z < max_depth)
    if not np.any(mask):
        rr.log("world/cam/points", rr.Clear(recursive=False))
        return
    uu = uu[mask].astype(np.float32)
    vv = vv[mask].astype(np.float32)
    z  = z[mask].astype(np.float32)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    # OpenGL camera frame: +X right, +Y up, -Z forward
    x_cam =  (uu - cx) * z / fx
    y_cam = -(vv - cy) * z / fy
    z_cam = -z
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

    w, qx, qy, qz = quat_wxyz
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*w),     2*(qx*qz + qy*w)],
        [2*(qx*qy + qz*w),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*w)],
        [2*(qx*qz - qy*w),     2*(qy*qz + qx*w),     1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float32)
    pts_world = pts_cam @ R.T + pos.astype(np.float32)

    colours = rgb[vv.astype(int), uu.astype(int)]
    rr.log("world/cam/points", rr.Points3D(pts_world, colors=colours, radii=0.02))


def _rerun_log(tick, cam, robot: CameraRobot):
    rr.set_time("tick", sequence=tick)
    rr.set_time("sim_time", duration=tick * SIM_DT)

    w, qx, qy, qz = _yaw_pitch_to_quat_wxyz(robot.yaw, robot.pitch)
    rr.log("world/robot", rr.Transform3D(
        translation=[robot.x, robot.y, robot.z],
        rotation=rr.Quaternion(xyzw=[qx, qy, qz, w]),
    ))
    rr.log("world/robot/axes", rr.Arrows3D(
        vectors=[[0.3, 0, 0], [0, 0.3, 0], [0, 0, 0.3]],
        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    ))
    rr.log("robot/x",   rr.Scalars(robot.x))
    rr.log("robot/y",   rr.Scalars(robot.y))
    rr.log("robot/z",   rr.Scalars(robot.z))
    rr.log("robot/yaw", rr.Scalars(robot.yaw))

    if tick % CAMERA_DECIMATION != 0:
        return
    try:
        output = cam.data.output
    except Exception:
        return
    if output is None:
        return

    rgb, depth = None, None
    if "rgb" in output:
        rgb = output["rgb"][0].cpu().numpy()
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        rr.log("camera/rgb", rr.Image(rgb))

    if "distance_to_image_plane" in output:
        depth = output["distance_to_image_plane"][0].cpu().numpy().astype(np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth = np.where(np.isfinite(depth), depth, 20.0)
        rr.log("camera/depth", rr.DepthImage(depth, meter=1.0))

    if rgb is not None and _sam3_model is not None:
        _log_sam3(rgb, depth=depth, cam=cam, tick=tick)
        _scene_graph.log_to_rerun()

    if args_cli.pointcloud and rgb is not None and depth is not None:
        _log_pointcloud(cam, depth, rgb,
                        stride=args_cli.pointcloud_stride,
                        max_depth=args_cli.pointcloud_max_depth)


# ── Autonomous controller ────────────────────────────────────────────────────
# FSM state and last-state logger. "search" = rotate in place; "approach" =
# walk+turn toward goal; "arrived" = stopped within stop_dist.
_ctrl_state = {
    "mode": "search",
    "arrived_printed": False,
    # Phase accumulator for the pitch oscillation during search (radians,
    # advances with dt · 2π / tilt_period_s).
    "tilt_phase_rad": 0.0,
}

# ── Next-prompt input thread (post-arrival) ───────────────────────────────────
# When the robot arrives, we ask the user for the next target without blocking
# the main loop. A daemon thread reads stdin and pushes the response through a
# Queue; main loop polls it each tick.
from queue import Queue, Empty
from threading import Thread

_prompt_queue: "Queue[str]" = Queue()
_prompt_thread_active = False


def _start_next_prompt_thread() -> None:
    global _prompt_thread_active
    if _prompt_thread_active:
        return
    _prompt_thread_active = True

    def worker():
        global _prompt_thread_active
        try:
            resp = input(
                f"\n[autonomous] Arrived at '{args_cli.prompt}'. "
                f"Next target? (blank / 'quit' to stop autonomy): "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            resp = ""
        _prompt_queue.put(resp)
        _prompt_thread_active = False

    Thread(target=worker, daemon=True).start()


def _switch_target(new_label: str) -> None:
    """Switch the autonomous target to `new_label`: unlock goal, reset FSM,
    add label to SAM3 prompt list so it can be detected."""
    args_cli.prompt = new_label
    if new_label not in _sam3_prompts:
        _sam3_prompts.append(new_label)
    _goal_state["xyz"] = None
    _goal_state["locked"] = False
    _goal_state["score"] = 0.0
    _goal_state["t"] = 0.0
    _ctrl_state["mode"] = "search"
    _ctrl_state["arrived_printed"] = False
    _ctrl_state["tilt_phase_rad"] = 0.0
    print(f"[autonomous] switched target → '{new_label}'. Searching …")

def _search_step(robot: "CameraRobot", dt: float) -> None:
    """Continuously yaw while oscillating pitch as a sine wave. Every yaw angle
    gets scanned at multiple elevations in the ±`--prompt-tilt-deg` range."""
    robot.rotate_yaw(args_cli.prompt_search_yaw * dt)

    if args_cli.prompt_tilt_period_s > 0.0 and args_cli.prompt_tilt_deg > 0.0:
        _ctrl_state["tilt_phase_rad"] += (2.0 * math.pi / args_cli.prompt_tilt_period_s) * dt
        # Keep phase bounded to avoid precision drift over long runs.
        if _ctrl_state["tilt_phase_rad"] > 2.0 * math.pi:
            _ctrl_state["tilt_phase_rad"] -= 2.0 * math.pi
        tilt_amp = math.radians(args_cli.prompt_tilt_deg)
        robot.set_pitch(tilt_amp * math.sin(_ctrl_state["tilt_phase_rad"]))


def _step_autonomous(robot: "CameraRobot", dt: float, tick: int) -> None:
    """One controller tick. Writes robot.x/y/z/yaw directly based on FSM."""
    goal = _goal_state["xyz"]
    if goal is None:
        fresh = False
    elif _goal_state["locked"]:
        fresh = True   # latched goal never goes stale
    else:
        fresh = (time.time() - _goal_state["t"]) < args_cli.prompt_stale_s

    forward_dist = None
    heading_err  = None
    if goal is not None:
        dx = float(goal[0] - robot.x)
        dy = float(goal[1] - robot.y)
        forward_dist = math.hypot(dx, dy)
        heading_err = math.atan2(dy, dx) - robot.yaw
        # wrap to [-pi, pi]
        heading_err = (heading_err + math.pi) % (2 * math.pi) - math.pi

    # --- FSM transitions
    if not fresh:
        mode = "search"
    elif forward_dist is not None and forward_dist < args_cli.prompt_stop_dist:
        mode = "arrived"
    else:
        mode = "approach"

    if mode != _ctrl_state["mode"]:
        print(f"[autonomous] state {_ctrl_state['mode']} → {mode}"
              f"  dist={forward_dist if forward_dist is not None else '—'}"
              f"  heading_err_deg={math.degrees(heading_err) if heading_err is not None else '—'}")
        _ctrl_state["mode"] = mode
        if mode != "arrived":
            _ctrl_state["arrived_printed"] = False

    # --- Actuate
    if mode == "search":
        _search_step(robot, dt)
    elif mode == "approach":
        robot.set_pitch(0.0)  # level the camera while walking
        _ctrl_state["tilt_phase_rad"] = 0.0
        # Turn to face target, walk forward. Cap yaw rate to avoid overshooting.
        yaw_rate = max(-args_cli.prompt_search_yaw * 2,
                       min(args_cli.prompt_search_yaw * 2,
                           args_cli.prompt_heading_kp * heading_err))
        robot.rotate_yaw(yaw_rate * dt)
        # Only walk forward if mostly facing the target (|err| < 30°), else turn first
        if abs(heading_err) < math.radians(30):
            robot.translate_local(args_cli.prompt_walk_speed * dt, 0, 0)
    elif mode == "arrived":
        if not _ctrl_state["arrived_printed"]:
            print(f"[autonomous] ARRIVED at '{args_cli.prompt}'  "
                  f"(dist {forward_dist:.2f} m, score {_goal_state['score']:.2f})")
            _ctrl_state["arrived_printed"] = True
            _start_next_prompt_thread()

    # --- Rerun diagnostics
    if args_cli.rerun:
        rr.log("controller/state",
               rr.Scalars({"search": 0, "approach": 1, "arrived": 2}[mode]))
        if forward_dist is not None:
            rr.log("controller/forward_dist", rr.Scalars(forward_dist))
        if heading_err is not None:
            rr.log("controller/heading_error_deg", rr.Scalars(math.degrees(heading_err)))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    sim, cam = setup_scene()
    sim.reset()

    import carb
    carb.settings.get_settings().set("/app/runLoops/main/rateLimitFrequency", 60)
    carb.settings.get_settings().set("/app/runLoops/rendering/rateLimitFrequency", 60)

    robot = CameraRobot(args_cli.spawn_x, args_cli.spawn_y, args_cli.spawn_z,
                         math.radians(args_cli.spawn_yaw_deg))
    kb = KeyboardState()

    if args_cli.rerun:
        _rerun_init()
        print("[rerun] Viewer spawned")

    print("\nControls: arrows=forward/back/yaw  Q/E=strafe  R/F=up/down  L=reset")
    print("Starting loop. Close viewer to exit.\n")

    tick = 0
    last_report = time.time()
    # Autonomous mode: any keyboard press pauses autonomy for a few seconds.
    auto_mode = args_cli.prompt is not None
    manual_override_until = 0.0

    while simulation_app.is_running():
        # Check if the post-arrival prompt thread returned a new target.
        try:
            resp = _prompt_queue.get_nowait()
            if resp and resp.lower() != "quit":
                _switch_target(resp)
            else:
                print("[autonomous] autonomy stopped (keyboard active)")
                auto_mode = False
        except Empty:
            pass

        keys_held = bool(kb._keys_down)
        if keys_held or kb._reset_requested:
            manual_override_until = time.time() + 1.0
        use_auto = auto_mode and time.time() > manual_override_until

        if use_auto:
            _step_autonomous(robot, SIM_DT, tick)
        else:
            _apply_keyboard(kb, robot, SIM_DT)
        robot.write_to_stage("/World/Robot")

        # Isaac viewport mirrors the sensor camera — both sit at the robot origin
        # and look along robot yaw/pitch.
        cy, sy = math.cos(robot.yaw), math.sin(robot.yaw)
        cp, sp = math.cos(robot.pitch), math.sin(robot.pitch)
        sim.set_camera_view(
            eye=[robot.x, robot.y, robot.z],
            target=[robot.x + cy * cp, robot.y + sy * cp, robot.z + sp],
        )

        sim.step(render=True)
        if tick % CAMERA_DECIMATION == 0:
            cam.update(SIM_DT)

        if args_cli.rerun:
            _rerun_log(tick, cam, robot)

        now = time.time()
        if now - last_report >= 1.0:
            _scene_graph.record_pose(robot.x, robot.y, math.degrees(robot.yaw))
            # print(f"[tick {tick:5d}] pos=({robot.x:+.2f},{robot.y:+.2f},{robot.z:+.2f})m  "
            #       f"yaw={math.degrees(robot.yaw):+6.1f}°  "
            #       f"scene: {_scene_graph.summary()}")
            last_report = now

        # Auto-save scene graph periodically so no data is lost on crash
        _save_ticks = max(1, int(args_cli.save_every_s / SIM_DT))
        if tick > 0 and tick % _save_ticks == 0 and len(_scene_graph.objects) > 0:
            _scene_graph.save(str(sg_path))

        tick += 1

    # Final save on clean exit
    _scene_graph.scan_complete = True
    _scene_graph.save(str(sg_path))


# ── atexit fallback — catches Ctrl+C and most kill signals ────────────────────
import atexit as _atexit

_sg_path_global = args_cli.scene_graph_path

def _save_on_exit():
    if len(_scene_graph.objects) > 0:
        _scene_graph.save(_sg_path_global)

_atexit.register(_save_on_exit)


if __name__ == "__main__":
    sg_path = _sg_path_global
    main()
    simulation_app.close()
