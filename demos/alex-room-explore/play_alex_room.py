#!/usr/bin/env python3
"""Play Alex V1 velocity policy in the room scene."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import torch
import mujoco
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scripts.play import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.native import keys as viewer_keys

DEFAULT_TASK = "Mjlab-Velocity-Flat-Alex-V1"
DEFAULT_CHECKPOINT = "../../trained-policies/Mjlab-Velocity-Flat-Alex-V1/model.pt"
DEFAULT_FLOORPLAN_XML = "scenes/ithor/FloorPlan1_physics_simple.xml"
ROOM_ATTACH_Z_OFFSET_M = 0.1

# Load reusable Alex sensor/camera helpers from file path.
_alex_sensors_path = "../../alex-models/alex_sensors.py"
_alex_spec = importlib.util.spec_from_file_location("alex_sensors", str(_alex_sensors_path))
if _alex_spec is None or _alex_spec.loader is None:
  raise RuntimeError(f"Failed to load module spec: {_alex_sensors_path}")
alex_sensors = importlib.util.module_from_spec(_alex_spec)
sys.modules["alex_sensors"] = alex_sensors
_alex_spec.loader.exec_module(alex_sensors)

_locomotion_controller_path = "../../controllers/locomotion_controller.py"
_locomotion_spec = importlib.util.spec_from_file_location(
  "locomotion_controller", str(_locomotion_controller_path)
)
if _locomotion_spec is None or _locomotion_spec.loader is None:
  raise RuntimeError(f"Failed to load module spec: {_locomotion_controller_path}")
locomotion_controller = importlib.util.module_from_spec(_locomotion_spec)
sys.modules["locomotion_controller"] = locomotion_controller
_locomotion_spec.loader.exec_module(locomotion_controller)

_alex_action_set_path = "../../controllers/alex_action_set.py"
_alex_action_set_spec = importlib.util.spec_from_file_location(
  "alex_action_set", str(_alex_action_set_path)
)
if _alex_action_set_spec is None or _alex_action_set_spec.loader is None:
  raise RuntimeError(f"Failed to load module spec: {_alex_action_set_path}")
alex_action_set = importlib.util.module_from_spec(_alex_action_set_spec)
sys.modules["alex_action_set"] = alex_action_set
_alex_action_set_spec.loader.exec_module(alex_action_set)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Load an Alex checkpoint and play it in the room scene."
  )
  parser.add_argument("--task", default=DEFAULT_TASK, help="Task id to play.")
  parser.add_argument(
    "--checkpoint",
    default=DEFAULT_CHECKPOINT,
    help="Path to checkpoint (.pt) file.",
  )
  parser.add_argument(
    "--viewer",
    choices=("auto", "native", "viser"),
    default="native",
    help="Viewer backend.",
  )
  parser.add_argument("--num-envs", type=int, default=1, help="Override num_envs.")
  parser.add_argument("--device", default=None, help='Torch device, e.g. "cuda:0".')
  parser.add_argument("--floorplan-xml", default=DEFAULT_FLOORPLAN_XML)
  parser.add_argument("--njmax", type=int, default=5000, help="MuJoCo njmax.")
  parser.add_argument("--nconmax", type=int, default=5000, help="MuJoCo nconmax.")
  parser.add_argument(
    "--contact-sensor-maxmatch",
    type=int,
    default=5000,
    help="MuJoCo contact sensor maxmatch.",
  )
  parser.add_argument(
    "--record-cameras",
    action="store_true",
    help="Record alex_head_rgb and alex_head_depth MP4 videos during play.",
  )
  parser.add_argument(
    "--record-width",
    type=int,
    default=1280,
    help="Recorded video width.",
  )
  parser.add_argument(
    "--record-height",
    type=int,
    default=720,
    help="Recorded video height.",
  )
  parser.add_argument(
    "--record-max-depth-m",
    type=float,
    default=5.0,
    help="Depth normalization max distance in meters.",
  )
  parser.add_argument(
    "--record-prefix",
    default="alex_play_room",
    help="Output filename prefix for recorded camera videos.",
  )
  parser.add_argument(
    "--record-dir",
    default=None,
    help="Output directory for recordings. Default: script directory.",
  )
  parser.add_argument(
    "--macro-action",
    default="stop",
    help=(
      "Initial macro action: stop, walk_straight, walk_backward, turn_left, "
      "turn_right, strafe_left, strafe_right, turn_head_left, turn_head_right."
    ),
  )
  return parser.parse_args()


def _attach_explore_scene(scene_spec: mujoco.MjSpec, floorplan_xml: Path) -> None:
  floorplan_spec = mujoco.MjSpec.from_file(str(floorplan_xml))
  frame = scene_spec.worldbody.add_frame()
  frame.pos = (0.0, 0.0, ROOM_ATTACH_Z_OFFSET_M)
  scene_spec.attach(floorplan_spec, prefix="room/", frame=frame)
  scene_spec.worldbody.add_camera(
    name="main",
    pos=(1.6, -2.5, 1.8),
    xyaxes=(0.786, 0.618, -0.000, -0.236, 0.300, 0.924),
    fovy=75.0,
  )


def _ensure_alex_head_cameras(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  camera_names = {
    cam.name
    for cam in spec.worldbody.find_all("camera")
    if getattr(cam, "name", None)
  }
  needs_rgb = "alex_head_rgb" not in camera_names
  needs_depth = "alex_head_depth" not in camera_names
  if not (needs_rgb or needs_depth):
    return spec

  head_body = None
  for body in spec.worldbody.find_all("body"):
    name = getattr(body, "name", "") or ""
    if name == "head" or name.endswith("/head"):
      head_body = body
      break
  if head_body is None:
    return spec

  if needs_rgb:
    cam = head_body.add_camera()
    cam.name = "alex_head_rgb"
    cam.pos = (0.11, 0.0, 0.06)
    cam.quat = (-0.5, -0.5, 0.5, 0.5)
    cam.fovy = 69.0
  if needs_depth:
    cam = head_body.add_camera()
    cam.name = "alex_head_depth"
    cam.pos = (0.11, 0.0, 0.06)
    cam.quat = (-0.5, -0.5, 0.5, 0.5)
    cam.fovy = 69.0
  return spec


def _resolve_viewer(viewer: str) -> str:
  if viewer != "auto":
    return viewer
  has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
  return "native" if has_display else "viser"


class FixedMainCameraViewer(NativeMujocoViewer):
  def __init__(
    self,
    env,
    loco_ctrl,
    record_cameras: bool = False,
    record_width: int = 1280,
    record_height: int = 720,
    record_max_depth_m: float = 5.0,
    record_prefix: str = "alex_play_room",
    record_dir: Path | None = None,
    initial_macro_action: str = "stop",
  ):
    super().__init__(env, loco_ctrl.policy, key_callback=self._on_manual_key)
    self._loco_ctrl = loco_ctrl
    self.lin_speed = 0.6
    self.yaw_speed = 0.8
    self.key_hold_timeout_s = 0.20
    self._last_key_time = {
      "up": 0.0,
      "down": 0.0,
      "left": 0.0,
      "right": 0.0,
      "strafe_left": 0.0,
      "strafe_right": 0.0,
    }
    # MuJoCo key callback only provides key codes; keep a short-lived latch
    # when CMD (Super) is pressed so the next arrow key can be treated as CMD+Arrow.
    self._cmd_latch_until = 0.0
    self._record_cameras = record_cameras
    self._record_width = record_width
    self._record_height = record_height
    self._record_max_depth_m = record_max_depth_m
    self._record_prefix = record_prefix
    self._record_dir = record_dir
    self._camera_ids = None
    self._camera_renderer = None
    self._rgb_writer = None
    self._depth_writer = None
    self._macro_actions = alex_action_set.build_default_action_set(
      lin_speed=self.lin_speed,
      yaw_speed=self.yaw_speed,
    )
    self._active_macro = alex_action_set.resolve_action_name(initial_macro_action)
    self._head_macro_warned = False

  def _is_key_active(self, key_name: str, now_s: float) -> bool:
    return (now_s - self._last_key_time[key_name]) <= self.key_hold_timeout_s

  def _clamp_twist(self, lin_x: float, lin_y: float, yaw: float) -> tuple[float, float, float]:
    clamped = self._loco_ctrl.clamp_command(
      locomotion_controller.TwistCommand(lin_x=lin_x, lin_y=lin_y, yaw=yaw)
    )
    return clamped.lin_x, clamped.lin_y, clamped.yaw

  def _apply_manual_twist(self) -> None:
    now_s = time.time()
    forward = 1.0 if self._is_key_active("up", now_s) else 0.0
    backward = 1.0 if self._is_key_active("down", now_s) else 0.0
    turn_left = 1.0 if self._is_key_active("left", now_s) else 0.0
    turn_right = 1.0 if self._is_key_active("right", now_s) else 0.0
    strafe_left = 1.0 if self._is_key_active("strafe_left", now_s) else 0.0
    strafe_right = 1.0 if self._is_key_active("strafe_right", now_s) else 0.0

    cmd_lin_x = self.lin_speed * (forward - backward)
    cmd_lin_y = self.lin_speed * (strafe_left - strafe_right)
    cmd_yaw = self.yaw_speed * (turn_left - turn_right)
    cmd_lin_x, cmd_lin_y, cmd_yaw = self._clamp_twist(cmd_lin_x, cmd_lin_y, cmd_yaw)

    self._loco_ctrl.set_command(
      locomotion_controller.TwistCommand(
        lin_x=cmd_lin_x, lin_y=cmd_lin_y, yaw=cmd_yaw
      )
    )

  def _on_manual_key(self, key: int) -> None:
    now_s = time.time()
    if key in alex_action_set.DEFAULT_KEY_BINDINGS:
      self._active_macro = alex_action_set.DEFAULT_KEY_BINDINGS[key]
      self._head_macro_warned = False
      macro = self._macro_actions[self._active_macro]
      self.log(f"[macro] {macro.action.value}: {macro.description}", level=1)
      return

    if key in (viewer_keys.KEY_LEFT_SUPER, viewer_keys.KEY_RIGHT_SUPER):
      self._cmd_latch_until = now_s + 0.4
      return

    cmd_modified = now_s <= self._cmd_latch_until
    if key == viewer_keys.KEY_UP and not cmd_modified:
      self._last_key_time["up"] = now_s
    elif key == viewer_keys.KEY_DOWN and not cmd_modified:
      self._last_key_time["down"] = now_s
    elif key == viewer_keys.KEY_LEFT and not cmd_modified:
      self._last_key_time["left"] = now_s
    elif key == viewer_keys.KEY_RIGHT and not cmd_modified:
      self._last_key_time["right"] = now_s
    elif key == viewer_keys.KEY_LEFT and cmd_modified:
      self._last_key_time["strafe_left"] = now_s
      self._cmd_latch_until = 0.0
    elif key == viewer_keys.KEY_RIGHT and cmd_modified:
      self._last_key_time["strafe_right"] = now_s
      self._cmd_latch_until = 0.0
    else:
      return
    self._active_macro = alex_action_set.AlexMacroAction.STOP
    # Show instantaneous held-command estimate.
    cmd_lin_x, cmd_lin_y, cmd_yaw = self._clamp_twist(
      self.lin_speed * (
        (1.0 if self._is_key_active("up", now_s) else 0.0) -
        (1.0 if self._is_key_active("down", now_s) else 0.0)
      ),
      self.lin_speed * (
        (1.0 if self._is_key_active("strafe_left", now_s) else 0.0) -
        (1.0 if self._is_key_active("strafe_right", now_s) else 0.0)
      ),
      self.yaw_speed * (
        (1.0 if self._is_key_active("left", now_s) else 0.0) -
        (1.0 if self._is_key_active("right", now_s) else 0.0)
      ),
    )
    self.log(
      f"[twist-hold] x={cmd_lin_x:+.2f} y={cmd_lin_y:+.2f} yaw={cmd_yaw:+.2f}",
      level=1,
    )

  def _apply_macro_action(self) -> bool:
    if self._active_macro == alex_action_set.AlexMacroAction.STOP:
      return False
    macro = self._macro_actions[self._active_macro]
    if macro.head_yaw_deg != 0.0:
      self._loco_ctrl.set_command(locomotion_controller.TwistCommand())
      if not self._head_macro_warned:
        self.log(
          f"[macro] {macro.action.value} selected. Head joint actuation is planner-level only in this viewer.",
          level=1,
        )
        self._head_macro_warned = True
      return True

    self._loco_ctrl.set_command(
      locomotion_controller.TwistCommand(
        lin_x=macro.lin_x,
        lin_y=macro.lin_y,
        yaw=macro.yaw,
      )
    )
    return True

  def step_simulation(self) -> None:
    if self._is_paused:
      return
    with self._sim_timer.measure_time():
      if not self._apply_macro_action():
        self._apply_manual_twist()
      self._loco_ctrl.step_policy()
      self._record_camera_frame_if_enabled()
      self._step_count += 1
    self._accumulated_sim_time += self._sim_timer.measured_time

  def _setup_camera(self) -> None:
    super()._setup_camera()
    if self.viewer is None or self.mjm is None:
      return
    self.mjm.vis.headlight.active = 0
    self.mjm.light_castshadow[:] = 0
    self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    alex_sensors.lock_view_to_main_camera(self.viewer, self.mjm)

  def _resolve_record_paths(self) -> tuple[Path, Path]:
    out_dir = self._record_dir or Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_out_path = out_dir / f"{self._record_prefix}_head_rgb.mp4"
    depth_out_path = out_dir / f"{self._record_prefix}_head_depth.mp4"
    return rgb_out_path, depth_out_path

  def _sync_mj_data_for_active_env(self) -> None:
    if self.mjm is None or self.mjd is None:
      return
    sim_data = self.env.unwrapped.sim.data
    if self.mjm.nq > 0:
      self.mjd.qpos[:] = sim_data.qpos[self.env_idx].cpu().numpy()
      self.mjd.qvel[:] = sim_data.qvel[self.env_idx].cpu().numpy()
    if self.mjm.nmocap > 0:
      self.mjd.mocap_pos[:] = sim_data.mocap_pos[self.env_idx].cpu().numpy()
      self.mjd.mocap_quat[:] = sim_data.mocap_quat[self.env_idx].cpu().numpy()
    mujoco.mj_forward(self.mjm, self.mjd)

  def _record_camera_frame_if_enabled(self) -> None:
    if not self._record_cameras:
      return
    if self.mjm is None or self.mjd is None:
      return

    if self._camera_renderer is None:
      self._camera_ids = alex_sensors.resolve_alex_camera_ids(self.mjm)
      self._camera_renderer = mujoco.Renderer(
        self.mjm,
        width=self._record_width,
        height=self._record_height,
      )
      fps = int(round(1.0 / self.mjm.opt.timestep))
      rgb_out_path, depth_out_path = self._resolve_record_paths()
      self._rgb_writer = alex_sensors.create_mp4_writer(
        rgb_out_path,
        fps,
        self._record_width,
        self._record_height,
      )
      self._depth_writer = alex_sensors.create_mp4_writer(
        depth_out_path,
        fps,
        self._record_width,
        self._record_height,
      )

    self._sync_mj_data_for_active_env()
    rgb_bgr, depth_bgr = alex_sensors.render_alex_rgb_depth(
      renderer=self._camera_renderer,
      data=self.mjd,
      camera_ids=self._camera_ids,
      max_depth_m=self._record_max_depth_m,
    )
    self._rgb_writer.write(rgb_bgr)
    self._depth_writer.write(depth_bgr)

  def close(self) -> None:
    try:
      if self._rgb_writer is not None:
        self._rgb_writer.release()
      if self._depth_writer is not None:
        self._depth_writer.release()
      if self._camera_renderer is not None:
        self._camera_renderer.close()
    finally:
      super().close()


def main() -> None:
  args = parse_args()
  configure_torch_backends()

  repo_root = Path(__file__).resolve().parents[2]
  checkpoint = (
    Path(args.checkpoint).expanduser().resolve()
    if Path(args.checkpoint).is_absolute()
    else (Path(__file__).resolve().parent / args.checkpoint).resolve()
  )
  if not checkpoint.exists():
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")

  floorplan_arg = getattr(args, "floorplan_xml")
  floorplan_xml = (
    Path(floorplan_arg).expanduser()
    if Path(floorplan_arg).is_absolute()
    else (repo_root / floorplan_arg)
  )
  if not floorplan_xml.exists():
    raise FileNotFoundError(f"Floorplan XML not found: {floorplan_xml}")

  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(args.task, play=True)
  agent_cfg = load_rl_cfg(args.task)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.scene.spec_fn = lambda spec: _attach_explore_scene(spec, floorplan_xml)
  if "robot" in env_cfg.scene.entities:
    robot_cfg = env_cfg.scene.entities["robot"]
    if hasattr(robot_cfg, "spec_fn") and callable(robot_cfg.spec_fn):
      orig_robot_spec_fn = robot_cfg.spec_fn
      robot_cfg.spec_fn = lambda: _ensure_alex_head_cameras(orig_robot_spec_fn())
  env_cfg.sim.njmax = args.njmax
  env_cfg.sim.nconmax = args.nconmax
  env_cfg.sim.contact_sensor_maxmatch = args.contact_sensor_maxmatch
  if "twist" in env_cfg.commands:
    twist_cmd = env_cfg.commands["twist"]
    twist_cmd.rel_standing_envs = 0.0
    twist_cmd.resampling_time_range = (1e9, 1e9)
  if "reset_base" in env_cfg.events:
    env_cfg.events["reset_base"].params["pose_range"] = {
      "x": (1.2, 1.2),
      "y": (-0.8, -0.8),
      "z": (0.0, 0.0),
      "roll": (0.0, 0.0),
      "pitch": (0.0, 0.0),
      "yaw": (0.0, 0.0),
    }

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  loco_ctrl = locomotion_controller.VelocityPolicyLocomotionController.from_checkpoint(
    env=env,
    task=args.task,
    checkpoint=checkpoint,
    device=device,
    agent_cfg=agent_cfg,
  )

  resolved_viewer = _resolve_viewer(args.viewer)
  try:
    if resolved_viewer == "native":
      record_dir = Path(args.record_dir).expanduser() if args.record_dir else None
      FixedMainCameraViewer(
        env,
        loco_ctrl,
        record_cameras=args.record_cameras,
        record_width=args.record_width,
        record_height=args.record_height,
        record_max_depth_m=args.record_max_depth_m,
        record_prefix=args.record_prefix,
        record_dir=record_dir,
        initial_macro_action=args.macro_action,
      ).run()
    else:
      ViserPlayViewer(env, loco_ctrl.policy).run()
  finally:
    env.close()


if __name__ == "__main__":
  main()
