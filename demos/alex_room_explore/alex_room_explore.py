#!/usr/bin/env python3
"""Play Alex V1 velocity policy in the room scene."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
import math
from pathlib import Path
from queue import Queue
from threading import Thread

import cv2
import torch
import mujoco
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from demos.cam_room_explore.cam_controller import AutoExploreController, TARGET_OBJECTS, YoloDetector
from demos.cam_room_explore.cam_room_explore import DashboardWindow
from alex_models import alex_sensors
from controllers import alex_action_set, llm_brain_controller, locomotion_controller
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scripts.play import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer, VerbosityLevel
from mjlab.viewer.native import keys as viewer_keys

DEFAULT_TASK = "Mjlab-Velocity-Flat-Alex-V1"
DEFAULT_CHECKPOINT = "../../pre_trained_models/Mjlab_Velocity_Flat_Alex_V1/model.pt"
DEFAULT_FLOORPLAN_XML = "scenes/ithor/FloorPlan1_physics_simple.xml"
ROOM_ATTACH_Z_OFFSET_M = 0.1
HEAD_PIP_WIDTH = 320
HEAD_PIP_HEIGHT = 180
HEAD_PIP_MARGIN_PX = 12
HEAD_PIP_WIDTH_FRACTION = 0.28
HEAD_PIP_MAX_HEIGHT_FRACTION = 0.35
STABILIZED_RGB_CAMERA_NAME = "alex_head_rgb_stabilized"
STABILIZED_DEPTH_CAMERA_NAME = "alex_head_depth_stabilized"
BASE_HEAD_CAMERA_QUAT_WXYZ = (-0.5, -0.5, 0.5, 0.5)


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
  parser.add_argument(
    "--brain-prompt",
    default=None,
    help='Goal prompt for LLM brain (example: "find the door").',
  )
  parser.add_argument(
    "--brain-model",
    default="gpt-4.1-mini",
    help="OpenAI model for LLM brain planning.",
  )
  parser.add_argument(
    "--brain-max-steps",
    type=int,
    default=30,
    help="Maximum planning steps for brain execution.",
  )
  parser.add_argument(
    "--brain-step-interval-s",
    type=float,
    default=1.5,
    help="Seconds between brain planning steps.",
  )
  parser.add_argument(
    "--head-pitch-target-rad",
    type=float,
    default=-0.23,
    help="Fixed neck_y target angle in radians (default: -0.23).",
  )
  parser.add_argument(
    "--verbose",
    action="store_true",
    help="Print verbose runtime logs including LLM interactions.",
  )
  parser.add_argument("--prompt", default=None, help="Target object label, for example 'door'.")
  parser.add_argument(
    "--yolo-model",
    default="/home/sravani/nadia/repository-group/ihmc-open-robotics-software/ihmc-perception/src/main/resources/yolo/best_multi_02_17_2026/best_multi_02_17_2026.onnx",
  )
  parser.add_argument("--target-labels", nargs="*", default=TARGET_OBJECTS)
  parser.add_argument("--confidence-threshold", type=float, default=0.25)
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

  head_body = None
  for body in spec.worldbody.find_all("body"):
    name = getattr(body, "name", "") or ""
    if name == "head" or name.endswith("/head"):
      head_body = body
      break

  if head_body is not None:
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

  camera_names = {
    cam.name
    for cam in spec.worldbody.find_all("camera")
    if getattr(cam, "name", None)
  }
  if STABILIZED_RGB_CAMERA_NAME not in camera_names:
    cam = spec.worldbody.add_camera()
    cam.name = STABILIZED_RGB_CAMERA_NAME
    cam.pos = (0.0, 0.0, 1.6)
    cam.quat = BASE_HEAD_CAMERA_QUAT_WXYZ
    cam.fovy = 69.0
  if STABILIZED_DEPTH_CAMERA_NAME not in camera_names:
    cam = spec.worldbody.add_camera()
    cam.name = STABILIZED_DEPTH_CAMERA_NAME
    cam.pos = (0.0, 0.0, 1.6)
    cam.quat = BASE_HEAD_CAMERA_QUAT_WXYZ
    cam.fovy = 69.0
  return spec


def _resolve_viewer(viewer: str) -> str:
  if viewer != "auto":
    return viewer
  has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
  return "native" if has_display else "viser"


def _quat_mul_wxyz(
  q1: tuple[float, float, float, float],
  q2: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
  w1, x1, y1, z1 = q1
  w2, x2, y2, z2 = q2
  return (
    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
  )


def _yaw_quat_wxyz(yaw_rad: float) -> tuple[float, float, float, float]:
  half = 0.5 * yaw_rad
  return (math.cos(half), 0.0, 0.0, math.sin(half))


def _find_camera_id_by_name(model: mujoco.MjModel, name: str) -> int:
  cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
  if cam_id >= 0:
    return cam_id
  for idx in range(model.ncam):
    cam_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, idx)
    if cam_name == name or (cam_name is not None and cam_name.endswith("/" + name)):
      return idx
  return -1


def _find_body_id_by_name_suffix(model: mujoco.MjModel, target_name: str) -> int:
  body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_name)
  if body_id >= 0:
    return body_id
  for idx in range(model.nbody):
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, idx)
    if body_name == target_name or (body_name is not None and body_name.endswith("/" + target_name)):
      return idx
  return -1


class _NoOpTimer:
  """Compatibility shim for mjlab timer objects missing in older pip installs."""
  measured_time: float = 0.0

  @contextlib.contextmanager
  def measure_time(self):
    yield


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
    brain_prompt: str | None = None,
    brain_model: str = "gpt-4.1-mini",
    brain_max_steps: int = 30,
    brain_step_interval_s: float = 1.5,
    verbose: bool = False,
  ):
    viewer_verbosity = VerbosityLevel.INFO if verbose else VerbosityLevel.SILENT
    super().__init__(
      env,
      loco_ctrl.policy,
      key_callback=self._on_manual_key,
      verbosity=viewer_verbosity,
    )
    self._loco_ctrl = loco_ctrl
    # Compatibility shim: older mjlab versions don't have these attributes
    if not hasattr(self, "_render_timer"):
      self._render_timer = _NoOpTimer()
    if not hasattr(self, "_sim_timer"):
      self._sim_timer = _NoOpTimer()
    if not hasattr(self, "_accumulated_sim_time"):
      self._accumulated_sim_time = 0.0
    if not hasattr(self, "_step_count"):
      self._step_count = 0
    self.lin_speed = 0.6
    self.yaw_speed = 0.8
    self.key_hold_timeout_s = 0.60
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
    self._pip_renderer = None
    self._rgb_writer = None
    self._depth_writer = None
    self._macro_actions = alex_action_set.build_default_action_set(
      lin_speed=self.lin_speed,
      yaw_speed=self.yaw_speed,
    )
    self._active_macro = alex_action_set.resolve_action_name(initial_macro_action)
    self._head_macro_warned = False
    self._show_head_pip = True
    self._pip_status_logged = False
    self._last_pip_update_s = 0.0
    self._pip_interval_s = 1.0 / 15.0
    self._stabilized_camera_ids = None
    self._head_body_id = -1
    self._brain = None

    if brain_prompt:
      action_descriptions = {
        action.value: cmd.description
        for action, cmd in self._macro_actions.items()
      }
      self._brain = llm_brain_controller.LLMBrainController(
        goal_prompt=brain_prompt,
        action_descriptions=action_descriptions,
        capture_rgb_bgr_fn=self._capture_head_rgb_bgr_for_brain,
        execute_action_fn=self._set_macro_action_by_name,
        model=brain_model,
        max_steps=brain_max_steps,
        step_interval_s=brain_step_interval_s,
        verbose=verbose,
        logger=lambda msg: self.log(msg, level=1),
      )
      self.log(
        f"[brain] enabled goal='{brain_prompt}' model={brain_model}",
        level=1,
      )

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
    if key == viewer_keys.KEY_I:
      self._show_head_pip = not self._show_head_pip
      if not self._show_head_pip and self.viewer is not None:
        self.viewer.clear_images()
      self.log(
        f"[viewer] head camera PiP {'enabled' if self._show_head_pip else 'disabled'}",
        level=1,
      )
      return

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

  def _set_macro_action_by_name(self, action_name: str) -> bool:
    try:
      action = alex_action_set.resolve_action_name(action_name)
    except Exception:
      self.log(f"[brain] unknown action '{action_name}'", level=1)
      return False
    self._active_macro = action
    self._head_macro_warned = False
    return True

  def step_simulation(self) -> None:
    if self._is_paused:
      return
    with self._sim_timer.measure_time():
      if self._brain is not None and self._brain.is_active:
        self._brain.tick()
      macro_applied = self._apply_macro_action()
      if self._brain is None and not macro_applied:
        self._apply_manual_twist()
      self._loco_ctrl.step_policy()
      self._record_camera_frame_if_enabled()
      self._step_count += 1
    self._accumulated_sim_time += self._sim_timer.measured_time

  def sync_env_to_viewer(self) -> None:
    self._render_head_camera_pip()
    super().sync_env_to_viewer()

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
    self._update_stabilized_cameras()

  def _resolve_stabilized_camera_ids(self) -> alex_sensors.AlexCameraIds:
    if self._stabilized_camera_ids is None:
      assert self.mjm is not None
      rgb_id = _find_camera_id_by_name(self.mjm, STABILIZED_RGB_CAMERA_NAME)
      depth_id = _find_camera_id_by_name(self.mjm, STABILIZED_DEPTH_CAMERA_NAME)
      if rgb_id < 0 or depth_id < 0:
        self._stabilized_camera_ids = alex_sensors.resolve_alex_camera_ids(self.mjm)
      else:
        self._stabilized_camera_ids = alex_sensors.AlexCameraIds(rgb=rgb_id, depth=depth_id)
    return self._stabilized_camera_ids

  def _update_stabilized_cameras(self) -> None:
    if self.mjm is None or self.mjd is None:
      return
    camera_ids = self._resolve_stabilized_camera_ids()
    if self._head_body_id < 0:
      self._head_body_id = _find_body_id_by_name_suffix(self.mjm, "head")
    if self._head_body_id < 0:
      return
    head_pos = self.mjd.xpos[self._head_body_id].copy()
    base_quat = np.asarray(self.mjd.qpos[3:7], dtype=np.float64)
    yaw_rad = _quat_wxyz_to_yaw_rad(base_quat)
    stabilized_quat = _quat_mul_wxyz(_yaw_quat_wxyz(yaw_rad), BASE_HEAD_CAMERA_QUAT_WXYZ)
    for cam_id in (camera_ids.rgb, camera_ids.depth):
      self.mjm.cam_pos[cam_id] = head_pos
      self.mjm.cam_quat[cam_id] = np.asarray(stabilized_quat, dtype=np.float64)
    mujoco.mj_forward(self.mjm, self.mjd)

  def _ensure_camera_pipeline(self, enable_recording_writers: bool) -> None:
    if self.mjm is None or self.mjd is None:
      return
    if self._camera_renderer is None:
      self._camera_ids = self._resolve_stabilized_camera_ids()
      self._camera_renderer = mujoco.Renderer(
        self.mjm,
        width=self._record_width,
        height=self._record_height,
      )
    if enable_recording_writers and self._rgb_writer is None:
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

  def _ensure_head_pip_pipeline(self) -> None:
    if self.mjm is None or self.mjd is None:
      return
    if self._camera_ids is None:
      self._camera_ids = self._resolve_stabilized_camera_ids()
    if self._pip_renderer is None:
      self._pip_renderer = mujoco.Renderer(
        self.mjm,
        width=HEAD_PIP_WIDTH,
        height=HEAD_PIP_HEIGHT,
      )

  def _render_head_camera_pip(self) -> None:
    if self.viewer is None or self.mjm is None or self.mjd is None:
      return
    if not self._show_head_pip:
      self.viewer.clear_images()
      return
    now_s = time.time()
    if (now_s - self._last_pip_update_s) < self._pip_interval_s:
      return
    try:
      with self._mj_lock:
        self._ensure_head_pip_pipeline()
        if self._pip_renderer is None or self._camera_ids is None:
          return
        self._pip_renderer.disable_depth_rendering()
        self._pip_renderer.update_scene(self.mjd, camera=self._camera_ids.rgb)
        rgb_frame = self._pip_renderer.render()

      viewport = self.viewer.viewport
      width = max(1, int(viewport.width * HEAD_PIP_WIDTH_FRACTION))
      aspect = rgb_frame.shape[0] / rgb_frame.shape[1]
      height = max(1, int(width * aspect))
      max_height = max(1, int(viewport.height * HEAD_PIP_MAX_HEIGHT_FRACTION))
      if height > max_height:
        height = max_height
        width = max(1, int(height / aspect))

      if rgb_frame.shape[1] != width or rgb_frame.shape[0] != height:
        rgb_frame = cv2.resize(rgb_frame, (width, height), interpolation=cv2.INTER_AREA)

      pip_rect = mujoco.MjrRect(HEAD_PIP_MARGIN_PX, HEAD_PIP_MARGIN_PX, width, height)
      self.viewer.set_images([(pip_rect, rgb_frame)])
      self._last_pip_update_s = now_s
      if not self._pip_status_logged:
        self.log(
          f"[viewer] head camera PiP ready ({width}x{height})",
          level=1,
        )
        self._pip_status_logged = True
    except Exception as exc:
      self.log(f"[WARN] Failed to render head camera PiP overlay: {exc}", level=1)
      if self.viewer is not None:
        self.viewer.clear_images()

  def _render_head_rgb_depth(self) -> tuple:
    self._sync_mj_data_for_active_env()
    return alex_sensors.render_alex_rgb_depth(
      renderer=self._camera_renderer,
      data=self.mjd,
      camera_ids=self._camera_ids,
      max_depth_m=self._record_max_depth_m,
    )

  def _capture_head_rgb_bgr_for_brain(self):
    if self.mjm is None or self.mjd is None:
      return None
    self._ensure_camera_pipeline(enable_recording_writers=False)
    rgb_bgr, _ = self._render_head_rgb_depth()
    return rgb_bgr

  def _record_camera_frame_if_enabled(self) -> None:
    if not self._record_cameras:
      return
    if self.mjm is None or self.mjd is None:
      return

    self._ensure_camera_pipeline(enable_recording_writers=True)
    rgb_bgr, depth_bgr = self._render_head_rgb_depth()
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
      if self._pip_renderer is not None:
        self._pip_renderer.close()
      if self.viewer is not None:
        self.viewer.clear_images()
    finally:
      super().close()


def _quat_wxyz_to_yaw_rad(quat_wxyz: np.ndarray) -> float:
  w, x, y, z = [float(v) for v in quat_wxyz]
  siny_cosp = 2.0 * (w * z + x * y)
  cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
  return math.atan2(siny_cosp, cosy_cosp)


def _start_prompt_thread(
  target_label: str,
  scene_graph: dict,
  response_queue: Queue[tuple[str, str | None]],
) -> None:
  found = target_label in scene_graph.get("object_index", {})
  seen_objects = sorted(scene_graph.get("object_index", {}).keys())

  def worker() -> None:
    print(f"Seen objects: {seen_objects}")
    print(f"Target '{target_label}' {'found' if found else 'not found'}.")
    choice = input(
      "Choose: (1) walk to object, (2) search for another object, blank to quit: "
    ).strip()
    if not choice:
      response_queue.put(("quit", None))
      return
    if choice == "1":
      if not found:
        print(f"Cannot walk: '{target_label}' was not found in the scene graph.")
        response_queue.put(("prompt_again", target_label))
        return
      response_queue.put(("walk", target_label))
      return
    if choice == "2":
      next_target = input("Which object should I look for next? ").strip()
      if not next_target:
        response_queue.put(("quit", None))
        return
      response_queue.put(("search", next_target))
      return
    print("Unrecognized choice.")
    response_queue.put(("prompt_again", target_label))

  Thread(target=worker, daemon=True).start()


def _step_native_viewer(viewer: FixedMainCameraViewer, dashboard: DashboardWindow | None = None) -> bool:
  if not viewer.is_running():
    return False
  viewer._process_actions()
  with viewer._render_timer.measure_time():
    viewer.sync_viewer_to_env()
    viewer.step_simulation()
    if dashboard is not None and viewer.viewer is not None:
      dashboard.update(viewer.viewer)
    viewer.sync_env_to_viewer()
  return True


class AlexPolicyRobotController:
  def __init__(self, viewer: FixedMainCameraViewer) -> None:
    self._viewer = viewer
    self._active_action = "stop"
    self._action_until_s = 0.0
    self.move_speed = viewer.lin_speed
    self.turn_speed_rad = viewer.yaw_speed
    self.depth_max_m = viewer._record_max_depth_m
    self.camera_width = viewer._record_width
    self.camera_height = viewer._record_height
    self._camera_ids = None

  def _ensure_initialized(self) -> None:
    if self._camera_ids is None:
      self._camera_ids = self._viewer._resolve_stabilized_camera_ids()

  @property
  def fovy_rad(self) -> float:
    self._ensure_initialized()
    assert self._camera_ids is not None
    return math.radians(float(self._viewer.mjm.cam_fovy[self._camera_ids.rgb]))

  @property
  def depth_camera_local_pos(self) -> tuple[float, float, float]:
    self._ensure_initialized()
    assert self._camera_ids is not None
    pos = self._viewer.mjm.cam_pos[self._camera_ids.depth]
    return (float(pos[0]), float(pos[1]), float(pos[2]))

  def _sync_viewer_data(self) -> None:
    self._ensure_initialized()
    self._viewer._sync_mj_data_for_active_env()
    self._viewer._ensure_camera_pipeline(enable_recording_writers=False)

  def _apply_active_action(self) -> None:
    now_s = time.time()
    if now_s > self._action_until_s:
      self.stop()
      return
    if self._active_action == "forward":
      cmd = locomotion_controller.TwistCommand(lin_x=self.move_speed)
    elif self._active_action == "backward":
      cmd = locomotion_controller.TwistCommand(lin_x=-self.move_speed)
    elif self._active_action == "strafe_left":
      cmd = locomotion_controller.TwistCommand(lin_y=self.move_speed)
    elif self._active_action == "strafe_right":
      cmd = locomotion_controller.TwistCommand(lin_y=-self.move_speed)
    elif self._active_action == "turn_left":
      cmd = locomotion_controller.TwistCommand(yaw=self.turn_speed_rad)
    elif self._active_action == "turn_right":
      cmd = locomotion_controller.TwistCommand(yaw=-self.turn_speed_rad)
    else:
      cmd = locomotion_controller.TwistCommand()
    self._viewer._loco_ctrl.set_command(cmd)

  def capture_rgb_depth(self) -> tuple[np.ndarray, np.ndarray]:
    self._sync_viewer_data()
    assert self._camera_ids is not None
    assert self._viewer._camera_renderer is not None
    renderer = self._viewer._camera_renderer
    renderer.disable_depth_rendering()
    renderer.update_scene(self._viewer.mjd, camera=self._camera_ids.rgb)
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    renderer.update_scene(self._viewer.mjd, camera=self._camera_ids.depth)
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth = np.clip(depth, 0.0, self.depth_max_m)
    return rgb, depth

  def capture_depth(self) -> np.ndarray:
    _, depth = self.capture_rgb_depth()
    return depth

  def get_pose(self) -> dict:
    sim_data = self._viewer.env.unwrapped.sim.data
    qpos = sim_data.qpos[self._viewer.env_idx].detach().cpu().numpy()
    pos = np.asarray(qpos[0:3], dtype=np.float64)
    quat = np.asarray(qpos[3:7], dtype=np.float64)
    return {
      "x": float(pos[0]),
      "y": float(pos[1]),
      "z": float(pos[2]),
      "yaw_rad": _quat_wxyz_to_yaw_rad(quat),
      "timestamp_s": time.time(),
    }

  def set_action(self, action_name: str, duration_s: float | None = None) -> None:
    self._active_action = action_name
    if duration_s is None:
      self._action_until_s = float("inf")
    else:
      self._action_until_s = time.time() + max(0.0, duration_s)
    self._apply_active_action()

  def stop(self) -> None:
    self._active_action = "stop"
    self._action_until_s = 0.0
    self._viewer._loco_ctrl.set_command(locomotion_controller.TwistCommand())

  def tick(self, viewer: FixedMainCameraViewer | None = None, dashboard: DashboardWindow | None = None) -> None:
    del viewer
    self._apply_active_action()
    if not _step_native_viewer(self._viewer, dashboard=dashboard):
      time.sleep(0.001)


def _run_viewer_loop(viewer: FixedMainCameraViewer, dashboard: DashboardWindow | None = None) -> None:
  viewer.setup()
  try:
    while viewer.is_running():
      if not _step_native_viewer(viewer, dashboard=dashboard):
        time.sleep(0.001)
  finally:
    if dashboard is not None:
      dashboard.close()
    viewer.close()


def _run_auto_loop(
  viewer: FixedMainCameraViewer,
  dashboard: DashboardWindow,
  auto: AutoExploreController,
  target_label: str,
) -> None:
  response_queue: Queue[tuple[str, str | None]] = Queue()
  prompt_active = False
  phase = "explore"
  viewer.setup()
  try:
    while viewer.is_running():
      if phase == "explore":
        scene_graph = auto.explore_room(viewer=viewer)
        print(f"Exploration complete. Seen objects: {sorted(scene_graph['object_index'].keys())}")
        print(f"Point cloud map summary: {scene_graph['point_cloud_map']}")
        phase = "prompt"
        prompt_active = False
        continue
      if phase == "prompt":
        if not prompt_active:
          _start_prompt_thread(target_label, auto.scene_graph, response_queue)
          prompt_active = True
        if not response_queue.empty():
          action, value = response_queue.get()
          prompt_active = False
          if action == "quit":
            break
          if action == "walk" and value is not None:
            phase = "walk"
          elif action == "search" and value is not None:
            target_label = value
            auto.scene_graph = {"views": [], "object_index": {}, "point_cloud_map": {}}
            auto.point_cloud_map.voxels.clear()
            phase = "explore"
          else:
            phase = "prompt"
        else:
          auto.robot.tick(viewer=viewer, dashboard=dashboard)
        continue
      if phase == "walk":
        success = auto.walk_to_target(target_label, viewer=viewer)
        print(f"walk_to_target('{target_label}') -> {success}")
        phase = "prompt"
        prompt_active = False
        continue
      auto.robot.tick(viewer=viewer, dashboard=dashboard)
  finally:
    dashboard.close()
    viewer.close()


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
    neck_pitch_target_rad=args.head_pitch_target_rad,
  )

  resolved_viewer = _resolve_viewer(args.viewer)
  try:
    if resolved_viewer == "native":
      record_dir = Path(args.record_dir).expanduser() if args.record_dir else None
      viewer = FixedMainCameraViewer(
        env,
        loco_ctrl,
        record_cameras=args.record_cameras,
        record_width=args.record_width,
        record_height=args.record_height,
        record_max_depth_m=args.record_max_depth_m,
        record_prefix=args.record_prefix,
        record_dir=record_dir,
        initial_macro_action=args.macro_action,
        brain_prompt=args.brain_prompt,
        brain_model=args.brain_model,
        brain_max_steps=args.brain_max_steps,
        brain_step_interval_s=args.brain_step_interval_s,
        verbose=args.verbose,
      )
      robot = AlexPolicyRobotController(viewer)
      detector = None
      try:
        detector = YoloDetector(
          model_name=args.yolo_model,
          target_labels=args.target_labels,
          confidence_threshold=args.confidence_threshold,
        )
      except Exception as exc:
        if args.prompt:
          raise
        print(f"YOLO dashboard detections disabled: {exc}")
      dashboard = DashboardWindow(robot, detector=detector)
      if args.prompt:
        assert detector is not None
        auto = AutoExploreController(
          robot,
          detector=detector,
          target_labels=args.target_labels,
          max_depth_m=args.record_max_depth_m,
          dashboard=dashboard,
        )
        _run_auto_loop(viewer, dashboard, auto, args.prompt)
      else:
        _run_viewer_loop(viewer, dashboard=dashboard)
    else:
      ViserPlayViewer(env, loco_ctrl.policy).run()
  finally:
    env.close()


if __name__ == "__main__":
  main()
