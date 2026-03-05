#!/usr/bin/env python3
"""Play Alex V1 velocity policy in the room scene."""

from __future__ import annotations

import argparse
import os
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import mujoco
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts.play import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.native import keys as viewer_keys

DEFAULT_TASK = "Mjlab-Velocity-Flat-Alex-V1"
DEFAULT_CHECKPOINT = "Mjlab-Velocity-Flat-Alex-V1/model.pt"
DEFAULT_FLOORPLAN_XML = "scenes/ithor/FloorPlan1_physics_simple.xml"


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
  return parser.parse_args()


def _attach_explore_scene(scene_spec: mujoco.MjSpec, floorplan_xml: Path) -> None:
  floorplan_spec = mujoco.MjSpec.from_file(str(floorplan_xml))
  frame = scene_spec.worldbody.add_frame()
  scene_spec.attach(floorplan_spec, prefix="room/", frame=frame)
  scene_spec.worldbody.add_camera(
    name="main",
    pos=(1.6, -2.5, 1.8),
    xyaxes=(0.786, 0.618, -0.000, -0.236, 0.300, 0.924),
    fovy=75.0,
  )


def _resolve_viewer(viewer: str) -> str:
  if viewer != "auto":
    return viewer
  has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
  return "native" if has_display else "viser"


class FixedMainCameraViewer(NativeMujocoViewer):
  def __init__(self, env, policy):
    super().__init__(env, policy, key_callback=self._on_manual_key)
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

  def _is_key_active(self, key_name: str, now_s: float) -> bool:
    return (now_s - self._last_key_time[key_name]) <= self.key_hold_timeout_s

  def _clamp_twist(self, lin_x: float, lin_y: float, yaw: float) -> tuple[float, float, float]:
    twist_term = self.env.unwrapped.command_manager.get_term("twist")
    ranges = twist_term.cfg.ranges
    lin_x = max(ranges.lin_vel_x[0], min(ranges.lin_vel_x[1], lin_x))
    lin_y = max(ranges.lin_vel_y[0], min(ranges.lin_vel_y[1], lin_y))
    yaw = max(ranges.ang_vel_z[0], min(ranges.ang_vel_z[1], yaw))
    return lin_x, lin_y, yaw

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

    twist_term = self.env.unwrapped.command_manager.get_term("twist")
    cmd = twist_term.command
    cmd[:, 0] = cmd_lin_x
    cmd[:, 1] = cmd_lin_y
    cmd[:, 2] = cmd_yaw

  def _on_manual_key(self, key: int) -> None:
    now_s = time.time()
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

  def step_simulation(self) -> None:
    if self._is_paused:
      return
    with torch.no_grad():
      with self._sim_timer.measure_time():
        self._apply_manual_twist()
        obs = self.env.get_observations()
        actions = self.policy(obs)
        self.env.step(actions)
        self._step_count += 1
      self._accumulated_sim_time += self._sim_timer.measured_time

  def _setup_camera(self) -> None:
    super()._setup_camera()
    if self.viewer is None or self.mjm is None:
      return
    main_cam_id = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_CAMERA, "main")
    if main_cam_id >= 0:
      lock_ctx = self.viewer.lock() if hasattr(self.viewer, "lock") else nullcontext()
      with lock_ctx:
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.viewer.cam.fixedcamid = main_cam_id


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

  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  resolved_viewer = _resolve_viewer(args.viewer)
  try:
    if resolved_viewer == "native":
      FixedMainCameraViewer(env, policy).run()
    else:
      ViserPlayViewer(env, policy).run()
  finally:
    env.close()


if __name__ == "__main__":
  main()
