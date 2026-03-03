#!/usr/bin/env python3
"""Play Alex V1 velocity policy in the room scene."""

from __future__ import annotations

import argparse
import os
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


def _absolutize_assets(spec: mujoco.MjSpec, base_dir: Path) -> None:
  for tex in spec.textures:
    if tex.file and not Path(tex.file).is_absolute():
      tex.file = (base_dir / tex.file).resolve().as_posix()
  for mesh in spec.meshes:
    if mesh.file and not Path(mesh.file).is_absolute():
      mesh.file = (base_dir / mesh.file).resolve().as_posix()


def _attach_explore_scene(scene_spec: mujoco.MjSpec, floorplan_xml: Path) -> None:
  floorplan_spec = mujoco.MjSpec.from_file(str(floorplan_xml))
  _absolutize_assets(floorplan_spec, floorplan_xml.parent)
  frame = scene_spec.worldbody.add_frame()
  scene_spec.attach(floorplan_spec, prefix="room/", frame=frame)
  scene_spec.worldbody.add_camera(
    name="main",
    pos=(1.640, -2.477, 2.559),
    xyaxes=(0.786, 0.618, -0.000, -0.236, 0.300, 0.924),
    fovy=90.0,
  )


def _resolve_viewer(viewer: str) -> str:
  if viewer != "auto":
    return viewer
  has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
  return "native" if has_display else "viser"


class FixedMainCameraViewer(NativeMujocoViewer):
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
