"""Convert LAFAN retargeted CSV motions to G1 tracking npz format.

Input CSV format is expected to match motions/README.md:
- root: XYZ + quaternion XYZW
- joints: G1 joint order from the README

Output npz is compatible with Mjlab tracking motion loader
(keys: fps, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_conjugate,
  quat_mul,
  quat_slerp,
)

G1_JOINT_ORDER = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)


class MotionResampler:
  def __init__(
    self,
    motion_xyzw: torch.Tensor,
    input_fps: float,
    output_fps: float,
    device: str,
  ):
    self.motion_xyzw = motion_xyzw.to(torch.float32).to(device)
    self.input_fps = float(input_fps)
    self.output_fps = float(output_fps)
    self.input_dt = 1.0 / self.input_fps
    self.output_dt = 1.0 / self.output_fps
    self.device = device

    self._split_inputs()
    self._interpolate()
    self._compute_velocities()

  def _split_inputs(self) -> None:
    self.base_pos_input = self.motion_xyzw[:, :3]
    # Convert XYZW -> WXYZ for simulator utilities.
    self.base_quat_input = self.motion_xyzw[:, 3:7][:, [3, 0, 1, 2]]
    self.joint_pos_input = self.motion_xyzw[:, 7:]

    self.num_input_frames = int(self.motion_xyzw.shape[0])
    if self.num_input_frames < 2:
      raise ValueError("Input must contain at least 2 frames.")

    self.duration = (self.num_input_frames - 1) * self.input_dt

  def _compute_frame_blend(
    self, times: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    phase = times / self.duration
    idx0 = (phase * (self.num_input_frames - 1)).floor().long()
    idx1 = torch.minimum(idx0 + 1, torch.tensor(self.num_input_frames - 1))
    blend = phase * (self.num_input_frames - 1) - idx0
    return idx0, idx1, blend

  def _interpolate(self) -> None:
    times = torch.arange(
      0.0,
      self.duration,
      self.output_dt,
      device=self.device,
      dtype=torch.float32,
    )
    if times.numel() == 0:
      times = torch.tensor([0.0], device=self.device)

    self.num_frames = int(times.shape[0])
    idx0, idx1, blend = self._compute_frame_blend(times)

    self.base_pos = self._lerp(
      self.base_pos_input[idx0], self.base_pos_input[idx1], blend
    )
    self.base_quat = self._slerp(
      self.base_quat_input[idx0], self.base_quat_input[idx1], blend
    )
    self.joint_pos = self._lerp(
      self.joint_pos_input[idx0], self.joint_pos_input[idx1], blend
    )

  def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    return a * (1.0 - blend.unsqueeze(-1)) + b * blend.unsqueeze(-1)

  def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(a)
    for i in range(a.shape[0]):
      out[i] = quat_slerp(a[i], b[i], float(blend[i]))
    return out

  def _so3_derivative(self, rotations_wxyz: torch.Tensor, dt: float) -> torch.Tensor:
    if rotations_wxyz.shape[0] < 3:
      return torch.zeros(rotations_wxyz.shape[0], 3, device=rotations_wxyz.device)

    q_prev, q_next = rotations_wxyz[:-2], rotations_wxyz[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))
    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
    return torch.cat([omega[:1], omega, omega[-1:]], dim=0)

  def _compute_velocities(self) -> None:
    self.base_lin_vel = torch.gradient(self.base_pos, spacing=self.output_dt, dim=0)[0]
    self.joint_vel = torch.gradient(self.joint_pos, spacing=self.output_dt, dim=0)[0]
    self.base_ang_vel = self._so3_derivative(self.base_quat, self.output_dt)


def _find_default_csv(motions_dir: Path) -> Path:
  candidates = sorted(
    p
    for p in motions_dir.glob("*.csv")
    if not p.name.startswith("._") and not p.name.startswith(".")
  )
  if not candidates:
    raise FileNotFoundError(f"No CSV files found in {motions_dir}")
  return candidates[0]


def _load_csv(path: Path, line_range: tuple[int, int] | None) -> np.ndarray:
  if line_range is None:
    return np.loadtxt(path, delimiter=",")

  start, end = line_range
  if start < 1 or end < start:
    raise ValueError("frame_range must be 1-indexed and satisfy start <= end.")

  return np.loadtxt(
    path,
    delimiter=",",
    skiprows=start - 1,
    max_rows=end - start + 1,
  )


def _validate_g1_csv(csv_data: np.ndarray) -> np.ndarray:
  if csv_data.ndim != 2:
    raise ValueError(f"Expected 2D CSV array, got shape {csv_data.shape}")

  required_cols = 7 + len(G1_JOINT_ORDER)
  if csv_data.shape[1] < required_cols:
    raise ValueError(
      f"CSV has {csv_data.shape[1]} columns but at least {required_cols} are required."
    )

  return csv_data[:, :required_cols]


def convert_lafan_csv_to_g1_npz(
  input_csv: Path,
  output_npz: Path,
  input_fps: float,
  output_fps: float,
  device: str,
  line_range: tuple[int, int] | None,
) -> None:
  csv_data = _validate_g1_csv(_load_csv(input_csv, line_range))

  if device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARNING] CUDA not available, falling back to CPU.")
    device = "cpu"

  resampled = MotionResampler(
    motion_xyzw=torch.from_numpy(csv_data),
    input_fps=input_fps,
    output_fps=output_fps,
    device=device,
  )

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps

  scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  robot: Entity = scene["robot"]
  robot_joint_indexes = robot.find_joints(G1_JOINT_ORDER, preserve_order=True)[0]

  log = {
    "fps": np.array([output_fps], dtype=np.float32),
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }

  scene.reset()

  for i in range(resampled.num_frames):
    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = resampled.base_pos[i : i + 1]
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = resampled.base_quat[i : i + 1]
    root_states[:, 7:10] = resampled.base_lin_vel[i : i + 1]
    root_states[:, 10:] = resampled.base_ang_vel[i : i + 1]
    robot.write_root_state_to_sim(root_states)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, robot_joint_indexes] = resampled.joint_pos[i : i + 1]
    joint_vel[:, robot_joint_indexes] = resampled.joint_vel[i : i + 1]
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
    log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
    log["body_pos_w"].append(robot.data.body_link_pos_w[0].cpu().numpy().copy())
    log["body_quat_w"].append(robot.data.body_link_quat_w[0].cpu().numpy().copy())
    log["body_lin_vel_w"].append(robot.data.body_link_lin_vel_w[0].cpu().numpy().copy())
    log["body_ang_vel_w"].append(robot.data.body_link_ang_vel_w[0].cpu().numpy().copy())

  for k in (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
  ):
    log[k] = np.stack(log[k], axis=0)

  output_npz.parent.mkdir(parents=True, exist_ok=True)
  np.savez(output_npz, **log)

  print(f"[INFO] Input CSV: {input_csv}")
  print(f"[INFO] Output NPZ: {output_npz}")
  print(f"[INFO] Frames: {log['joint_pos'].shape[0]}, fps: {output_fps}")


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Convert LAFAN/G1-order CSV motion to G1 tracking npz."
  )
  parser.add_argument(
    "--input_file",
    "--input-csv",
    dest="input_csv",
    type=Path,
    default=None,
    help="Path to input CSV. Default: first valid *.csv under motions/.",
  )
  parser.add_argument(
    "--motions-dir",
    type=Path,
    default=Path("motions"),
    help="Directory scanned when --input_file is not provided.",
  )
  parser.add_argument(
    "--output_file",
    "--output-npz",
    dest="output_npz",
    type=Path,
    default=None,
    help="Output path. Default: motions/<input_stem>_g1_tracking.npz",
  )
  parser.add_argument(
    "--output_name",
    type=str,
    default=None,
    help="Output base name. Writes to motions/<output_name>.npz if output path is not set.",
  )
  parser.add_argument("--input-fps", "--input_fps", dest="input_fps", type=float, default=30.0)
  parser.add_argument(
    "--output-fps", "--output_fps", dest="output_fps", type=float, default=50.0
  )
  parser.add_argument("--device", type=str, default="cuda:0")
  parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    default=None,
    help="Optional 1-indexed inclusive frame range: START END.",
  )
  args = parser.parse_args()

  input_csv = args.input_csv or _find_default_csv(args.motions_dir)
  output_npz: Path
  if args.output_npz is not None:
    output_npz = args.output_npz
  elif args.output_name is not None:
    output_npz = args.motions_dir / f"{args.output_name}.npz"
  else:
    output_npz = args.motions_dir / f"{input_csv.stem}_g1_tracking.npz"
  line_range = tuple(args.frame_range) if args.frame_range is not None else None

  convert_lafan_csv_to_g1_npz(
    input_csv=input_csv,
    output_npz=output_npz,
    input_fps=args.input_fps,
    output_fps=args.output_fps,
    device=args.device,
    line_range=line_range,
  )


if __name__ == "__main__":
  main()
