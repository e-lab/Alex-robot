"""Reusable locomotion controller for Alex velocity policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import torch
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.scripts.play import load_rl_cfg, load_runner_cls


@dataclass(frozen=True)
class TwistCommand:
  lin_x: float = 0.0
  lin_y: float = 0.0
  yaw: float = 0.0


class VelocityPolicyLocomotionController:
  """Velocity policy wrapper with a simple command interface."""

  def __init__(
    self,
    env: Any,
    policy: Any,
    runner: Any | None = None,
    neck_pitch_target_rad: float = -0.23,
  ):
    self.env = env
    self.policy = policy
    self.runner = runner
    self._neck_pitch_action_idx: int | None = None
    self._neck_pitch_qpos_idx: int | None = None
    self._neck_pitch_target: float = 0.0
    self._neck_pitch_target = neck_pitch_target_rad
    self._neck_pitch_kp: float = 2.0
    self._init_neck_pitch_lock()

  def _init_neck_pitch_lock(self) -> None:
    """Lock head tilt (neck_y) around initial pose to avoid walk-induced nodding."""
    try:
      model = self.env.unwrapped.sim.mj_model
      data = self.env.unwrapped.sim.data
      aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "neck_y")
      if aid < 0:
        return
      jid = int(model.actuator_trnid[aid, 0])
      qpos_idx = int(model.jnt_qposadr[jid])
      self._neck_pitch_action_idx = aid
      self._neck_pitch_qpos_idx = qpos_idx
      if model.jnt_limited[jid]:
        lo, hi = model.jnt_range[jid]
        self._neck_pitch_target = max(float(lo), min(float(hi), self._neck_pitch_target))
    except Exception:
      # If model layout differs, skip neck lock and keep baseline behavior.
      self._neck_pitch_action_idx = None
      self._neck_pitch_qpos_idx = None

  def _apply_neck_pitch_lock(self, actions: Any) -> Any:
    if self._neck_pitch_action_idx is None or self._neck_pitch_qpos_idx is None:
      return actions
    qpos = self.env.unwrapped.sim.data.qpos[:, self._neck_pitch_qpos_idx]
    error = self._neck_pitch_target - qpos
    correction = torch.clamp(self._neck_pitch_kp * error, -1.0, 1.0)
    actions[:, self._neck_pitch_action_idx] = correction.to(actions.dtype)
    return actions

  @classmethod
  def from_checkpoint(
    cls,
    env: Any,
    task: str,
    checkpoint: str | Path,
    device: str,
    agent_cfg: Any | None = None,
    neck_pitch_target_rad: float = -0.23,
  ) -> "VelocityPolicyLocomotionController":
    cfg = agent_cfg or load_rl_cfg(task)
    runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(cfg), device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
    return cls(
      env=env,
      policy=policy,
      runner=runner,
      neck_pitch_target_rad=neck_pitch_target_rad,
    )

  def _twist_term(self) -> Any:
    return self.env.unwrapped.command_manager.get_term("twist")

  def clamp_command(self, command: TwistCommand) -> TwistCommand:
    ranges = self._twist_term().cfg.ranges
    lin_x = max(ranges.lin_vel_x[0], min(ranges.lin_vel_x[1], command.lin_x))
    lin_y = max(ranges.lin_vel_y[0], min(ranges.lin_vel_y[1], command.lin_y))
    yaw = max(ranges.ang_vel_z[0], min(ranges.ang_vel_z[1], command.yaw))
    return TwistCommand(lin_x=lin_x, lin_y=lin_y, yaw=yaw)

  def set_command(self, command: TwistCommand, clamp: bool = True) -> TwistCommand:
    cmd = self.clamp_command(command) if clamp else command
    twist_term = self._twist_term()
    twist_term.command[:, 0] = cmd.lin_x
    twist_term.command[:, 1] = cmd.lin_y
    twist_term.command[:, 2] = cmd.yaw
    return cmd

  def get_command(self) -> TwistCommand:
    twist_term = self._twist_term()
    cmd = twist_term.command[0]
    return TwistCommand(lin_x=float(cmd[0]), lin_y=float(cmd[1]), yaw=float(cmd[2]))

  def step_policy(self) -> Any:
    with torch.no_grad():
      obs = self.env.get_observations()
      actions = self.policy(obs)
      actions = self._apply_neck_pitch_lock(actions)
      self.env.step(actions)
      return actions
