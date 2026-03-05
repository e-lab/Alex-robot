"""Reusable locomotion controller for Alex velocity policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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

  def __init__(self, env: Any, policy: Any, runner: Any | None = None):
    self.env = env
    self.policy = policy
    self.runner = runner

  @classmethod
  def from_checkpoint(
    cls,
    env: Any,
    task: str,
    checkpoint: str | Path,
    device: str,
    agent_cfg: Any | None = None,
  ) -> "VelocityPolicyLocomotionController":
    cfg = agent_cfg or load_rl_cfg(task)
    runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(cfg), device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
    return cls(env=env, policy=policy, runner=runner)

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
      self.env.step(actions)
      return actions
