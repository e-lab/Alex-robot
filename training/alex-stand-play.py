#!/usr/bin/env python3
"""Play a trained Alex PPO standing policy in MuJoCo."""

from __future__ import annotations

import argparse
import os
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn


def repo_root_from_script() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy_mean = nn.Linear(hidden, act_dim)
        self.value_head = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        mean = torch.tanh(self.policy_mean(h))
        value = self.value_head(h).squeeze(-1)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std, value


class AlexPlayEnv:
    def __init__(
        self,
        mjcf_path: str,
        frame_skip: int = 5,
        hand_disturbance_scale: float = 0.0,
        hand_disturbance_smooth: float = 0.85,
    ):
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.dt = self.model.opt.timestep * frame_skip
        self.nu = self.model.nu
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.obs_dim = self.nq + self.nv + self.nu

        self.hand_disturbance_scale = hand_disturbance_scale
        self.hand_disturbance_smooth = hand_disturbance_smooth
        self.arm_noise = np.zeros(self.nu, dtype=np.float64)

        self.act_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.act_high = self.model.actuator_ctrlrange[:, 1].copy()
        self.act_mid = 0.5 * (self.act_low + self.act_high)
        self.act_amp = 0.5 * (self.act_high - self.act_low)
        self.prev_action = np.zeros(self.nu, dtype=np.float64)

        self.home_kf = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.arm_ids = self._find_arm_actuators()

    def _find_arm_actuators(self) -> np.ndarray:
        ids = []
        for i in range(self.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name and any(k in name for k in ("shoulder", "elbow", "wrist", "neck")):
                ids.append(i)
        return np.asarray(ids, dtype=np.int32)

    def reset(self):
        self.arm_noise.fill(0.0)
        self.prev_action.fill(0.0)
        if self.home_kf >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_kf)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def obs(self):
        return np.concatenate(
            [self.data.qpos.copy(), self.data.qvel.copy(), self.prev_action.copy()]
        ).astype(np.float32)

    def step(self, action: np.ndarray):
        action = np.clip(action.astype(np.float64), -1.0, 1.0)
        ctrl = self.act_mid + action * self.act_amp

        if self.hand_disturbance_scale > 0.0 and self.arm_ids.size > 0:
            eps = np.random.randn(self.arm_ids.shape[0]) * self.hand_disturbance_scale
            self.arm_noise[self.arm_ids] = (
                self.hand_disturbance_smooth * self.arm_noise[self.arm_ids]
                + (1.0 - self.hand_disturbance_smooth) * eps
            )
            ctrl[self.arm_ids] += self.arm_noise[self.arm_ids]

        ctrl = np.clip(ctrl, self.act_low, self.act_high)
        self.data.ctrl[:] = ctrl
        self.prev_action[:] = action
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)


def load_ckpt(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    if "model" not in ckpt:
        raise RuntimeError(f"Checkpoint missing 'model' key: {path}")
    obs_mean = ckpt.get("obs_mean")
    obs_var = ckpt.get("obs_var")
    return ckpt["model"], obs_mean, obs_var


def normalize_obs(obs: np.ndarray, obs_mean, obs_var):
    if obs_mean is None or obs_var is None:
        return obs
    return (obs - obs_mean) / np.sqrt(obs_var + 1e-8)


def run(args):
    scene = args.scene
    if not os.path.isabs(scene):
        scene = os.path.join(repo_root_from_script(), scene)
    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(repo_root_from_script(), ckpt_path)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    env = AlexPlayEnv(
        scene,
        frame_skip=args.frame_skip,
        hand_disturbance_scale=args.hand_disturbance_scale,
    )
    model = ActorCritic(env.obs_dim, env.nu, hidden=args.hidden).to(device)
    state, obs_mean, obs_var = load_ckpt(ckpt_path, device)
    model.load_state_dict(state)
    model.eval()

    env.reset()

    if args.no_viewer:
        for _ in range(args.steps):
            obs = env.obs()
            obs = normalize_obs(obs, obs_mean, obs_var).astype(np.float32)
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                mean, _, _ = model.forward(obs_t)
            action = mean.squeeze(0).cpu().numpy()
            env.step(action)
        print("no_viewer rollout complete")
        return

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            obs = env.obs()
            obs = normalize_obs(obs, obs_mean, obs_var).astype(np.float32)
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                mean, _, _ = model.forward(obs_t)
            action = mean.squeeze(0).cpu().numpy()
            env.step(action)
            viewer.sync()
            time.sleep(max(0.0, env.dt - 0.001))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        type=str,
        default="scenes/alex-scenes/scene_alex_v1_full_body_mjx.xml",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="training/checkpoints/alex_ppo_final.pt",
    )
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--hand-disturbance-scale", type=float, default=0.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--steps", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
