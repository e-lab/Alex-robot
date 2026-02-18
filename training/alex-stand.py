#!/usr/bin/env python3
"""Train Alex to stand using PPO in MuJoCo.

This script trains on:
  scenes/alex-scenes/scene_alex_v1_full_body_mjx.xml

Key feature:
  Random small arm/hand actuator disturbances are injected during rollouts
  so the policy learns to maintain balance despite perturbations.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass

import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def repo_root_from_script() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class RunningNorm:
    """Running mean/std with Welford updates."""

    def __init__(self, size: int, eps: float = 1e-8):
        self.mean = np.zeros(size, dtype=np.float64)
        self.var = np.ones(size, dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot
        new_var = m2 / tot

        self.mean = new_mean
        self.var = new_var
        self.count = tot

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


class AlexStandEnv:
    def __init__(
        self,
        mjcf_path: str,
        episode_steps: int = 1000,
        frame_skip: int = 5,
        hand_disturbance_scale: float = 0.06,
        hand_disturbance_smooth: float = 0.85,
        noise_scale: float = 0.01,
    ):
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)

        self.episode_steps = episode_steps
        self.frame_skip = frame_skip
        self.hand_disturbance_scale = hand_disturbance_scale
        self.hand_disturbance_smooth = hand_disturbance_smooth
        self.noise_scale = noise_scale
        self.t = 0

        self.nu = self.model.nu
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.dt = self.model.opt.timestep * frame_skip

        self.act_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.act_high = self.model.actuator_ctrlrange[:, 1].copy()
        self.act_mid = 0.5 * (self.act_low + self.act_high)
        self.act_amp = 0.5 * (self.act_high - self.act_low)

        self.home_kf = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "home"
        )

        self.pelvis_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link"
        )
        if self.pelvis_body < 0:
            raise RuntimeError("Body 'pelvis_link' not found in model.")

        self.arm_ids = self._find_arm_actuators()
        self.arm_noise = np.zeros(self.nu, dtype=np.float64)

        # Observation: qpos + qvel + previous action
        self.obs_dim = self.nq + self.nv + self.nu
        self.prev_action = np.zeros(self.nu, dtype=np.float64)

    def _find_arm_actuators(self) -> np.ndarray:
        ids = []
        for i in range(self.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name is None:
                continue
            if any(k in name for k in ("shoulder", "elbow", "wrist", "neck")):
                ids.append(i)
        return np.asarray(ids, dtype=np.int32)

    def _pelvis_z(self) -> float:
        return float(self.data.xpos[self.pelvis_body, 2])

    def _pelvis_upright(self) -> float:
        # 1 when pelvis local +Z aligns with world +Z.
        xmat = self.data.xmat[self.pelvis_body].reshape(3, 3)
        return float(np.clip(xmat[2, 2], -1.0, 1.0))

    def _obs(self) -> np.ndarray:
        return np.concatenate(
            [
                self.data.qpos.copy(),
                self.data.qvel.copy(),
                self.prev_action.copy(),
            ]
        ).astype(np.float32)

    def reset(self) -> np.ndarray:
        self.t = 0
        self.arm_noise.fill(0.0)
        self.prev_action.fill(0.0)

        if self.home_kf >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_kf)
        else:
            mujoco.mj_resetData(self.model, self.data)

        # Add small reset noise to joint states (keep base pose mostly intact).
        self.data.qpos[7:] += self.noise_scale * np.random.randn(self.nq - 7)
        self.data.qvel[:] = self.noise_scale * np.random.randn(self.nv)
        mujoco.mj_forward(self.model, self.data)
        return self._obs()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, -1.0, 1.0)
        ctrl = self.act_mid + action * self.act_amp

        # Smooth random arm disturbances.
        eps = np.random.randn(self.arm_ids.shape[0]) * self.hand_disturbance_scale
        self.arm_noise[self.arm_ids] = (
            self.hand_disturbance_smooth * self.arm_noise[self.arm_ids]
            + (1.0 - self.hand_disturbance_smooth) * eps
        )
        ctrl[self.arm_ids] += self.arm_noise[self.arm_ids]
        ctrl = np.clip(ctrl, self.act_low, self.act_high)

        self.data.ctrl[:] = ctrl
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.t += 1
        self.prev_action[:] = action

        z = self._pelvis_z()
        up = self._pelvis_upright()
        qvel_norm = float(np.linalg.norm(self.data.qvel))
        arm_disturb_mag = float(np.linalg.norm(self.arm_noise[self.arm_ids]))

        # Standing reward with disturbance robustness.
        reward = (
            1.0
            + 2.0 * np.exp(-20.0 * (z - 0.98) ** 2)
            + 1.5 * max(up, 0.0)
            - 0.002 * qvel_norm
            - 0.001 * float(np.square(action).sum())
            - 0.01 * arm_disturb_mag
        )

        fallen = (z < 0.65) or (up < 0.45) or (not np.isfinite(self.data.qpos).all())
        timeout = self.t >= self.episode_steps
        done = bool(fallen or timeout)

        if fallen:
            reward -= 5.0

        info = {
            "pelvis_z": z,
            "upright": up,
            "fallen": fallen,
            "timeout": timeout,
        }
        return self._obs(), float(reward), done, info


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

    def act(self, obs: torch.Tensor):
        mean, std, value = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        action = torch.clamp(action, -1.0, 1.0)
        logp = dist.log_prob(action).sum(-1)
        return action, logp, value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        mean, std, value = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        logp = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logp, entropy, value


@dataclass
class PPOCfg:
    total_steps: int = 1_500_000
    rollout_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    lr: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    update_epochs: int = 10
    minibatch_size: int = 256


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n)):
        nonterminal = 1.0 - float(dones[t])
        next_value = last_value if t == n - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae
    returns = adv + values
    return adv, returns


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    scene_path = args.scene
    if not os.path.isabs(scene_path):
        scene_path = os.path.join(repo_root_from_script(), scene_path)

    env = AlexStandEnv(
        mjcf_path=scene_path,
        episode_steps=args.episode_steps,
        frame_skip=args.frame_skip,
        hand_disturbance_scale=args.hand_disturbance_scale,
    )
    obs_rms = RunningNorm(env.obs_dim)

    model = ActorCritic(env.obs_dim, env.nu, hidden=args.hidden).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.outdir, exist_ok=True)
    cfg = PPOCfg(
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        lr=args.lr,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
    )

    obs = env.reset()
    ep_ret, ep_len = 0.0, 0
    global_step = 0
    episode_count = 0
    success_count = 0
    start_time = time.time()

    viewer = None
    if args.render:
        try:
            viewer = mujoco.viewer.launch_passive(env.model, env.data)
        except RuntimeError as exc:
            msg = str(exc)
            if "launch_passive" in msg and "mjpython" in msg:
                print(
                    "render disabled: launch_passive on macOS requires mjpython.\n"
                    "run instead:\n"
                    "  mjpython training/alex-stand.py --render"
                )
                viewer = None
            else:
                raise

    while global_step < cfg.total_steps:
        obs_buf = np.zeros((cfg.rollout_steps, env.obs_dim), dtype=np.float32)
        act_buf = np.zeros((cfg.rollout_steps, env.nu), dtype=np.float32)
        logp_buf = np.zeros(cfg.rollout_steps, dtype=np.float32)
        rew_buf = np.zeros(cfg.rollout_steps, dtype=np.float32)
        done_buf = np.zeros(cfg.rollout_steps, dtype=np.float32)
        val_buf = np.zeros(cfg.rollout_steps, dtype=np.float32)

        for t in range(cfg.rollout_steps):
            obs_rms.update(obs[None, :])
            nobs = obs_rms.normalize(obs).astype(np.float32)
            obs_t = torch.from_numpy(nobs).to(device).unsqueeze(0)
            with torch.no_grad():
                action_t, logp_t, value_t = model.act(obs_t)
            action = action_t.squeeze(0).cpu().numpy()
            logp = float(logp_t.item())
            value = float(value_t.item())

            next_obs, reward, done, info = env.step(action)
            if viewer is not None:
                if viewer.is_running():
                    viewer.sync()
                else:
                    viewer.close()
                    viewer = None

            obs_buf[t] = nobs
            act_buf[t] = action
            logp_buf[t] = logp
            rew_buf[t] = reward
            done_buf[t] = float(done)
            val_buf[t] = value

            obs = next_obs
            ep_ret += reward
            ep_len += 1
            global_step += 1

            if done:
                episode_count += 1
                if bool(info.get("timeout", False)) and not bool(info.get("fallen", True)):
                    success_count += 1
                if episode_count % args.log_every_episodes == 0:
                    print(f"episodes={episode_count} standing_success={success_count}")
                obs = env.reset()
                ep_ret, ep_len = 0.0, 0

            if global_step >= cfg.total_steps:
                break

        nobs = obs_rms.normalize(obs).astype(np.float32)
        with torch.no_grad():
            _, _, last_val_t = model.forward(
                torch.from_numpy(nobs).to(device).unsqueeze(0)
            )
        last_val = float(last_val_t.item())
        adv_buf, ret_buf = compute_gae(
            rew_buf, val_buf, done_buf, last_val, cfg.gamma, cfg.gae_lambda
        )
        adv_buf = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)

        obs_t = torch.from_numpy(obs_buf).to(device)
        act_t = torch.from_numpy(act_buf).to(device)
        logp_old_t = torch.from_numpy(logp_buf).to(device)
        adv_t = torch.from_numpy(adv_buf).to(device)
        ret_t = torch.from_numpy(ret_buf).to(device)

        n = obs_t.shape[0]
        idx = np.arange(n)
        for _ in range(cfg.update_epochs):
            np.random.shuffle(idx)
            for s in range(0, n, cfg.minibatch_size):
                mb = idx[s : s + cfg.minibatch_size]
                mb_obs = obs_t[mb]
                mb_act = act_t[mb]
                mb_logp_old = logp_old_t[mb]
                mb_adv = adv_t[mb]
                mb_ret = ret_t[mb]

                logp, entropy, value = model.evaluate(mb_obs, mb_act)
                ratio = torch.exp(logp - mb_logp_old)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * ((value - mb_ret) ** 2).mean()
                entropy_loss = -entropy.mean()
                loss = policy_loss + cfg.value_coef * value_loss + cfg.entropy_coef * entropy_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

        if global_step % args.save_every < cfg.rollout_steps:
            ckpt = {
                "model": model.state_dict(),
                "obs_mean": obs_rms.mean,
                "obs_var": obs_rms.var,
                "obs_count": obs_rms.count,
                "step": global_step,
            }
            path = os.path.join(args.outdir, f"alex_ppo_step_{global_step}.pt")
            torch.save(ckpt, path)
            elapsed = time.time() - start_time
            print(f"saved={path} elapsed={elapsed/60.0:.1f}min")

    final_path = os.path.join(args.outdir, "alex_ppo_final.pt")
    torch.save({"model": model.state_dict()}, final_path)
    print(f"training_done model={final_path}")
    if viewer is not None:
        viewer.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        type=str,
        default="scenes/alex-scenes/scene_alex_v1_full_body_mjx.xml",
    )
    parser.add_argument("--outdir", type=str, default="training/checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--total-steps", type=int, default=1_500_000)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--episode-steps", type=int, default=1000)
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--hand-disturbance-scale", type=float, default=0.06)
    parser.add_argument("--save-every", type=int, default=50_000)
    parser.add_argument("--log-every-episodes", type=int, default=1000)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
