#!/usr/bin/env python3
"""
PPO training for Alex humanoid walking task.

The robot learns to walk by following a periodic reference gait (fixed set of steps)
on the leg pitch axes only (hip_y, knee, ankle_y).

Uses stable-baselines3 PPO.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

# ─── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = Path(__file__).resolve().parents[2]
SCENE_XML  = REPO_ROOT / "scenes/alex-scenes/scene_alex_v1_train.xml"
MODELS_DIR = SCRIPT_DIR / "rl_models"

# ─── Stand-prep reference pose (from test-init-pose.py) ───────────────────────

STAND_PREP_TARGET: dict[str, float] = {
    "spine_z": 0.0,
    "neck_y": 0.0,
    "neck_z": 0.0,
    "left_hip_z": 0.0,
    "right_hip_z": 0.0,
    "left_hip_x": 0.1,
    "right_hip_x": -0.1,
    "left_hip_y": -0.45,
    "right_hip_y": -0.45,
    "left_knee": 0.7,
    "right_knee": 0.7,
    "left_ankle_y": -0.28,
    "right_ankle_y": -0.28,
    "left_ankle_x": 0.0,
    "right_ankle_x": 0.0,
    "left_shoulder_x": 0.4,
    "right_shoulder_x": -0.4,
    "left_shoulder_z": -0.4,
    "right_shoulder_z": 0.4,
    "left_shoulder_y": 0.7,
    "right_shoulder_y": 0.7,
    "left_elbow": -1.9,
    "right_elbow": -1.9,
}

# ─── Task config ──────────────────────────────────────────────────────────────

UPRIGHT_MIN_COS = 0.50   # terminate if tilted > ~60°
MAX_STEPS       = 2000   # 10 s at 5 ms/step
ACTION_SCALE    = 0.20   # max joint deviation from reference gait (rad)
GAIT_PERIOD     = 0.8    # Gait cycle duration (seconds)

# Minimum set of axes for walking (pitch only)
ACTIVE_JOINTS: set[str] = {
    "left_hip_y",       "right_hip_y",
    "left_knee",        "right_knee",
    "left_ankle_y",     "right_ankle_y",
}

# Reward weights
W_FORWARD   = 10.0
W_HEIGHT    = 2.0
W_UPRIGHT   = 3.0
W_ALIVE     = 1.0
W_CTRL      = 0.01
W_DRIFT     = 2.0   # penalty for Y movement
W_SMOOTH    = 0.1
FALL_PENALTY = 50.0

# ─── Reference Gait ───────────────────────────────────────────────────────────

def get_ref_gait_offsets(phase: float) -> dict[str, float]:
    """
    Returns joint position offsets relative to STAND_PREP_TARGET for a given phase [0, 1].
    """
    l_phase = phase
    r_phase = (phase + 0.5) % 1.0
    
    def leg_gait(p):
        # Swing phase: lift and move forward
        if p < 0.5:
            s = np.sin(2 * np.pi * p) # 0 to 1 back to 0
            # Lift hip, bend knee, flex ankle
            return -0.4 * s, 0.6 * s, 0.2 * s
        else:
            # Stance phase: push back
            s = np.sin(2 * np.pi * (p - 0.5))
            return 0.3 * s, -0.1 * s, -0.1 * s

    l_hip, l_knee, l_ankle = leg_gait(l_phase)
    r_hip, r_knee, r_ankle = leg_gait(r_phase)
    
    return {
        "left_hip_y": l_hip, "left_knee": l_knee, "left_ankle_y": l_ankle,
        "right_hip_y": r_hip, "right_knee": r_knee, "right_ankle_y": r_ankle,
    }

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll  / 2), np.sin(roll  / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw   / 2), np.sin(yaw   / 2)
    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ])

def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def _find_floor_height(model: mujoco.MjModel, data: mujoco.MjData,
                       joint_pose: dict[str, float],
                       name2id: dict[str, int],
                       qadr: np.ndarray) -> float:
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 1.5
    data.qpos[3] = 1.0
    for nm, q in joint_pose.items():
        if nm in name2id:
            data.qpos[qadr[name2id[nm]]] = q
    mujoco.mj_forward(model, data)

    lowest = np.inf
    for i in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if "foot_collision" in gname:
            gz = float(data.geom_xpos[i, 2])
            rot = data.geom_xmat[i].reshape(3, 3)
            half_world_z = (abs(rot[2, 0]) * float(model.geom_size[i, 0]) +
                            abs(rot[2, 1]) * float(model.geom_size[i, 1]) +
                            abs(rot[2, 2]) * float(model.geom_size[i, 2]))
            lowest = min(lowest, gz - half_world_z)
    return 1.5 - lowest + 0.002

# ─── Environment ──────────────────────────────────────────────────────────────

class AlexWalkingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 200}

    def __init__(self, render_mode: Optional[str] = None, random_init: bool = True) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.random_init = random_init

        MODELS_DIR.mkdir(exist_ok=True)

        self._model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self._data  = mujoco.MjData(self._model)

        nu = self._model.nu
        self._act_names: list[str] = []
        self._qadr = np.empty(nu, dtype=np.int32)
        for a in range(nu):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            self._act_names.append(name)
            j = self._model.actuator_trnid[a, 0]
            self._qadr[a] = self._model.jnt_qposadr[j]

        self._name2id: dict[str, int] = {n: i for i, n in enumerate(self._act_names)}
        self._active_mask = np.array([n in ACTIVE_JOINTS for n in self._act_names], dtype=bool)

        self._stand_joint_q = np.zeros(nu, dtype=np.float64)
        for name, q in STAND_PREP_TARGET.items():
            if name in self._name2id:
                self._stand_joint_q[self._name2id[name]] = q

        self._pelvis_z_init = _find_floor_height(self._model, self._data, STAND_PREP_TARGET, self._name2id, self._qadr)

        self._ctrl_lo = self._model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_hi = self._model.actuator_ctrlrange[:, 1].copy()
        self._pelvis_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")

        # Obs: pelvis(z, quat, linvel, angvel) + joints(pos, vel) + phase(sin, cos) + prev_action
        nq_j = self._model.nq - 7
        nv_j = self._model.nv - 6
        obs_dim = 1 + 4 + 3 + 3 + nq_j + nv_j + 2 + nu
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(nu,), dtype=np.float32)

        self._prev_action = np.zeros(nu, dtype=np.float32)
        self._step_count  = 0
        self._viewer      = None

    def _get_obs(self) -> np.ndarray:
        pid = self._pelvis_id
        phase = (self._step_count * self._model.opt.timestep / GAIT_PERIOD) % 1.0
        return np.concatenate([
            [self._data.xpos[pid, 2]],
            self._data.xquat[pid],
            self._data.qvel[:3],
            self._data.qvel[3:6],
            self._data.qpos[7:],
            self._data.qvel[6:],
            [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)],
            self._prev_action,
        ]).astype(np.float32)

    def _is_terminated(self) -> bool:
        pid = self._pelvis_id
        if self._data.xpos[pid, 2] < 0.4: return True
        rot = self._data.xmat[pid].reshape(3, 3)
        return float(rot[2, 2]) < UPRIGHT_MIN_COS

    def _compute_reward(self, action: np.ndarray) -> tuple[float, dict]:
        pid = self._pelvis_id
        vel = self._data.qvel[:3]
        upright = float(self._data.xmat[pid].reshape(3, 3)[2, 2])
        height = float(self._data.xpos[pid, 2])

        r_forward = W_FORWARD * vel[0]
        r_drift   = -W_DRIFT * (abs(vel[1]) + abs(self._data.xpos[pid, 1]))
        r_height  = W_HEIGHT * float(np.exp(-10.0 * (height - self._pelvis_z_init) ** 2))
        r_upright = W_UPRIGHT * upright
        r_alive   = W_ALIVE
        r_ctrl    = -W_CTRL * float(np.dot(action, action))
        r_smooth  = -W_SMOOTH * float(np.dot(action - self._prev_action, action - self._prev_action))

        total = r_forward + r_drift + r_height + r_upright + r_alive + r_ctrl + r_smooth
        return float(total), {"vel_x": vel[0]}

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self._model, self._data)
        self._data.qpos[2] = self._pelvis_z_init
        self._data.qpos[3] = 1.0
        for i in range(self._model.nu):
            self._data.qpos[self._qadr[i]] = self._stand_joint_q[i]

        if self.random_init:
            self._data.qpos[7:] += self.np_random.uniform(-0.02, 0.02, size=self._model.nq-7)
            self._data.qvel[:] = self.np_random.uniform(-0.01, 0.01, size=self._model.nv)

        mujoco.mj_forward(self._model, self._data)
        self._prev_action = np.zeros(self._model.nu, dtype=np.float32)
        self._step_count  = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        action = action * self._active_mask

        phase = (self._step_count * self._model.opt.timestep / GAIT_PERIOD) % 1.0
        offsets = get_ref_gait_offsets(phase)
        
        q_ref = self._stand_joint_q.copy()
        for name, val in offsets.items():
            if name in self._name2id:
                q_ref[self._name2id[name]] += val
        
        q_des = q_ref + action * ACTION_SCALE
        self._data.ctrl[:] = np.clip(q_des, self._ctrl_lo, self._ctrl_hi)

        mujoco.mj_step(self._model, self._data)
        self._step_count += 1

        obs          = self._get_obs()
        reward, info = self._compute_reward(action)
        terminated   = self._is_terminated()
        truncated    = self._step_count >= MAX_STEPS

        if terminated: reward -= FALL_PENALTY
        self._prev_action = action.copy()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer as _mv
                self._viewer = _mv.launch_passive(self._model, self._data)
            self._viewer.sync()

def train():
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    import torch

    n_envs = 8
    vec_env = make_vec_env(AlexWalkingEnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    model = PPO(
        "MlpPolicy", vec_env, verbose=1,
        learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10,
        gamma=0.99, gae_lambda=0.95,
        policy_kwargs=dict(activation_fn=torch.nn.ELU, net_arch=dict(pi=[256, 256], vf=[256, 256])),
    )

    print("Training Alex to walk...")
    try:
        model.learn(total_timesteps=5_000_000, progress_bar=True)
    except KeyboardInterrupt:
        pass

    model.save(str(MODELS_DIR / "alex_walking_final"))
    vec_env.save(str(MODELS_DIR / "vec_normalize_walking.pkl"))
    print(f"Model saved to {MODELS_DIR}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    if args.eval:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        env = AlexWalkingEnv(render_mode="human")
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize.load(str(MODELS_DIR / "vec_normalize_walking.pkl"), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        model = PPO.load(str(MODELS_DIR / "alex_walking_final"), env=vec_env)
        
        while True:
            obs = vec_env.reset()
            for _ in range(MAX_STEPS):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _ = vec_env.step(action)
                env.render()
                time.sleep(env._model.opt.timestep)
                if done: break
    else:
        train()
