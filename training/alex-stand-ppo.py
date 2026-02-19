#!/usr/bin/env python3
"""
PPO training for Alex humanoid standing task.

Uses position actuators (MuJoCo handles PD internally) + stable-baselines3 PPO.
The policy outputs desired joint positions; position actuators produce the torques.
action = 0  →  hold the stand-prep pose exactly.

Dependencies:
    pip install stable-baselines3[extra] gymnasium mujoco torch

Training:
    python training/alex-stand-ppo.py

Evaluate a saved model:
    python training/alex-stand-ppo.py --eval rl_models/best/best_model \
        --vec-norm rl_models/vec_normalize_final.pkl

TensorBoard:
    tensorboard --logdir training/rl_models/tensorboard
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

REPO_ROOT  = Path(__file__).resolve().parent.parent
SCENE_XML  = REPO_ROOT / "scenes/alex-scenes/scene_alex_v1_train.xml"
MODELS_DIR = Path(__file__).resolve().parent / "rl_models"

# ─── Stand-prep reference pose ────────────────────────────────────────────────

# Starting reference pose: near-upright stance is more stable for RL to learn from.
# Arms are folded (elbow bent) to lower the CoM and prevent arm swing instability.
# Legs are nearly straight with a slight forward lean (hip_y > 0) so the CoM
# sits more centrally over the foot support polygon.
STAND_PREP_TARGET: dict[str, float] = {
    "spine_z":      0.00,
    "neck_y":       0.00,   "neck_z":       0.00,
    "left_hip_z":   0.00,   "right_hip_z":  0.00,
    "left_hip_x":   0.05,   "right_hip_x": -0.05,
    "left_hip_y":   0.05,   "right_hip_y":  0.05,   # very slight forward lean
    "left_knee":    0.10,   "right_knee":   0.10,   # nearly straight knees
    "left_ankle_y":-0.03,   "right_ankle_y":-0.03,
    "left_ankle_x": 0.00,   "right_ankle_x": 0.00,
    "left_shoulder_x":  0.20,  "right_shoulder_x": -0.20,
    "left_shoulder_z": -0.20,  "right_shoulder_z":  0.20,
    "left_shoulder_y":  0.30,  "right_shoulder_y":  0.30,
    "left_elbow":  -0.80,   "right_elbow":  -0.80,  # arms at sides, slightly bent
    "left_wrist_x": 0.00,   "right_wrist_x": 0.00,
    "left_wrist_z": 0.00,   "right_wrist_z": 0.00,
}

# ─── Task config ──────────────────────────────────────────────────────────────

UPRIGHT_MIN_COS = 0.30   # terminate if tilted > ~73°  (gives agent more room to learn)
MAX_STEPS       = 1000   # max steps per episode  (5 s at 5 ms/step)
ACTION_SCALE    = 0.30   # max joint deviation from stand pose (rad)

# Reward weights
W_HEIGHT   = 5.0
W_UPRIGHT  = 3.0
W_ALIVE    = 1.0
W_CTRL     = 0.05
W_LIN_VEL  = 0.10
W_POSE     = 0.50
W_SMOOTH   = 0.10
FALL_PENALTY = 10.0


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
    """
    Find the pelvis z that puts the lowest foot-collision corner exactly on the floor.
    Iterates: set z=1.0, run forward, measure deepest foot penetration, adjust.
    """
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 1.0
    data.qpos[3] = 1.0
    data.qpos[4:7] = 0.0
    data.qpos[7:] = 0.0
    for nm, q in joint_pose.items():
        if nm in name2id:
            data.qpos[qadr[name2id[nm]]] = q
    mujoco.mj_forward(model, data)

    # Find the lowest point of any foot collision geom
    lowest = np.inf
    for i in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if "foot_collision" in gname:
            # Box geom: lowest point = center_z - half_height_z
            gz = float(data.geom_xpos[i, 2])
            half_z = float(model.geom_size[i, 2])
            # Account for tilt (rotation matrix row 2 col 2 = cos of z-tilt)
            rot = data.geom_xmat[i].reshape(3, 3)
            # Project box half-extents to world z
            half_world_z = (abs(rot[2, 0]) * float(model.geom_size[i, 0]) +
                            abs(rot[2, 1]) * float(model.geom_size[i, 1]) +
                            abs(rot[2, 2]) * float(model.geom_size[i, 2]))
            lowest = min(lowest, gz - half_world_z)

    # Offset pelvis so foot bottom = 0 (add tiny margin to stay off ground)
    return 1.0 - lowest + 0.002


# ─── Environment ──────────────────────────────────────────────────────────────

class AlexStandEnv(gym.Env):
    """
    Alex humanoid stand-and-balance task with position actuators.

    Observation (float32, 92 dims):
        pelvis_z (1) | pelvis_quat wxyz (4) | pelvis_linvel (3) | pelvis_angvel (3)
        | joint_pos_offset_from_stand (27) | joint_vel (27) | prev_action (27)

    Action (float32, 27 dims, clipped to [-1, 1]):
        Desired joint-position offsets from stand-prep pose (× ACTION_SCALE rad).
        Written directly to position-actuator ctrl; MuJoCo handles the torques.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 200}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        random_init: bool = True,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.random_init = random_init

        MODELS_DIR.mkdir(exist_ok=True)

        self._model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self._data  = mujoco.MjData(self._model)

        nu = self._model.nu
        self._act_names: list[str] = []
        self._qadr = np.empty(nu, dtype=np.int32)
        self._vadr = np.empty(nu, dtype=np.int32)

        for a in range(nu):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            self._act_names.append(name)
            j = self._model.actuator_trnid[a, 0]
            self._qadr[a] = self._model.jnt_qposadr[j]
            self._vadr[a] = self._model.jnt_dofadr[j]

        self._name2id: dict[str, int] = {n: i for i, n in enumerate(self._act_names)}

        # ── Stand-prep reference and correct initial height ───────────────────
        self._stand_joint_q = np.zeros(nu, dtype=np.float64)
        for name, q in STAND_PREP_TARGET.items():
            if name in self._name2id:
                self._stand_joint_q[self._name2id[name]] = q

        # Find the pelvis height that puts feet exactly on the floor
        self._pelvis_z_init = _find_floor_height(
            self._model, self._data, STAND_PREP_TARGET,
            self._name2id, self._qadr,
        )

        # Build the full initial qpos vector
        self._stand_qpos = np.zeros(self._model.nq, dtype=np.float64)
        self._stand_qpos[2] = self._pelvis_z_init
        self._stand_qpos[3] = 1.0   # quaternion w
        for name, q in STAND_PREP_TARGET.items():
            if name in self._name2id:
                self._stand_qpos[self._qadr[self._name2id[name]]] = q

        self._stand_qpos_joints = self._stand_qpos[7:].copy()

        # ── Ctrl range of position actuators (= allowed desired positions) ────
        self._ctrl_lo = self._model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_hi = self._model.actuator_ctrlrange[:, 1].copy()

        # ── Body ids ──────────────────────────────────────────────────────────
        self._pelvis_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link"
        )

        # ── Spaces ────────────────────────────────────────────────────────────
        nq_j = self._model.nq - 7
        nv_j = self._model.nv - 6
        obs_dim = 1 + 4 + 3 + 3 + nq_j + nv_j + nu
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(nu,), dtype=np.float32)

        self._prev_action = np.zeros(nu, dtype=np.float32)
        self._step_count  = 0
        self._viewer      = None

        # Report for debug
        print(f"[AlexStandEnv] scene: {SCENE_XML.name}")
        print(f"[AlexStandEnv] pelvis_z_init (auto): {self._pelvis_z_init:.4f}")

    # ── Observation ───────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        pid = self._pelvis_id
        return np.concatenate([
            [self._data.xpos[pid, 2]],
            self._data.xquat[pid],
            self._data.qvel[:3],
            self._data.qvel[3:6],
            self._data.qpos[7:] - self._stand_qpos_joints,
            self._data.qvel[6:],
            self._prev_action,
        ]).astype(np.float32)

    # ── Termination ───────────────────────────────────────────────────────────

    def _is_terminated(self) -> bool:
        pid     = self._pelvis_id
        # Only terminate on orientation — sinking crouch is penalised by height reward,
        # but we don't cut the episode short so the agent can learn to recover.
        # Use an absolute floor as a last-resort catch for catastrophic falls.
        if self._data.xpos[pid, 2] < 0.35:
            return True
        rot     = self._data.xmat[pid].reshape(3, 3)
        upright = float(rot[2, 2])
        return upright < UPRIGHT_MIN_COS

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self, action: np.ndarray) -> tuple[float, dict]:
        pid = self._pelvis_id
        pelvis_z = float(self._data.xpos[pid, 2])
        rot      = self._data.xmat[pid].reshape(3, 3)
        upright  = float(rot[2, 2])

        r_height  = W_HEIGHT  * float(np.exp(-8.0 * (pelvis_z - self._pelvis_z_init) ** 2))
        r_upright = W_UPRIGHT * upright
        r_alive   = W_ALIVE
        r_ctrl    = -W_CTRL   * float(np.dot(action, action))
        r_linvel  = -W_LIN_VEL * float(np.dot(self._data.qvel[:3], self._data.qvel[:3]))

        q_err = float(np.mean(
            (self._data.qpos[self._qadr] - self._stand_joint_q) ** 2
        ))
        r_pose   = -W_POSE * q_err
        r_smooth = -W_SMOOTH * float(np.dot(action - self._prev_action,
                                             action - self._prev_action))

        total = r_height + r_upright + r_alive + r_ctrl + r_linvel + r_pose + r_smooth
        return float(total), dict(
            r_height=r_height, r_upright=r_upright, r_alive=r_alive,
            r_ctrl=r_ctrl, r_linvel=r_linvel, r_pose=r_pose, r_smooth=r_smooth,
            pelvis_z=pelvis_z, upright=upright,
        )

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self._model, self._data)

        self._data.qpos[:] = self._stand_qpos.copy()
        # Set ctrl to stand pose so position actuators start in the right state
        self._data.ctrl[:] = self._stand_joint_q.copy()

        if self.random_init:
            # Small random joint perturbations
            self._data.qpos[7:] += self.np_random.uniform(
                -0.03, 0.03, size=self._model.nq - 7
            )
            # Small random pelvis height ±1 cm
            self._data.qpos[2] += float(self.np_random.uniform(-0.01, 0.01))
            # Random yaw ±5°
            yaw = float(self.np_random.uniform(-0.08, 0.08))
            dq  = _quat_from_euler_xyz(0.0, 0.0, yaw)
            self._data.qpos[3:7] = _quat_mul(self._data.qpos[3:7].copy(), dq)

        self._data.qvel[:] = 0.0
        mujoco.mj_forward(self._model, self._data)

        self._prev_action = np.zeros(self._model.nu, dtype=np.float32)
        self._step_count  = 0
        return self._get_obs(), {}

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # Map action [-1,1] → desired joint positions around stand pose
        q_des = self._stand_joint_q + action * ACTION_SCALE
        q_des = np.clip(q_des, self._ctrl_lo, self._ctrl_hi)
        self._data.ctrl[:] = q_des

        mujoco.mj_step(self._model, self._data)
        self._step_count += 1

        obs          = self._get_obs()
        reward, info = self._compute_reward(action)
        terminated   = self._is_terminated()
        truncated    = self._step_count >= MAX_STEPS

        if terminated:
            reward -= FALL_PENALTY
            info["fall"] = True

        self._prev_action = action.copy()
        return obs, reward, terminated, truncated, info

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer as _mv
                self._viewer = _mv.launch_passive(self._model, self._data)
            self._viewer.sync()
        elif self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self._model, height=480, width=640)
            renderer.update_scene(self._data)
            return renderer.render()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


# ─── Training ─────────────────────────────────────────────────────────────────

def train(
    total_timesteps: int = 10_000_000,
    n_envs: int = 8,
) -> None:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    MODELS_DIR.mkdir(exist_ok=True)

    try:
        vec_env = make_vec_env(
            AlexStandEnv, n_envs=n_envs,
            env_kwargs={"random_init": True},
            vec_env_cls=SubprocVecEnv,
        )
    except Exception:
        print("SubprocVecEnv failed, falling back to DummyVecEnv.")
        from stable_baselines3.common.vec_env import DummyVecEnv
        vec_env = make_vec_env(
            AlexStandEnv, n_envs=n_envs,
            env_kwargs={"random_init": True},
            vec_env_cls=DummyVecEnv,
        )

    vec_env = VecNormalize(
        vec_env, norm_obs=True, norm_reward=True,
        clip_obs=10.0, clip_reward=10.0, gamma=0.99,
    )

    eval_env = make_vec_env(AlexStandEnv, n_envs=1,
                            env_kwargs={"random_init": False})
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            clip_obs=10.0, training=False)

    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=torch.nn.ELU,
            log_std_init=-1.0,
            ortho_init=True,
        ),
        verbose=1,
        tensorboard_log=str(MODELS_DIR / "tensorboard"),
        seed=42,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=max(100_000 // n_envs, 1),
        save_path=str(MODELS_DIR / "checkpoints"),
        name_prefix="alex_stand",
        save_vecnormalize=True,
        verbose=1,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(MODELS_DIR / "best"),
        log_path=str(MODELS_DIR / "eval_logs"),
        eval_freq=max(50_000 // n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    print(f"\nTraining for {total_timesteps:,} timesteps across {n_envs} envs.")
    print("Press Ctrl-C to stop early.\n")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[ckpt_cb, eval_cb],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted.")

    model.save(str(MODELS_DIR / "alex_stand_final"))
    vec_env.save(str(MODELS_DIR / "vec_normalize_final.pkl"))
    print(f"\nSaved to {MODELS_DIR}/")
    vec_env.close()
    eval_env.close()


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(
    model_path: str,
    vec_norm_path: Optional[str] = None,
    n_episodes: int = 5,
) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    env     = AlexStandEnv(render_mode="human", random_init=False)
    vec_env = DummyVecEnv([lambda: env])

    if vec_norm_path and Path(vec_norm_path).exists():
        vec_env = VecNormalize.load(vec_norm_path, vec_env)
        vec_env.training    = False
        vec_env.norm_reward = False
    else:
        print("No VecNormalize stats — observations will not be normalised.")

    model = PPO.load(model_path, env=vec_env)
    dt    = env._model.opt.timestep

    for ep in range(1, n_episodes + 1):
        obs          = vec_env.reset()
        total_reward = 0.0
        info_last    = {}
        for step in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            total_reward += float(reward[0])
            info_last     = info[0]
            env.render()
            time.sleep(dt)
            if done[0]:
                break

        fell = info_last.get("fall", False)
        print(
            f"  Episode {ep:2d}  reward={total_reward:8.1f}  "
            f"pelvis_z={info_last.get('pelvis_z', 0.0):.3f}  "
            f"upright={info_last.get('upright', 0.0):.3f}  "
            f"steps={step+1:4d}  {'FELL' if fell else 'OK'}"
        )

    env.close()


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", metavar="MODEL_PATH", default=None)
    ap.add_argument("--vec-norm", metavar="PKL_PATH", default=None)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--timesteps", type=int, default=10_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    args = ap.parse_args()

    if args.eval:
        evaluate(args.eval, args.vec_norm, args.episodes)
    else:
        train(total_timesteps=args.timesteps, n_envs=args.n_envs)
