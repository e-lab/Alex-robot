import os
import numpy as np
import warp as wp
import mujoco
import mujoco_warp as mjw
from pathlib import Path
from typing import Optional

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "rl_models_warp"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Humanoid XML path (from gymnasium) ───────────────────────────────────────
HUMANOID_XML = str(Path(gym.__file__).parent / "envs/mujoco/assets/humanoid.xml")

# ─── Hyperparameters ──────────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 100_000_000
N_ENVS = 2048
BATCH_SIZE = 16384
N_STEPS = 16           # steps per rollout per env
LEARNING_RATE = 3e-4
ENT_COEF = 0.01
GAMMA = 0.99
GAE_LAMBDA = 0.95

# ─── Humanoid-v5 constants (matching gymnasium defaults) ──────────────────────
FRAME_SKIP = 5
HEALTHY_Z_MIN = 1.0
HEALTHY_Z_MAX = 2.0
HEALTHY_REWARD = 5.0
FORWARD_REWARD_WEIGHT = 1.25
CTRL_COST_WEIGHT = 0.1
CONTACT_COST_WEIGHT = 5e-7
CONTACT_COST_MAX = 10.0
RESET_NOISE_SCALE = 1e-2
MAX_EPISODE_STEPS = 1000


class HumanoidWarpVecEnv(VecEnv):
    """
    Vectorized Humanoid environment using mujoco-warp for massively parallel
    GPU simulation. Implements the SB3 VecEnv interface.

    Matches gymnasium Humanoid-v5 observation (348-dim) and reward.
    """

    def __init__(self, n_envs: int = 2048, device: str = "cuda:0"):
        self.n_envs = n_envs
        self.device = device

        # ── Load and patch the MuJoCo model ───────────────────────────────────
        self.mjm = mujoco.MjModel.from_xml_path(HUMANOID_XML)
        # PGS solver is unsupported by mujoco-warp; use CG instead
        self.mjm.opt.solver = mujoco.mjtSolver.mjSOL_CG

        self._nq = self.mjm.nq          # 24
        self._nv = self.mjm.nv          # 23
        self._nu = self.mjm.nu          # 17
        self._nbody = self.mjm.nbody    # 14
        self._qpos0 = self.mjm.qpos0.copy()
        self._dt = self.mjm.opt.timestep * FRAME_SKIP

        # ── Put model on GPU, create parallel data ─────────────────────────────
        with wp.ScopedDevice(device):
            self.m = mjw.put_model(self.mjm)
            self.d = mjw.make_data(self.mjm, nworld=n_envs)

        # ── Observation / action spaces (match Humanoid-v5) ───────────────────
        # obs: qpos[2:](22) + qvel(23) + cinert[1:](13*10) + cvel[1:](13*6)
        #      + qfrc_actuator[6:](17) + cfrc_ext[1:](13*6) = 348
        obs_dim = (self._nq - 2) + self._nv + (self._nbody - 1) * 10 + \
                  (self._nbody - 1) * 6 + (self._nv - 6) + (self._nbody - 1) * 6
        obs_high = np.full(obs_dim, np.inf, dtype=np.float32)
        act_high = np.ones(self._nu, dtype=np.float32) * 0.4  # Humanoid-v5 bounds

        observation_space = gym.spaces.Box(-obs_high, obs_high, dtype=np.float32)
        action_space = gym.spaces.Box(-act_high, act_high, dtype=np.float32)

        super().__init__(n_envs, observation_space, action_space)

        # ── Episode tracking ───────────────────────────────────────────────────
        self._episode_steps = np.zeros(n_envs, dtype=np.int32)
        self._prev_com_x = np.zeros(n_envs, dtype=np.float32)

        # ── Pending actions ────────────────────────────────────────────────────
        self._actions: Optional[np.ndarray] = None

        # ── Initial reset ──────────────────────────────────────────────────────
        self._reset_all()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _reset_all(self):
        """Reset all worlds to initial state with small noise."""
        rng = np.random.default_rng()
        qpos = np.tile(self._qpos0, (self.n_envs, 1)).astype(np.float32)
        qpos += rng.uniform(-RESET_NOISE_SCALE, RESET_NOISE_SCALE,
                            qpos.shape).astype(np.float32)
        qvel = rng.uniform(-RESET_NOISE_SCALE, RESET_NOISE_SCALE,
                           (self.n_envs, self._nv)).astype(np.float32)
        with wp.ScopedDevice(self.device):
            wp.copy(self.d.qpos, wp.array(qpos, dtype=wp.float32, device=self.device))
            wp.copy(self.d.qvel, wp.array(qvel, dtype=wp.float32, device=self.device))
            mjw.forward(self.m, self.d)
        self._episode_steps[:] = 0
        com = self.d.subtree_com.numpy()[:, 0, 0]  # x of world body COM
        self._prev_com_x = com.astype(np.float32)

    def _reset_envs(self, mask: np.ndarray):
        """Reset specific worlds identified by boolean mask."""
        if not mask.any():
            return
        rng = np.random.default_rng()
        indices = np.where(mask)[0]
        n_reset = len(indices)

        # Read current qpos/qvel from GPU
        qpos = self.d.qpos.numpy().copy()
        qvel = self.d.qvel.numpy().copy()

        # Apply noise to reset worlds
        qpos[indices] = (self._qpos0 +
                         rng.uniform(-RESET_NOISE_SCALE, RESET_NOISE_SCALE,
                                     (n_reset, self._nq))).astype(np.float32)
        qvel[indices] = rng.uniform(-RESET_NOISE_SCALE, RESET_NOISE_SCALE,
                                    (n_reset, self._nv)).astype(np.float32)

        with wp.ScopedDevice(self.device):
            wp.copy(self.d.qpos, wp.array(qpos, dtype=wp.float32, device=self.device))
            wp.copy(self.d.qvel, wp.array(qvel, dtype=wp.float32, device=self.device))

        self._episode_steps[mask] = 0

    def _get_obs(self) -> np.ndarray:
        """Compute (n_envs, 348) observation array from GPU data."""
        qpos = self.d.qpos.numpy()           # (N, 24)
        qvel = self.d.qvel.numpy()           # (N, 23)
        cinert = self.d.cinert.numpy()       # (N, 14, 10)
        cvel = self.d.cvel.numpy()           # (N, 14, 6)
        qfrc_act = self.d.qfrc_actuator.numpy()  # (N, 23)
        cfrc_ext = self.d.cfrc_ext.numpy()   # (N, 14, 6)

        obs = np.concatenate([
            qpos[:, 2:],                     # exclude x,y: (N, 22)
            qvel,                            # (N, 23)
            cinert[:, 1:].reshape(self.n_envs, -1),   # (N, 13*10=130)
            cvel[:, 1:].reshape(self.n_envs, -1),     # (N, 13*6=78)
            qfrc_act[:, 6:],                 # (N, 17)
            cfrc_ext[:, 1:].reshape(self.n_envs, -1), # (N, 13*6=78)
        ], axis=1).astype(np.float32)
        return obs

    def _compute_rewards(self, actions: np.ndarray,
                         prev_com_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-env rewards and done flags."""
        # Forward velocity from COM x displacement
        com_x = self.d.subtree_com.numpy()[:, 0, 0]  # world body x
        x_velocity = (com_x - prev_com_x) / self._dt

        # Health: torso z between 1.0 and 2.0
        qpos_z = self.d.qpos.numpy()[:, 2]
        is_healthy = (qpos_z > HEALTHY_Z_MIN) & (qpos_z < HEALTHY_Z_MAX)

        forward_reward = FORWARD_REWARD_WEIGHT * x_velocity
        healthy_reward = HEALTHY_REWARD * is_healthy.astype(np.float32)

        ctrl_cost = CTRL_COST_WEIGHT * np.sum(np.square(actions), axis=1)

        cfrc_ext = self.d.cfrc_ext.numpy()   # (N, 14, 6)
        contact_sq = np.sum(np.square(cfrc_ext[:, 1:]), axis=(1, 2))
        contact_cost = np.clip(CONTACT_COST_WEIGHT * contact_sq,
                               0.0, CONTACT_COST_MAX)

        rewards = (forward_reward + healthy_reward
                   - ctrl_cost - contact_cost).astype(np.float32)

        # Episode end: unhealthy OR max steps reached
        dones = (~is_healthy) | (self._episode_steps >= MAX_EPISODE_STEPS)
        return rewards, dones.astype(bool), com_x.astype(np.float32)

    # ── VecEnv interface ───────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        self._reset_all()
        return self._get_obs()

    def step_async(self, actions: np.ndarray):
        self._actions = actions

    def step_wait(self):
        actions = self._actions
        assert actions is not None

        # Record COM before step
        prev_com_x = self._prev_com_x.copy()

        # Write ctrl to GPU and step physics (frame_skip times)
        ctrl = actions.astype(np.float32)
        with wp.ScopedDevice(self.device):
            wp.copy(self.d.ctrl,
                    wp.array(ctrl, dtype=wp.float32, device=self.device))
            for _ in range(FRAME_SKIP):
                mjw.step(self.m, self.d)

        self._episode_steps += 1

        rewards, dones, new_com_x = self._compute_rewards(actions, prev_com_x)
        self._prev_com_x = new_com_x

        obs = self._get_obs()

        # Reset done environments before returning final obs
        if dones.any():
            # Store terminal observations
            terminal_obs = obs[dones]
            self._reset_envs(dones)
            with wp.ScopedDevice(self.device):
                mjw.forward(self.m, self.d)
            obs_after_reset = self._get_obs()
            obs[dones] = obs_after_reset[dones]
            # Update COM for reset envs
            com_after = self.d.subtree_com.numpy()[:, 0, 0]
            self._prev_com_x[dones] = com_after[dones]

        infos = [{} for _ in range(self.n_envs)]
        if dones.any():
            done_indices = np.where(dones)[0]
            for rank, i in enumerate(done_indices):
                infos[i]["terminal_observation"] = terminal_obs[rank]

        return obs, rewards, dones, infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name)] * (self.n_envs if indices is None else len(indices))

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [None] * (self.n_envs if indices is None else len(indices))

    def env_is_wrapped(self, wrapper_class, indices=None):
        n = self.n_envs if indices is None else len(indices)
        return [False] * n

    def seed(self, seed=None):
        return [seed] * self.n_envs

    def render(self, mode="human"):
        pass


# ─── Training ─────────────────────────────────────────────────────────────────

def train():
    print("Initializing mujoco-warp vectorized environment...")
    vec_env = HumanoidWarpVecEnv(n_envs=N_ENVS)

    print(f"\n--- Mujoco-Warp Training Configuration ---")
    print(f"Device:       {vec_env.device}")
    print(f"Environments: {N_ENVS}")
    print(f"Obs dim:      {vec_env.observation_space.shape}")
    print(f"Act dim:      {vec_env.action_space.shape}")
    print(f"Batch size:   {BATCH_SIZE}")
    print(f"------------------------------------------\n")

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        n_steps=N_STEPS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        ent_coef=ENT_COEF,
        tensorboard_log=str(MODELS_DIR / "tensorboard"),
        device="cuda",
    )

    # Eval env: standard CPU gymnasium for simplicity
    from stable_baselines3.common.vec_env import SubprocVecEnv
    eval_env = make_vec_env("Humanoid-v5", n_envs=4, vec_env_cls=SubprocVecEnv)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(MODELS_DIR / "best"),
        log_path=str(MODELS_DIR / "eval_logs"),
        eval_freq=max(500_000 // N_ENVS, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1_000_000 // N_ENVS, 1),
        save_path=str(MODELS_DIR / "checkpoints"),
        name_prefix="ppo_humanoid_warp",
    )

    print(f"Starting training for {TOTAL_TIMESTEPS:,} timesteps...")
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True,
            tb_log_name="PPO_warp",
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted.")

    model.save(str(MODELS_DIR / "ppo_humanoid_warp_final"))
    print(f"Model saved to {MODELS_DIR}")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    train()
