import os
import time
import gymnasium as gym
import mujoco
import numpy as np
from pathlib import Path
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parents[1]
SCENE_XML  = REPO_ROOT / "scenes/alex-scenes/scene_alex_v1_full_body_mjx.xml"
MODELS_DIR = SCRIPT_DIR / "rl_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Stand-prep reference pose ───────────────────────────────────────────────
STAND_PREP_TARGET: dict[str, float] = {
    "spine_z": 0.0,
    "left_hip_x": 0.1,   "right_hip_x": -0.1,
    "left_hip_y": -0.45,  "right_hip_y": -0.45,
    "left_hip_z": 0.0,    "right_hip_z": 0.0,
    "left_knee_y": 0.7,   "right_knee_y": 0.7,
    "left_ankle_y": -0.28, "right_ankle_y": -0.28,
    "left_ankle_x": 0.0,   "right_ankle_x": 0.0,
    "left_shoulder_y": 0.2, "right_shoulder_y": 0.2,
    "left_shoulder_x": 0.3, "right_shoulder_x": -0.3,
    "left_elbow": -0.5, "right_elbow": -0.5,
}

class AlexHumanoidWalkEnv(gym.Env):
    def __init__(self, render_mode=None):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)
        self.prev_action = np.zeros(self.model.nu)
        obs_size = self._get_obs().shape[0]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)

    def _get_obs(self):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        return np.concatenate([
            qpos[2:3], qpos[3:7], qpos[7:], qvel[:], self.prev_action
        ]).astype(np.float64)

    def step(self, action):
        ctrl_limit = self.model.actuator_ctrlrange
        action = np.clip(action, -1, 1)
        action_scaled = ctrl_limit[:, 0] + (action + 1.0) * 0.5 * (ctrl_limit[:, 1] - ctrl_limit[:, 0])
        self.data.ctrl[:] = action_scaled
        mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        vel_x, vel_y = self.data.qvel[0], self.data.qvel[1]
        z_height = self.data.qpos[2]
        upright = self.data.xmat[self.torso_id].reshape(3, 3)[2, 2]
        
        # Enhanced Reward
        is_healthy = (0.75 < z_height < 1.4) and (upright > 0.6)
        reward_forward = 4.0 * vel_x * max(0.0, upright)
        reward_healthy = 5.0 if is_healthy else 0.0
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_smooth = -0.1 * np.square(action - self.prev_action).sum()
        reward_drift = -1.0 * (abs(vel_y) + abs(self.data.qpos[1]))
        reward_still = -2.0 if (is_healthy and vel_x < 0.1) else 0.0
        
        reward = reward_forward + reward_healthy + reward_ctrl + reward_smooth + reward_drift + reward_still
        self.prev_action = action.copy()
        return obs, reward, not is_healthy, False, {"vel_x": vel_x, "upright": upright}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.prev_action = np.zeros(self.model.nu)
        for name, val in STAND_PREP_TARGET.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.data.qpos[self.model.jnt_qposadr[jid]] = val
        self.data.qpos[:] += np.random.uniform(-0.01, 0.01, self.model.nq)
        self.data.qvel[:] += np.random.uniform(-0.01, 0.01, self.model.nv)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

def train():
    n_envs = 12
    vec_env = make_vec_env(AlexHumanoidWalkEnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    model = PPO(
        "MlpPolicy", vec_env, verbose=1, 
        tensorboard_log=str(MODELS_DIR / "tensorboard"),
        ent_coef=0.01, # Higher exploration
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=1024
    )
    
    eval_env = make_vec_env(AlexHumanoidWalkEnv, n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    eval_callback = EvalCallback(eval_env, best_model_save_path=str(MODELS_DIR / "best"), eval_freq=10000)
    
    print("Starting training with enhanced reward and exploration...")
    try:
        model.learn(total_timesteps=10_000_000, callback=eval_callback, progress_bar=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        
    model.save(str(MODELS_DIR / "alex_humanoid_walk_final"))
    vec_env.save(str(MODELS_DIR / "vec_normalize_final.pkl"))
    vec_env.close()

if __name__ == "__main__":
    train()
