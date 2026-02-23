import argparse
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecEnvWrapper
from pathlib import Path


class CastObsToFloat32(VecEnvWrapper):
    """Cast observations to float32 to match warp-trained model expectations."""
    def __init__(self, venv):
        obs_space = spaces.Box(
            low=venv.observation_space.low.astype(np.float32),
            high=venv.observation_space.high.astype(np.float32),
            dtype=np.float32,
        )
        super().__init__(venv, observation_space=obs_space)

    def reset(self):
        return self.venv.reset().astype(np.float32)

    def step_wait(self):
        obs, rew, done, info = self.venv.step_wait()
        return obs.astype(np.float32), rew, done, info

ENV_NAME = "Humanoid-v5"
SCRIPT_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument("--warp", action="store_true",
                    help="Evaluate model trained with mujoco-warp (rl_models_warp/)")
parser.add_argument("--episodes", type=int, default=5,
                    help="Number of evaluation episodes")
parser.add_argument("--no-render", action="store_true",
                    help="Disable rendering")
args = parser.parse_args()

if args.warp:
    MODELS_DIR = SCRIPT_DIR / "rl_models_warp"
    MODEL_PATH = MODELS_DIR / "best" / "best_model.zip"
    if not MODEL_PATH.exists():
        MODEL_PATH = MODELS_DIR / "ppo_humanoid_warp_final.zip"
    STATS_PATH = None  # warp env does not use VecNormalize
else:
    MODELS_DIR = SCRIPT_DIR / "rl_models"
    MODEL_PATH = MODELS_DIR / "best" / "best_model.zip"
    STATS_PATH = MODELS_DIR / "vec_normalize_final.pkl"

render_mode = None if args.no_render else "human"


def make_env(render_mode=None):
    def _init():
        return gym.make(ENV_NAME, render_mode=render_mode)
    return _init


eval_env = DummyVecEnv([make_env(render_mode=render_mode)])

if not args.warp and STATS_PATH and os.path.exists(STATS_PATH):
    eval_env = VecNormalize.load(STATS_PATH, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

if args.warp:
    eval_env = CastObsToFloat32(eval_env)

if not MODEL_PATH.exists():
    print(f"Model not found at {MODEL_PATH}")
    exit(1)

model = PPO.load(MODEL_PATH, env=eval_env)
print(f"Loaded {'warp' if args.warp else 'standard'} model from {MODEL_PATH}")

rewards = []
for episode in range(args.episodes):
    obs = eval_env.reset()
    done = False
    episode_reward = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        episode_reward += reward[0]
    rewards.append(episode_reward)
    print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

print(f"\nMean reward over {args.episodes} episodes: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
eval_env.close()
