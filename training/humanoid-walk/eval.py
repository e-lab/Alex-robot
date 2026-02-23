import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from pathlib import Path
import os
import numpy as np

# Paths
ENV_NAME = "Humanoid-v5"
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "rl_models"
MODEL_PATH = MODELS_DIR / "best" / "best_model.zip"
STATS_PATH = MODELS_DIR / "vec_normalize_final.pkl"

def make_env(render_mode=None):
    def _init():
        env = gym.make(ENV_NAME, render_mode=render_mode)
        return env
    return _init

# Load the environment and normalization stats
eval_env = DummyVecEnv([make_env(render_mode='human')])
if os.path.exists(STATS_PATH):
    eval_env = VecNormalize.load(STATS_PATH, eval_env)
    # Stop updating normalization stats during evaluation
    eval_env.training = False
    # Reward normalization is not needed for evaluation
    eval_env.norm_reward = False

# Load the trained model
if os.path.exists(MODEL_PATH):
    model = PPO.load(MODEL_PATH, env=eval_env)
    print(f"Loaded model from {MODEL_PATH}")
else:
    print(f"Model not found at {MODEL_PATH}")
    exit()

# Evaluation Loop
num_episodes = 5
for episode in range(num_episodes):
    obs = eval_env.reset()
    done = False
    episode_reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        episode_reward += reward
    print(f"Episode {episode + 1}: Reward = {episode_reward[0]:.2f}")

eval_env.close()
