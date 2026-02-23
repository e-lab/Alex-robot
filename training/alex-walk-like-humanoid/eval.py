import time
import gymnasium as gym
import mujoco
import numpy as np
import mujoco.viewer
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from train_ppo import AlexHumanoidWalkEnv, MODELS_DIR

def evaluate():
    model_path = MODELS_DIR / "best" / "best_model.zip"
    if not model_path.exists():
        model_path = MODELS_DIR / "alex_humanoid_walk_final.zip"
    stats_path = MODELS_DIR / "vec_normalize_final.pkl"
    
    env = AlexHumanoidWalkEnv()
    vec_env = DummyVecEnv([lambda: env])
    if stats_path.exists():
        vec_env = VecNormalize.load(str(stats_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    if not model_path.exists():
        print("No model found.")
        return

    model = PPO.load(str(model_path), env=vec_env)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            obs = vec_env.reset()
            done = False
            while not done and viewer.is_running():
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = vec_env.step(action)
                viewer.sync()
                time.sleep(env.model.opt.timestep)

if __name__ == "__main__":
    evaluate()
