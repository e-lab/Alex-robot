import time
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from train import AlexHumanoidEnv, MODELS_DIR


def evaluate():
    model_path = MODELS_DIR / "best" / "best_model.zip"
    if not model_path.exists():
        model_path = MODELS_DIR / "alex_humanoid_walk_final.zip"

    if not model_path.exists():
        print("No model found. Train first.")
        return

    stats_path = MODELS_DIR / "vec_normalize_final.pkl"

    env = AlexHumanoidEnv()  # no render_mode — viewer is managed explicitly below
    vec_env = DummyVecEnv([lambda: env])
    if stats_path.exists():
        vec_env = VecNormalize.load(str(stats_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = PPO.load(str(model_path), env=vec_env)
    print(f"Loaded model from {model_path}")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        episode = 0
        while viewer.is_running():
            obs = vec_env.reset()
            episode_reward = 0.0
            steps = 0
            episode += 1

            while viewer.is_running():
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = vec_env.step(action)
                episode_reward += reward[0]
                steps += 1
                viewer.sync()
                time.sleep(env.dt)  # real-time playback (frame_skip * timestep)

                if done[0]:
                    break

            print(f"Episode {episode:3d} | steps: {steps:4d} | reward: {episode_reward:.1f}")


if __name__ == "__main__":
    evaluate()
