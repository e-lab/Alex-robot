import argparse
import base64
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper, VecNormalize

from alex_standup_env import AlexStandupEnv

SCRIPT_DIR = Path(__file__).resolve().parent
MAX_EPISODE_STEPS = 1000


class CastObsToFloat32(VecEnvWrapper):
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


def find_model() -> tuple[Path | None, Path | None]:
    base = SCRIPT_DIR / "rl_models"
    candidates = [base / "best" / "best_model.zip", base / "ppo_alex_standup_final.zip"]
    stats = base / "vec_normalize_final.pkl"

    model_path = next((p for p in candidates if p.exists()), None)
    stats_path = stats if stats.exists() else None
    return model_path, stats_path


def saved_obs_space(model_path: Path) -> tuple[tuple[int, ...], np.dtype]:
    with zipfile.ZipFile(str(model_path), "r") as zf:
        raw_data = zf.read("data")
        if raw_data.startswith(b"{"):
            data = json.loads(raw_data)
            obs_space = data.get("observation_space")
            if isinstance(obs_space, dict):
                if "dtype" in obs_space and "shape" in obs_space:
                    return tuple(obs_space["shape"]), np.dtype(obs_space["dtype"])
                if ":serialized:" in obs_space:
                    space = pickle.loads(base64.b64decode(obs_space[":serialized:"]))
                    return space.shape, space.dtype
            raise ValueError(f"Could not parse observation_space in model: {model_path}")

        data = pickle.loads(raw_data)
        space = data["observation_space"]
        return space.shape, space.dtype


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    model_path, stats_path = find_model()
    if model_path is None:
        print("No model found in rl_models/")
        return

    saved_obs_shape, saved_obs_dtype = saved_obs_space(model_path)

    render_mode = None if args.no_render else "human"
    eval_env = DummyVecEnv(
        [lambda: TimeLimit(AlexStandupEnv(render_mode=render_mode), max_episode_steps=MAX_EPISODE_STEPS)]
    )

    if eval_env.observation_space.shape != saved_obs_shape:
        print(
            f"Saved model expects obs shape {saved_obs_shape}, but env gives {eval_env.observation_space.shape}."
        )

    if stats_path:
        try:
            eval_env = VecNormalize.load(str(stats_path), eval_env)
            eval_env.training = False
            eval_env.norm_reward = False
        except AssertionError as exc:
            print(f"Skipping VecNormalize stats due to shape mismatch: {exc}")

    if saved_obs_dtype == np.float32:
        eval_env = CastObsToFloat32(eval_env)

    model = PPO.load(str(model_path), env=eval_env)
    print(f"Loaded model from {model_path}")

    rewards = []
    for episode in range(args.episodes):
        obs = eval_env.reset()
        episode_reward = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = eval_env.step(action)
            episode_reward += reward[0]
            if done[0]:
                break
        rewards.append(episode_reward)
        print(f"Episode {episode + 1:2d} | reward: {episode_reward:.1f}")

    print(
        f"\nMean over {args.episodes} episodes: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}"
    )
    eval_env.close()


if __name__ == "__main__":
    main()
