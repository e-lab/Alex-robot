import argparse
import pickle
import json
import zipfile
import base64
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


ENV_NAME  = "Humanoid-v5"
SCRIPT_DIR = Path(__file__).resolve().parent
XML_FILE = str(
    (SCRIPT_DIR / "../../alex-models" / "alex_V1_description" / "mjcf" / "alex_v1_humanoid_train.xml").resolve()
)
ENV_KWARGS = {
    "xml_file": XML_FILE,
    "healthy_z_range": (0.80, 2.20),
}
REDUCED_OBS_KWARGS = {
    "include_cinert_in_observation": False,
    "include_cvel_in_observation": False,
    "include_cfrc_ext_in_observation": False,
}


def find_model(warp: bool) -> tuple[Path | None, Path | None]:
    """Return (model_path, stats_path) for the requested mode, trying best then final."""
    if warp:
        base = SCRIPT_DIR / "rl_models_warp"
        candidates = [base / "best" / "best_model.zip", base / "ppo_humanoid_warp_final.zip"]
        stats = None  # warp training does not save VecNormalize
    else:
        base = SCRIPT_DIR / "rl_models"
        candidates = [base / "best" / "best_model.zip", base / "ppo_humanoid_final.zip"]
        stats = base / "vec_normalize_final.pkl"

    model_path = next((p for p in candidates if p.exists()), None)
    stats_path = stats if (stats and stats.exists()) else None
    return model_path, stats_path


def saved_obs_space(model_path: Path) -> tuple[tuple[int, ...], np.dtype]:
    """Read observation space shape/dtype from a saved SB3 model without loading weights."""
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
            raise ValueError(f"Could not find observation_space.dtype in JSON model: {model_path}")
        
        data = pickle.loads(raw_data)
        space = data["observation_space"]
        return space.shape, space.dtype


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warp", action="store_true",
                        help="Evaluate model from rl_models_warp/ (mujoco-warp trained)")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    model_path, stats_path = find_model(args.warp)
    if model_path is None:
        print(f"No model found in {'rl_models_warp' if args.warp else 'rl_models'}/")
        return

    saved_obs_shape, saved_obs_dtype = saved_obs_space(model_path)
    env_kwargs = dict(ENV_KWARGS)
    if saved_obs_shape == (92,):
        env_kwargs.update(REDUCED_OBS_KWARGS)
        print("Using reduced observation config to match saved model shape (92,).")

    render_mode = None if args.no_render else "human"
    eval_env = DummyVecEnv(
        [lambda: gym.make(ENV_NAME, render_mode=render_mode, **env_kwargs)]
    )

    if stats_path:
        try:
            eval_env = VecNormalize.load(str(stats_path), eval_env)
            eval_env.training = False
            eval_env.norm_reward = False
        except AssertionError as exc:
            print(f"Skipping VecNormalize stats due to shape mismatch: {exc}")

    # Auto-detect obs dtype from saved model and cast if needed (float32 = warp-trained).
    if saved_obs_dtype == np.float32:
        eval_env = CastObsToFloat32(eval_env)

    model = PPO.load(str(model_path), env=eval_env)
    label = "warp" if args.warp else "standard"
    print(f"Loaded {label} model from {model_path}")

    rewards = []
    for episode in range(args.episodes):
        obs = eval_env.reset()
        episode_reward = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            episode_reward += reward[0]
            if done[0]:
                break
        rewards.append(episode_reward)
        print(f"Episode {episode + 1:2d} | reward: {episode_reward:.1f}")

    print(f"\nMean over {args.episodes} episodes: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    eval_env.close()


if __name__ == "__main__":
    main()
