import gymnasium as gym
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
MODELS_DIR = SCRIPT_DIR / "rl_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def resolve_training_xml() -> Path:
    candidates = [
        REPO_ROOT / "alex_models" / "alex_V1_description" / "mjcf" / "alex_v1_humanoid_train.xml",
        REPO_ROOT / "alex-models" / "alex_V1_description" / "mjcf" / "alex_v1_humanoid_train.xml",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]

# ─── Hyperparameters ──────────────────────────────────────────────────────────
ENV_NAME = "Humanoid-v5"
XML_FILE = str(resolve_training_xml())
ENV_KWARGS = {
    "xml_file": XML_FILE,
    "healthy_z_range": (0.80, 2.20),
}
TOTAL_TIMESTEPS = 10_000_000
N_ENVS = 12
LEARNING_RATE = 3e-4
BATCH_SIZE = 1024
N_STEPS = 2048
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.005 # Small entropy coefficient for exploration

def train():
    # Create vectorized environment
    vec_env = make_vec_env(
        ENV_NAME,
        n_envs=N_ENVS,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=ENV_KWARGS,
    )
    # Normalize observations and rewards
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # Initialize PPO agent
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        n_steps=N_STEPS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        tensorboard_log=str(MODELS_DIR / "tensorboard"),
        device="auto"
    )

    # Callbacks
    eval_env = make_vec_env(ENV_NAME, n_envs=1, env_kwargs=ENV_KWARGS)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10., training=False)
    
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=str(MODELS_DIR / "best"),
        log_path=str(MODELS_DIR / "eval_logs"), 
        eval_freq=max(50_000 // N_ENVS, 1),
        deterministic=True, 
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // N_ENVS, 1), 
        save_path=str(MODELS_DIR / "checkpoints"),
        name_prefix="ppo_humanoid",
        save_vecnormalize=True
    )

    print(f"Training env: {ENV_NAME} with xml_file={XML_FILE}")
    print(f"Starting PPO training for {TOTAL_TIMESTEPS} steps across {N_ENVS} envs...")
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True,
            tb_log_name="PPO"
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    # Save final model and stats
    model.save(str(MODELS_DIR / "ppo_humanoid_final"))
    vec_env.save(str(MODELS_DIR / "vec_normalize_final.pkl"))
    print(f"Training finished! Models saved to {MODELS_DIR}")
    
    vec_env.close()
    eval_env.close()

if __name__ == "__main__":
    train()
