__credits__ = ["Kallinteris-Andreas"]

import numpy as np
from pathlib import Path

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import TimeLimit

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parents[1]
SCENE_XML  = REPO_ROOT / "scenes/alex-scenes/scene_alex_v1_rl.xml"
MODELS_DIR = SCRIPT_DIR / "rl_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 0.5)),
    "elevation": -20.0,
}

# Alex lying on its back: pelvis near the floor, rotated -90° around global Y.
# Quaternion convention: [w, x, y, z].  cos(-π/4)=0.7071, sin(-π/4)=-0.7071.
_LYING_Z    = 0.30
_LYING_QUAT = np.array([0.7071068, 0.0, -0.7071068, 0.0])  # w, x, y, z


class AlexStandupEnv(MujocoEnv, utils.EzPickle):
    """
    Alex robot standup environment modelled after Gymnasium HumanoidStandup-v5.

    The robot starts lying on the ground and must learn to stand upright.
    Reward structure follows HumanoidStandup-v5 (Kallinteris-Andreas):
      reward = uph_cost - impact_cost + 1
    where uph_cost = pelvis_z / timestep.
    There is no termination condition; episodes end only via the TimeLimit wrapper.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
    }

    def __init__(
        self,
        xml_file: str = str(SCENE_XML),
        frame_skip: int = 5,
        default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
        uph_cost_weight: float = 1.0,
        impact_cost_weight: float = 0.5e-6,
        impact_cost_range: tuple[float, float] = (-np.inf, 10.0),
        reset_noise_scale: float = 1e-2,
        exclude_current_positions_from_observation: bool = True,
        include_cinert_in_observation: bool = True,
        include_cvel_in_observation: bool = True,
        include_qfrc_actuator_in_observation: bool = True,
        include_cfrc_ext_in_observation: bool = True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            uph_cost_weight,
            impact_cost_weight,
            impact_cost_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            include_cinert_in_observation,
            include_cvel_in_observation,
            include_qfrc_actuator_in_observation,
            include_cfrc_ext_in_observation,
            **kwargs,
        )

        self._uph_cost_weight    = uph_cost_weight
        self._impact_cost_weight = impact_cost_weight
        self._impact_cost_range  = impact_cost_range
        self._reset_noise_scale  = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self._include_cinert_in_observation       = include_cinert_in_observation
        self._include_cvel_in_observation         = include_cvel_in_observation
        self._include_qfrc_actuator_in_observation = include_qfrc_actuator_in_observation
        self._include_cfrc_ext_in_observation     = include_cfrc_ext_in_observation

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = self.data.qpos.size + self.data.qvel.size
        obs_size -= 2 * exclude_current_positions_from_observation
        obs_size += self.data.cinert[1:].size  * include_cinert_in_observation
        obs_size += self.data.cvel[1:].size    * include_cvel_in_observation
        obs_size += (self.data.qvel.size - 6)  * include_qfrc_actuator_in_observation
        obs_size += self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 2 * exclude_current_positions_from_observation,
            "qpos":  self.data.qpos.size - 2 * exclude_current_positions_from_observation,
            "qvel":  self.data.qvel.size,
            "cinert": self.data.cinert[1:].size    * include_cinert_in_observation,
            "cvel":   self.data.cvel[1:].size      * include_cvel_in_observation,
            "qfrc_actuator": (self.data.qvel.size - 6) * include_qfrc_actuator_in_observation,
            "cfrc_ext": self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation,
        }

    # ── Locked joints (same as walk env: arms, wrists, neck) ──────────────────
    # idx: name              target  reason
    #  13: left_shoulder_y    0.2    arms slightly forward
    #  14: left_shoulder_x    0.3    arms slightly out
    #  15: left_shoulder_z    0.0    neutral
    #  16: left_elbow_y      -0.5    bent arm
    #  17: left_wrist_z       0.0    neutral
    #  18: left_wrist_x       0.0    neutral
    #  19: neck_z             0.0    head forward
    #  20: neck_y             0.0    head level
    #  21: right_shoulder_y   0.2    arms slightly forward
    #  22: right_shoulder_x  -0.3    arms slightly out (mirrored)
    #  23: right_shoulder_z   0.0    neutral
    #  24: right_elbow_y     -0.5    bent arm
    #  25: right_wrist_z      0.0    neutral
    #  26: right_wrist_x      0.0    neutral
    _LOCKED_ACTUATORS: dict[int, float] = {
        13:  0.2, 14:  0.3, 15: 0.0,   # left shoulder
        16: -0.5, 17:  0.0, 18: 0.0,   # left elbow + wrists
        19:  0.0, 20:  0.0,            # neck
        21:  0.2, 22: -0.3, 23: 0.0,   # right shoulder
        24: -0.5, 25:  0.0, 26: 0.0,   # right elbow + wrists
    }
    _LOCKED_KP = 0.5

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def impact_cost(self):
        impact_cost = self._impact_cost_weight * np.sum(np.square(self.data.cfrc_ext))
        min_cost, max_cost = self._impact_cost_range
        return float(np.clip(impact_cost, min_cost, max_cost))

    # ── Observation ───────────────────────────────────────────────────────────

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()

        com_inertia = (
            self.data.cinert[1:].flatten()
            if self._include_cinert_in_observation else np.array([])
        )
        com_velocity = (
            self.data.cvel[1:].flatten()
            if self._include_cvel_in_observation else np.array([])
        )
        actuator_forces = (
            self.data.qfrc_actuator[6:].flatten()
            if self._include_qfrc_actuator_in_observation else np.array([])
        )
        external_contact_forces = (
            self.data.cfrc_ext[1:].flatten()
            if self._include_cfrc_ext_in_observation else np.array([])
        )

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        return np.concatenate((
            position,
            velocity,
            com_inertia,
            com_velocity,
            actuator_forces,
            external_contact_forces,
        ))

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).copy()

        # Soft PD hold for locked joints
        for idx, target in self._LOCKED_ACTUATORS.items():
            jid  = self.model.actuator_trnid[idx, 0]
            qadr = self.model.jnt_qposadr[jid]
            error = target - self.data.qpos[qadr]
            action[idx] = np.clip(self._LOCKED_KP * error, -1.0, 1.0)

        self.do_simulation(action, self.frame_skip)
        pos_after = float(self.data.qpos[2])   # pelvis z after step

        observation = self._get_obs()
        reward, reward_info = self._get_rew(pos_after)

        # No termination — only the TimeLimit wrapper truncates
        terminated = False

        info = {
            "x_position": float(self.data.qpos[0]),
            "y_position": float(self.data.qpos[1]),
            "z_position": pos_after,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info

    def _get_rew(self, pos_after: float):
        uph_cost    = self._uph_cost_weight * pos_after / self.model.opt.timestep
        impact_cost = self.impact_cost
        reward      = uph_cost - impact_cost + 1.0   # +1 survive bonus

        reward_info = {
            "reward_uph":     uph_cost,
            "reward_impact":  -impact_cost,
            "reward_survive": 1.0,
        }
        return reward, reward_info

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset_model(self):
        noise_low  = -self._reset_noise_scale
        noise_high =  self._reset_noise_scale

        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # Place Alex lying on its back
        qpos[2]   = _LYING_Z      # pelvis z near floor
        qpos[3:7] = _LYING_QUAT   # -90° around global Y → lying on back
        qpos[7:]  = 0.0           # all joints neutral

        # Small noise so each episode starts slightly differently
        qpos += self.np_random.uniform(low=noise_low, high=noise_high,
                                       size=self.model.nq)
        qvel += self.np_random.uniform(low=noise_low, high=noise_high,
                                       size=self.model.nv)

        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_reset_info(self):
        return {
            "x_position": float(self.data.qpos[0]),
            "y_position": float(self.data.qpos[1]),
            "z_position": float(self.data.qpos[2]),
        }


# ─── Training ─────────────────────────────────────────────────────────────────

TOTAL_TIMESTEPS   = 30_000_000
MAX_EPISODE_STEPS = 1000


def make_env():
    env = AlexStandupEnv()
    env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
    return env


def train(resume: bool = False):
    n_envs  = 12
    vec_env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

    stats_path = MODELS_DIR / "vec_normalize_final.pkl"
    final_path = MODELS_DIR / "alex_standup_final.zip"
    best_path  = MODELS_DIR / "best" / "best_model.zip"

    if resume and stats_path.exists() and (final_path.exists() or best_path.exists()):
        vec_env   = VecNormalize.load(str(stats_path), vec_env)
        vec_env.training    = True
        vec_env.norm_reward = True

        checkpoint = final_path if final_path.exists() else best_path
        model = PPO.load(str(checkpoint), env=vec_env, device="cpu",
                         tensorboard_log=str(MODELS_DIR / "tensorboard"))
        remaining = max(0, TOTAL_TIMESTEPS - model.num_timesteps)
        print(f"Resuming from {checkpoint.name} "
              f"({model.num_timesteps:,} steps done, {remaining:,} remaining).")
    else:
        if resume:
            print("No checkpoint found — starting fresh.")
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        model = PPO(
            "MlpPolicy", vec_env, verbose=1, device="cpu", ent_coef=0.01,
            tensorboard_log=str(MODELS_DIR / "tensorboard"),
            batch_size=1024, n_steps=2048,
        )
        remaining = TOTAL_TIMESTEPS
        print("Starting fresh training.")

    eval_env = make_vec_env(make_env, n_envs=1, vec_env_cls=DummyVecEnv)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(MODELS_DIR / "best"),
        eval_freq=10_000,
    )

    try:
        model.learn(
            total_timesteps=remaining,
            callback=eval_callback,
            progress_bar=True,
            reset_num_timesteps=not resume,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted.")

    model.save(str(MODELS_DIR / "alex_standup_final"))
    vec_env.save(str(MODELS_DIR / "vec_normalize_final.pkl"))
    vec_env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    args = parser.parse_args()
    train(resume=args.resume)
