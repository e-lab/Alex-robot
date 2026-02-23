import os
import numpy as np
from pathlib import Path

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parents[1]
SCENE_XML  = REPO_ROOT / "scenes/alex-scenes/scene_alex_v1_rl.xml"
MODELS_DIR = SCRIPT_DIR / "rl_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.0)),
    "elevation": -20.0,
}

# ─── Stand-prep reference pose ───────────────────────────────────────────────
# A slightly crouched stable stance for Alex
STAND_PREP_TARGET: dict[str, float] = {
    "spine_z": 0.0,
    "left_hip_x": 0.1,    "right_hip_x": -0.1,
    "left_hip_y": -0.45,  "right_hip_y": -0.45,
    "left_hip_z": 0.0,    "right_hip_z": 0.0,
    "left_knee_y": 0.7,   "right_knee_y": 0.7,
    "left_ankle_y": -0.28,"right_ankle_y": -0.28,
    "left_ankle_x": 0.0,  "right_ankle_x": 0.0,
    "left_shoulder_y": 0.2,"right_shoulder_y": 0.2,
    "left_shoulder_x": 0.3,"right_shoulder_x": -0.3,
    "left_elbow_y": -0.5, "right_elbow_y": -0.5,
}


def mass_center(model, data):
    """Calculate center of mass as weighted average: (model.body_mass.T * data.xipos) / sum(model.body_mass)."""
    num = np.einsum("b,bj->j", model.body_mass, data.xipos)
    denom = model.body_mass.sum()
    return (num / denom)[0:2].copy()


class AlexHumanoidEnv(MujocoEnv, utils.EzPickle):
    """
    Alex robot walking environment modelled after Gymnasium Humanoid-v5.

    The 3-D bipedal Alex robot is trained to walk forward as fast as possible.
    Reward structure, observation layout, and episode-end conditions follow the
    Humanoid-v5 implementation by Kallinteris-Andreas.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def __init__(
        self,
        xml_file: str = str(SCENE_XML),
        frame_skip: int = 5,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.25,
        ctrl_cost_weight: float = 0.1,
        contact_cost_weight: float = 5e-7,
        contact_cost_range: tuple[float, float] = (-np.inf, 10.0),
        healthy_reward: float = 5.0,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: tuple[float, float] = (0.7, 1.5),  # Alex pelvis ~1.0 m
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
            forward_reward_weight,
            ctrl_cost_weight,
            contact_cost_weight,
            contact_cost_range,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            include_cinert_in_observation,
            include_cvel_in_observation,
            include_qfrc_actuator_in_observation,
            include_cfrc_ext_in_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._contact_cost_range = contact_cost_range
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self._include_cinert_in_observation = include_cinert_in_observation
        self._include_cvel_in_observation = include_cvel_in_observation
        self._include_qfrc_actuator_in_observation = include_qfrc_actuator_in_observation
        self._include_cfrc_ext_in_observation = include_cfrc_ext_in_observation

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = self.data.qpos.size + self.data.qvel.size
        obs_size -= 2 * exclude_current_positions_from_observation
        obs_size += self.data.cinert[1:].size * include_cinert_in_observation
        obs_size += self.data.cvel[1:].size * include_cvel_in_observation
        obs_size += (self.data.qvel.size - 6) * include_qfrc_actuator_in_observation
        obs_size += self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 2 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 2 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
            "cinert": self.data.cinert[1:].size * include_cinert_in_observation,
            "cvel": self.data.cvel[1:].size * include_cvel_in_observation,
            "qfrc_actuator": (self.data.qvel.size - 6)
            * include_qfrc_actuator_in_observation,
            "cfrc_ext": self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation,
            "ten_length": 0,
            "ten_velocity": 0,
        }

    @property
    def healthy_reward(self):
        return self.is_healthy * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(self.data.ctrl))
        return control_cost

    @property
    def contact_cost(self):
        contact_forces = self.data.cfrc_ext
        contact_cost = self._contact_cost_weight * np.sum(np.square(contact_forces))
        min_cost, max_cost = self._contact_cost_range
        contact_cost = np.clip(contact_cost, min_cost, max_cost)
        return contact_cost

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.data.qpos[2] < max_z
        return is_healthy

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()

        if self._include_cinert_in_observation is True:
            com_inertia = self.data.cinert[1:].flatten()
        else:
            com_inertia = np.array([])
        if self._include_cvel_in_observation is True:
            com_velocity = self.data.cvel[1:].flatten()
        else:
            com_velocity = np.array([])

        if self._include_qfrc_actuator_in_observation is True:
            actuator_forces = self.data.qfrc_actuator[6:].flatten()
        else:
            actuator_forces = np.array([])
        if self._include_cfrc_ext_in_observation is True:
            external_contact_forces = self.data.cfrc_ext[1:].flatten()
        else:
            external_contact_forces = np.array([])

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        return np.concatenate(
            (
                position,
                velocity,
                com_inertia,
                com_velocity,
                actuator_forces,
                external_contact_forces,
            )
        )

    # Actuator indices that are NOT needed for walking — hold at zero.
    # Order in scene_alex_v1_train.xml:
    #  17: left_wrist_z  18: left_wrist_x  19: neck_z  20: neck_y
    #  25: right_wrist_z  26: right_wrist_x
    _PASSIVE_ACTUATORS = np.array([17, 18, 19, 20, 25, 26], dtype=int)

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).copy()
        action[self._PASSIVE_ACTUATORS] = 0.0  # wrists & neck: hold at zero
        xy_position_before = mass_center(self.model, self.data)
        self.do_simulation(action, self.frame_skip)
        xy_position_after = mass_center(self.model, self.data)

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "tendon_length": self.data.ten_length,
            "tendon_velocity": self.data.ten_velocity,
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def _get_rew(self, x_velocity: float, action):
        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward
        rewards = forward_reward + healthy_reward

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        costs = ctrl_cost + contact_cost

        reward = rewards - costs

        reward_info = {
            "reward_survive": healthy_reward,
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
        }

        return reward, reward_info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # Apply Alex-specific stable crouched stance
        for name, val in STAND_PREP_TARGET.items():
            try:
                jid = self.model.joint(name).id
                qpos[self.model.jnt_qposadr[jid]] = val
            except Exception:
                pass

        qpos[2] = 0.90  # Start pelvis lower for stability
        qpos += self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel += self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "tendon_length": self.data.ten_length,
            "tendon_velocity": self.data.ten_velocity,
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }


TOTAL_TIMESTEPS = 30_000_000


def train(resume: bool = False):
    n_envs = 12
    vec_env = make_vec_env(AlexHumanoidEnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

    stats_path = MODELS_DIR / "vec_normalize_final.pkl"
    final_path = MODELS_DIR / "alex_humanoid_walk_final.zip"
    best_path  = MODELS_DIR / "best" / "best_model.zip"

    if resume and stats_path.exists() and (final_path.exists() or best_path.exists()):
        vec_env = VecNormalize.load(str(stats_path), vec_env)
        vec_env.training = True
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
            tensorboard_log=str(MODELS_DIR / "tensorboard"), batch_size=1024, n_steps=2048,
        )
        remaining = TOTAL_TIMESTEPS
        print("Starting fresh training.")

    eval_env = make_vec_env(AlexHumanoidEnv, n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    eval_callback = EvalCallback(eval_env, best_model_save_path=str(MODELS_DIR / "best"),
                                 eval_freq=10000)

    try:
        model.learn(total_timesteps=remaining, callback=eval_callback,
                    progress_bar=True, reset_num_timesteps=not resume)
    except KeyboardInterrupt:
        print("\nTraining interrupted.")

    model.save(str(MODELS_DIR / "alex_humanoid_walk_final"))
    vec_env.save(str(MODELS_DIR / "vec_normalize_final.pkl"))
    vec_env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()
    train(resume=args.resume)
