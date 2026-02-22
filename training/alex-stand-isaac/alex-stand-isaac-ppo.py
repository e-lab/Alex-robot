#!/usr/bin/env python3
"""
PPO training for Alex standing — IsaacLab config port (MuJoCo backend).

Faithfully mirrors the isaacsimlab/ configuration inside MuJoCo:

  Initial pose  : bent-knee stance from isaacsimlab/alex_stand_env_cfg.py
                  (hip_y=-0.772, knee=1.419, ankle_y=-0.634)
  Timing        : CONTROL_DT=0.02 s, SIM_DT=0.005 s → decimation=4
                  (4 physics steps per policy step, same as IsaacLab)
  Actuator delay: action delayed 0-2 sim steps (4-8 ms), matching
                  DelayedPDActuatorCfg(min_delay=0, max_delay=2)
  Observations  : gravity projection (not quaternion), joint offsets,
                  joint vels, prev action, target height command
                  (mirrors AlexObservationsCfg policy group).
                  Uniform noise added during training:
                  ang_vel ±0.3, gravity ±0.05, joint_pos ±0.01,
                  joint_vel ±0.3 (mirrors enable_corruption=True).
  Rewards       : direct port of StandingEnvCfg / AlexRewards terms:
                  height tracking (exp, std=0.05), flat orientation (exp,
                  std=0.2), joint deviation by group (arms/torso/hips/
                  ankle_x), foot sliding, ankle torques, undesired thigh
                  contacts, action rate, joint velocity, lin/ang velocity
                  penalties.
  Randomisation : push disturbances every 2-4 s (±0.7 m/s x/y,
                  ±0.05 m/s z, ±0.1 rad/s roll/pitch, ±0.2 rad/s yaw),
                  mass randomisation ±2 kg (torso), joint reset ±0.7 rad
                  + ±0.5 rad/s velocity, base reset ±0.5 m xy / ±0.01 m z
                  / ±0.1 rad roll-pitch / full yaw.
                  NOTE: with 8 CPU envs (vs 4096 GPU), reduce rand ranges
                  if policy cannot learn early episodes.
  PPO           : mirrors AlexStandingEnvPPORunnerCfg (RSL-RL):
                  [128,128,128] ELU, lr=1e-3, 5 epochs, 4 mini-batches,
                  gamma=0.99, gae_lambda=0.95, entropy=0.01, clip=0.2

Dependencies:
    pip install stable-baselines3[extra] gymnasium mujoco torch

Training:
    python training/alex-stand-isaac/alex-stand-isaac-ppo.py

Evaluate:
    mjpython training/alex-stand-isaac/alex-stand-isaac-ppo.py \\
        --eval rl_models/best/best_model --vec-norm rl_models/vec_normalize_final.pkl
"""
from __future__ import annotations

import math
import re
import time
from collections import deque
from pathlib import Path
from typing import Optional

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = Path(__file__).resolve().parents[2]
SCENE_XML  = REPO_ROOT / "scenes/alex-scenes/scene_alex_v1_train.xml"
MODELS_DIR = SCRIPT_DIR / "rl_models"

# ── Timing — from isaacsimlab/alex.py ─────────────────────────────────────────

CONTROL_DT  = 0.02     # policy rate: 50 Hz
SIM_DT      = 0.005    # physics rate: 200 Hz
DECIMATION  = int(CONTROL_DT / SIM_DT)   # = 4 sim steps per control step

# Actuator communication delay (sim steps) — from DelayedPDActuatorCfg
MIN_DELAY_DT    = 0.004
MAX_DELAY_DT    = 0.008
MIN_DELAY_STEPS = math.floor(MIN_DELAY_DT / SIM_DT)   # = 0
MAX_DELAY_STEPS = math.ceil(MAX_DELAY_DT  / SIM_DT)   # = 2

# ── Action scale — ALEX_JOINT_SCALE from isaacsimlab/alex.py ──────────────────

ACTION_SCALE = 0.30   # rad — policy output in [-1,1] maps to ±0.3 rad

# ── Reference poses ────────────────────────────────────────────────────────────
# Select with --pose {isaaclab|original|zero} at the command line.

# "isaaclab": bent-knee stance from isaacsimlab/alex_stand_env_cfg.py.
# Matches GPU-parallel training config but is harder to learn from scratch.
POSE_ISAACLAB: dict[str, float] = {
    "left_hip_x":    0.00,   "right_hip_x":   0.00,
    "left_hip_y":   -0.772,  "right_hip_y":  -0.772,   # ≈ −44°
    "left_hip_z":    0.00,   "right_hip_z":   0.00,
    "left_knee":     1.419,  "right_knee":    1.419,    # ≈ +81°
    "left_ankle_y": -0.634,  "right_ankle_y": -0.634,   # ≈ −36°
    "left_ankle_x":  0.00,   "right_ankle_x":  0.00,
    "spine_z":  0.00,  "neck_z":  0.00,  "neck_y":  0.00,
    "left_shoulder_y":   0.15,  "right_shoulder_y":  0.15,
    "left_shoulder_z":   0.05,  "right_shoulder_z": -0.05,
    "left_shoulder_x":   0.05,  "right_shoulder_x": -0.05,
    "left_elbow":       -0.50,  "right_elbow":      -0.50,
    "left_wrist_x":  0.00,  "right_wrist_x":  0.00,
    "left_wrist_z":  0.00,  "right_wrist_z":  0.00,
}

# "original": near-upright stance from training/alex-stand/alex-stand-ppo.py.
# Easier for single-machine training; policy converges faster.
POSE_ORIGINAL: dict[str, float] = {
    "spine_z":      0.00,
    "neck_y":       0.00,   "neck_z":       0.00,
    "left_hip_z":   0.00,   "right_hip_z":  0.00,
    "left_hip_x":   0.05,   "right_hip_x": -0.05,
    "left_hip_y":   0.05,   "right_hip_y":  0.05,
    "left_knee":    0.10,   "right_knee":   0.10,
    "left_ankle_y":-0.03,   "right_ankle_y":-0.03,
    "left_ankle_x": 0.00,   "right_ankle_x": 0.00,
    "left_shoulder_x":  0.20,  "right_shoulder_x": -0.20,
    "left_shoulder_z": -0.20,  "right_shoulder_z":  0.20,
    "left_shoulder_y":  0.30,  "right_shoulder_y":  0.30,
    "left_elbow":  -0.80,   "right_elbow":  -0.80,
    "left_wrist_x": 0.00,   "right_wrist_x": 0.00,
    "left_wrist_z": 0.00,   "right_wrist_z": 0.00,
}

# "zero": all joints at 0 — useful for quick sanity tests.
POSE_ZERO: dict[str, float] = {}

POSES = {"isaaclab": POSE_ISAACLAB, "original": POSE_ORIGINAL, "zero": POSE_ZERO}

# Default pose used when constructing the env without a --pose flag.
STAND_PREP_TARGET = POSE_ISAACLAB

# ── Task parameters ────────────────────────────────────────────────────────────

EPISODE_LENGTH_S  = 20.0
MAX_STEPS         = int(EPISODE_LENGTH_S / CONTROL_DT)   # = 1000 control steps
PELVIS_Z_MIN      = 0.35    # catastrophic fall floor (m)
UPRIGHT_MIN_COS   = 0.30    # terminate if cos(tilt) < 0.30 (≈73°), same as original

# Height command range — from isaacsimlab/alex_stand_env_cfg.py
TARGET_HEIGHT_DEFAULT = 0.93
TARGET_HEIGHT_MIN     = 0.65
TARGET_HEIGHT_MAX     = 0.95

# ── Reward weights — ported from StandingEnvCfg / walk_flat_env_cfg.py ────────

W_HEIGHT          =  3.0    # track_lin_pos_z_exp              (w=3.0)
W_ORIENTATION     =  2.0    # track_flat_orientation / flat_orientation_exp (w=2.0)
W_ALIVE           =  0.5    # per-step survival bonus (not in IsaacLab but useful here)
W_ACTION_RATE     = -0.10   # action_rate_l2                    (w=-0.1)
W_LIN_VEL_Z      = -0.10   # lin_vel_z_l2                      (w=-0.1)
W_ANG_VEL_XY     = -0.20   # ang_vel_xy_l2                     (w=-0.2)
W_JOINT_VEL      = -5e-6   # dof_acc_l2 proxy                  (w=-5e-6 × scale²)
W_JOINT_TORQUE   = -1e-5   # dof_torques_l2                    (w=-1e-5)
W_JOINT_ARMS     = -0.60   # joint_deviation_arms (L1)          (w=-0.6)
W_JOINT_TORSO    = -0.60   # joint_deviation_torso (L1)         (w=-0.6)
W_JOINT_HIP_YAW  = -1.50   # joint_mean_deviation_hip           (w=-1.5)
W_ANKLE_TORQUE   = -2e-5   # ankle_roll_torques_l2              (w=-2e-5)
W_JOINT_ANKLE_X  = -0.60   # joint_mean_deviation_ankle L1      (w=-0.6)
W_FOOT_SLIDE     = -0.50   # foot_sliding (penalise sliding vel when contacting floor)
W_UNDESIRED_CONT = -0.10   # undesired_contacts (thigh/shin)    (w=-0.1)
# IsaacLab's death=-200 is calibrated for GPU-normalized rewards.
# With SB3 VecNormalize and ~2-5 per-step rewards, use -10 (same as original).
FALL_PENALTY      = -10.0

# Exponential reward shape parameters (matching IsaacLab std params)
HEIGHT_STD        = 0.05   # track_lin_pos_z_exp std
ORIENTATION_STD   = 0.20   # flat_orientation_exp std

# ── Domain randomisation — from isaacsimlab/alex_stand_env_cfg.py ─────────────

# All ranges match isaacsimlab/alex_stand_env_cfg.py exactly.
# NOTE: with only 8 CPU envs (vs 4096 GPU), early training may struggle with
# large reset ranges. Reduce JOINT_RESET_RANGE / BASE_RESET_XY if the policy
# cannot recover from resets. Full ranges are kept here for faithful porting.
JOINT_RESET_RANGE     = 0.70      # ±0.70 rad  — reset_robot_joints position_range
JOINT_RESET_VEL       = 0.50      # ±0.50 rad/s — reset_robot_joints velocity_range
BASE_RESET_XY         = 0.50      # ±0.50 m    — reset_base pose_range x/y
BASE_RESET_Z          = 0.01      # ±0.01 m    — reset_base pose_range z
BASE_RESET_YAW        = math.pi   # ±π rad     — reset_base pose_range yaw
BASE_RESET_ROLL_PITCH = 0.10      # ±0.10 rad  — reset_base pose_range roll/pitch
MASS_RAND_KG          = 2.0       # ±2 kg on torso body — add_base_mass
PUSH_INTERVAL_MIN     = 2.0       # s between pushes — push_robot interval_range_s
PUSH_INTERVAL_MAX     = 4.0       # s between pushes
PUSH_VEL_XY           = 0.7       # m/s x/y — push_robot velocity_range x/y
PUSH_VEL_Z            = 0.05      # m/s z   — push_robot velocity_range z
PUSH_VEL_ROLL_PITCH   = 0.10      # rad/s   — push_robot velocity_range roll/pitch
PUSH_VEL_YAW          = 0.2       # rad/s   — push_robot velocity_range yaw

# ── Observation noise — from isaacsimlab/walk_flat_env_cfg.py AlexObservationsCfg ──
# Uniform noise applied during training (enable_corruption=True equivalent).
OBS_NOISE_ANG_VEL   = 0.30   # ±0.3 rad/s on base angular velocity
OBS_NOISE_GRAV      = 0.05   # ±0.05 on projected gravity vector
OBS_NOISE_JOINT_POS = 0.01   # ±0.01 rad on joint positions
OBS_NOISE_JOINT_VEL = 0.30   # ±0.3 rad/s on joint velocities


# ── Helpers ────────────────────────────────────────────────────────────────────

def _quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll  / 2), math.sin(roll  / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw   / 2), math.sin(yaw   / 2)
    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ])


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _find_floor_height(model: mujoco.MjModel, data: mujoco.MjData,
                       joint_pose: dict[str, float],
                       name2id: dict[str, int],
                       qadr: np.ndarray) -> float:
    """Auto-calibrate pelvis z so the lowest foot geom sits exactly on the floor."""
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 1.0
    data.qpos[3] = 1.0
    data.qpos[4:7] = 0.0
    data.qpos[7:] = 0.0
    for nm, q in joint_pose.items():
        if nm in name2id:
            data.qpos[qadr[name2id[nm]]] = q
    mujoco.mj_forward(model, data)

    lowest = np.inf
    for i in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if "foot" in gname.lower():
            gz   = float(data.geom_xpos[i, 2])
            rot  = data.geom_xmat[i].reshape(3, 3)
            sz   = model.geom_size[i]
            half_world_z = (abs(rot[2,0]) * sz[0] + abs(rot[2,1]) * sz[1]
                            + abs(rot[2,2]) * sz[2])
            lowest = min(lowest, gz - half_world_z)

    return 1.0 - lowest + 0.002


def _bodies_matching(model: mujoco.MjModel, pattern: str) -> list[int]:
    """Return body IDs whose names match the given regex pattern."""
    ids = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        if re.search(pattern, name, re.IGNORECASE):
            ids.append(i)
    return ids


def _geoms_of_bodies(model: mujoco.MjModel, body_ids: list[int]) -> set[int]:
    """Return the set of geom IDs belonging to any of the given body IDs."""
    geom_set: set[int] = set()
    for bid in body_ids:
        for gi in range(model.geom_bodyid.shape[0]):
            if model.geom_bodyid[gi] == bid:
                geom_set.add(gi)
    return geom_set


def _geom_in_contact(data: mujoco.MjData, geom_ids: set[int]) -> bool:
    """Return True if any of the given geoms currently has a contact."""
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 in geom_ids or c.geom2 in geom_ids:
            return True
    return False


def _geom_contact_normal_force(data: mujoco.MjData, geom_ids: set[int]) -> float:
    """Sum of normal contact forces for the given geoms (N)."""
    total = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 in geom_ids or c.geom2 in geom_ids:
            # contact.efc_address -> cfrc, but simpler: just count contact
            total += 1.0
    return total


# ── Environment ───────────────────────────────────────────────────────────────

class AlexStandIsaacEnv(gym.Env):
    """
    Alex stand-and-balance task mirroring the isaacsimlab/ configuration.

    Observation (float32):
        gravity_proj (3)      gravity vector in pelvis frame (≈[0,0,-1] upright)
        pelvis_z (1)          absolute pelvis height
        pelvis_angvel (3)     pelvis angular velocity in body frame
        pelvis_linvel (3)     pelvis linear velocity in world frame
        height_cmd (1)        target height command (sampled from [0.65, 0.95])
        joint_pos_offset (N)  joint positions minus stand-pose reference
        joint_vel (N)         joint velocities
        prev_action (N)       previous normalised action

    Action (float32, N dims, clipped to [-1, 1]):
        Desired joint-position offsets from stand pose × ACTION_SCALE rad.
        Delayed by 0-2 sim steps to simulate actuator communication latency.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: Optional[str] = None,
                 random_init: bool = True,
                 pose: str = "isaaclab",
                 enable_push: bool = False) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.random_init = random_init
        self._enable_push = enable_push

        if pose not in POSES:
            raise ValueError(f"Unknown pose '{pose}'. Choose from: {list(POSES)}")
        self._pose_name = pose

        MODELS_DIR.mkdir(exist_ok=True)

        self._model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self._data  = mujoco.MjData(self._model)

        # Actuator → joint address maps
        nu = self._model.nu
        self._act_names: list[str] = []
        self._qadr = np.empty(nu, dtype=np.int32)
        self._vadr = np.empty(nu, dtype=np.int32)
        for a in range(nu):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or f"act{a}"
            self._act_names.append(name)
            j = self._model.actuator_trnid[a, 0]
            self._qadr[a] = self._model.jnt_qposadr[j]
            self._vadr[a] = self._model.jnt_dofadr[j]
        self._name2id: dict[str, int] = {n: i for i, n in enumerate(self._act_names)}

        # ── Stand-prep reference (selected by --pose) ─────────────────────────
        _pose_dict = POSES[self._pose_name]
        self._stand_joint_q = np.zeros(nu, dtype=np.float64)
        for name, q in _pose_dict.items():
            if name in self._name2id:
                self._stand_joint_q[self._name2id[name]] = q

        # ── Auto-calibrate initial pelvis height ──────────────────────────────
        self._pelvis_z_init = _find_floor_height(
            self._model, self._data, _pose_dict,
            self._name2id, self._qadr,
        )

        # Build full initial qpos
        self._stand_qpos = np.zeros(self._model.nq, dtype=np.float64)
        self._stand_qpos[2] = self._pelvis_z_init
        self._stand_qpos[3] = 1.0   # quaternion w
        for name, q in _pose_dict.items():
            if name in self._name2id:
                self._stand_qpos[self._qadr[self._name2id[name]]] = q
        self._stand_qpos_joints = self._stand_qpos[7:].copy()

        # Control range (for clipping desired positions)
        self._ctrl_lo = self._model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_hi = self._model.actuator_ctrlrange[:, 1].copy()

        # ── Body IDs ──────────────────────────────────────────────────────────
        self._pelvis_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
        if self._pelvis_id < 0:
            # Fallback: first body whose name contains 'pelvis'
            ids = _bodies_matching(self._model, "pelvis")
            self._pelvis_id = ids[0] if ids else 1

        self._torso_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        if self._torso_id < 0:
            ids = _bodies_matching(self._model, "torso")
            self._torso_id = ids[0] if ids else self._pelvis_id

        # Foot body IDs (for sliding penalty)
        foot_body_ids = _bodies_matching(self._model, r"foot")
        self._foot_geoms: set[int] = _geoms_of_bodies(self._model, foot_body_ids)
        # Foot body positions (for stance geometry)
        self._foot_body_ids = foot_body_ids

        # Undesired contact bodies: thighs and shins
        thigh_ids = _bodies_matching(self._model, r"thigh")
        shin_ids  = _bodies_matching(self._model, r"shin")
        self._bad_contact_geoms: set[int] = _geoms_of_bodies(
            self._model, thigh_ids + shin_ids)

        # ── Joint group masks (for reward computation) ─────────────────────────
        def _mask(patterns: list[str]) -> np.ndarray:
            m = np.zeros(nu, dtype=bool)
            for i, name in enumerate(self._act_names):
                if any(re.search(p, name) for p in patterns):
                    m[i] = True
            return m

        self._mask_arms  = _mask([r".*shoulder.*", r".*elbow.*"])
        self._mask_torso = _mask([r"spine", r"neck"])
        self._mask_hip_yaw  = _mask([r".*hip_z"])
        self._mask_ankle_x  = _mask([r".*ankle_x"])

        # ── Action delay buffer ───────────────────────────────────────────────
        # Stores the last MAX_DELAY_STEPS+1 desired position vectors.
        # On each step the policy's q_des is enqueued and the delayed
        # version (0-2 steps old) is actually applied to the sim.
        self._delay_buf: deque[np.ndarray] = deque(
            [np.zeros(nu, dtype=np.float64)] * (MAX_DELAY_STEPS + 1),
            maxlen=MAX_DELAY_STEPS + 1,
        )
        self._cur_delay: int = 0  # resampled each episode

        # ── Torso mass for domain randomisation ───────────────────────────────
        self._torso_mass_default = float(self._model.body_mass[self._torso_id])

        # ── Push state ────────────────────────────────────────────────────────
        self._next_push_step: int = 0   # control step at which to apply push

        # ── Height command ────────────────────────────────────────────────────
        self._target_height: float = TARGET_HEIGHT_DEFAULT

        # ── Previous joint acceleration estimate (for dof_acc penalty) ────────
        self._prev_joint_vel = np.zeros(self._model.nv - 6, dtype=np.float64)

        # ── Spaces ────────────────────────────────────────────────────────────
        nq_j = self._model.nq - 7    # joint dofs (no freejoint)
        nv_j = self._model.nv - 6
        # obs: gravity_proj(3) + pelvis_z(1) + ang_vel(3) + lin_vel(3)
        #      + height_cmd(1) + joint_pos(nq_j) + joint_vel(nv_j) + prev_action(nu)
        obs_dim = 3 + 1 + 3 + 3 + 1 + nq_j + nv_j + nu
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(nu,), dtype=np.float32)

        self._prev_action = np.zeros(nu, dtype=np.float32)
        self._step_count  = 0
        self._viewer      = None

        print(f"[AlexStandIsaacEnv] scene       : {SCENE_XML.name}")
        print(f"[AlexStandIsaacEnv] pose        : {self._pose_name}")
        print(f"[AlexStandIsaacEnv] pelvis_z_init: {self._pelvis_z_init:.4f}")
        print(f"[AlexStandIsaacEnv] obs_dim     : {obs_dim}")
        print(f"[AlexStandIsaacEnv] action_dim  : {nu}")
        print(f"[AlexStandIsaacEnv] push        : {'enabled' if self._enable_push else 'disabled'}")
        print(f"[AlexStandIsaacEnv] decimation  : {DECIMATION}")
        print(f"[AlexStandIsaacEnv] delay steps : {MIN_DELAY_STEPS}-{MAX_DELAY_STEPS}")

    # ── Observation ───────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        pid = self._pelvis_id
        # Rotation matrix of pelvis (world frame)
        rot = self._data.xmat[pid].reshape(3, 3)

        # Gravity projection: gravity vector ([0,0,-1]) expressed in pelvis frame
        # = R^T @ [0, 0, -1].  For a perfectly upright robot this is [0, 0, -1].
        # Mirrors IsaacLab projected_gravity observation term.
        gravity_world = np.array([0.0, 0.0, -1.0])
        gravity_proj  = rot.T @ gravity_world     # shape (3,)

        pelvis_z      = float(self._data.xpos[pid, 2])
        # Angular velocity in body frame
        ang_vel_world = self._data.qvel[3:6]
        ang_vel_body  = rot.T @ ang_vel_world
        lin_vel       = self._data.qvel[:3]

        joint_pos_offset = self._data.qpos[7:] - self._stand_qpos_joints
        joint_vel        = self._data.qvel[6:].copy()

        # Observation noise — mirrors AlexObservationsCfg Unoise terms
        # (enable_corruption=True during training, disabled in PLAY/eval)
        if self.random_init:
            ang_vel_body     = ang_vel_body + self.np_random.uniform(
                -OBS_NOISE_ANG_VEL, OBS_NOISE_ANG_VEL, size=3)
            gravity_proj     = gravity_proj + self.np_random.uniform(
                -OBS_NOISE_GRAV, OBS_NOISE_GRAV, size=3)
            joint_pos_offset = joint_pos_offset + self.np_random.uniform(
                -OBS_NOISE_JOINT_POS, OBS_NOISE_JOINT_POS, size=joint_pos_offset.shape)
            joint_vel        = joint_vel + self.np_random.uniform(
                -OBS_NOISE_JOINT_VEL, OBS_NOISE_JOINT_VEL, size=joint_vel.shape)

        return np.concatenate([
            gravity_proj,                             # (3)  orientation signal
            [pelvis_z],                               # (1)  height
            ang_vel_body,                             # (3)  angular velocity in body frame
            lin_vel,                                  # (3)  linear velocity (world)
            [self._target_height],                    # (1)  height command
            joint_pos_offset,                         # joint pos offsets
            joint_vel,                                # joint velocities
            self._prev_action,                        # previous action
        ]).astype(np.float32)

    # ── Termination ───────────────────────────────────────────────────────────

    def _is_terminated(self) -> bool:
        pid    = self._pelvis_id
        if self._data.xpos[pid, 2] < PELVIS_Z_MIN:
            return True
        rot    = self._data.xmat[pid].reshape(3, 3)
        upright = float(rot[2, 2])   # cos of tilt angle
        return upright < UPRIGHT_MIN_COS

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self, action: np.ndarray,
                        prev_action: np.ndarray) -> tuple[float, dict]:
        pid = self._pelvis_id
        rot = self._data.xmat[pid].reshape(3, 3)

        # --- Height tracking (track_lin_pos_z_exp, std=0.05) -----------------
        pelvis_z = float(self._data.xpos[pid, 2])
        z_err    = pelvis_z - self._target_height
        r_height = W_HEIGHT * math.exp(-(z_err ** 2) / (2.0 * HEIGHT_STD ** 2))

        # --- Flat orientation (flat_orientation_exp, std=0.2) ----------------
        # Penalise deviation of gravity projection xy from zero.
        gravity_proj = rot.T @ np.array([0.0, 0.0, -1.0])
        ori_err_sq   = float(gravity_proj[0]**2 + gravity_proj[1]**2)
        r_orient     = W_ORIENTATION * math.exp(
            -ori_err_sq / (2.0 * ORIENTATION_STD ** 2))

        # --- Alive bonus ------------------------------------------------------
        r_alive = W_ALIVE

        # --- Action rate L2 (action_rate_l2, w=-0.1) -------------------------
        delta_action = action - prev_action
        r_action_rate = W_ACTION_RATE * float(np.dot(delta_action, delta_action))

        # --- Lin vel Z L2 (lin_vel_z_l2, w=-0.1) ----------------------------
        r_lin_vel_z = W_LIN_VEL_Z * float(self._data.qvel[2] ** 2)

        # --- Angular velocity XY L2 (ang_vel_xy_l2, w=-0.2) -----------------
        ang_vel_world = self._data.qvel[3:6]
        ang_vel_body  = rot.T @ ang_vel_world
        r_ang_vel_xy  = W_ANG_VEL_XY * float(ang_vel_body[0]**2 + ang_vel_body[1]**2)

        # --- Joint velocity L2 (dof_acc proxy) --------------------------------
        joint_vel     = self._data.qvel[6:]
        r_joint_vel   = W_JOINT_VEL * float(np.dot(joint_vel, joint_vel))

        # --- Joint torque L2 (dof_torques_l2, w=-1e-5) -----------------------
        # Approximate: τ ≈ qfrc_actuator for the joint dofs
        joint_torque  = self._data.qfrc_actuator[6:]
        r_torque      = W_JOINT_TORQUE * float(np.dot(joint_torque, joint_torque))

        # --- Joint deviation — arms (joint_deviation_arms L1, w=-0.6) --------
        q_arm_err = float(np.sum(np.abs(
            self._data.qpos[self._qadr[self._mask_arms]]
            - self._stand_joint_q[self._mask_arms]
        )))
        r_joint_arms = W_JOINT_ARMS * q_arm_err

        # --- Joint deviation — torso (joint_deviation_torso L1, w=-0.6) ------
        q_torso_err = float(np.sum(np.abs(
            self._data.qpos[self._qadr[self._mask_torso]]
            - self._stand_joint_q[self._mask_torso]
        )))
        r_joint_torso = W_JOINT_TORSO * q_torso_err

        # --- Hip yaw deviation (joint_mean_deviation_hip, w=-1.5) ------------
        q_hipz_err = float(np.sum(np.abs(
            self._data.qpos[self._qadr[self._mask_hip_yaw]]
            - self._stand_joint_q[self._mask_hip_yaw]
        )))
        r_hip_yaw = W_JOINT_HIP_YAW * q_hipz_err

        # --- Ankle roll torque L2 (ankle_roll_torques_l2, w=-2e-5) ----------
        ankle_tau   = self._data.qfrc_actuator[self._vadr[self._mask_ankle_x]]
        r_ankle_tau = W_ANKLE_TORQUE * float(np.dot(ankle_tau, ankle_tau))

        # --- Ankle roll deviation L1 (joint_mean_deviation_ankle, w=-0.6) ---
        q_anklex_err = float(np.sum(np.abs(
            self._data.qpos[self._qadr[self._mask_ankle_x]]
            - self._stand_joint_q[self._mask_ankle_x]
        )))
        r_ankle_dev = W_JOINT_ANKLE_X * q_anklex_err

        # --- Foot sliding (foot_sliding) -------------------------------------
        # When a foot is in contact, penalise its horizontal velocity.
        r_foot_slide = 0.0
        for bid in self._foot_body_ids:
            geoms = _geoms_of_bodies(self._model, [bid])
            if _geom_in_contact(self._data, geoms):
                # Foot body linear velocity (world frame) from subtree vel
                bvel = self._data.cvel[bid]   # 6D: [rot(3), lin(3)]
                foot_linvel_xy = bvel[3:5]     # x, y components
                r_foot_slide += W_FOOT_SLIDE * float(np.dot(foot_linvel_xy,
                                                             foot_linvel_xy))

        # --- Undesired contacts (thigh / shin, w=-0.1) -----------------------
        r_bad_contact = 0.0
        if self._bad_contact_geoms and _geom_in_contact(self._data,
                                                         self._bad_contact_geoms):
            r_bad_contact = W_UNDESIRED_CONT

        total = (r_height + r_orient + r_alive + r_action_rate
                 + r_lin_vel_z + r_ang_vel_xy + r_joint_vel + r_torque
                 + r_joint_arms + r_joint_torso + r_hip_yaw + r_ankle_tau
                 + r_ankle_dev + r_foot_slide + r_bad_contact)

        return float(total), dict(
            r_height=r_height, r_orient=r_orient, r_alive=r_alive,
            r_action_rate=r_action_rate, r_lin_vel_z=r_lin_vel_z,
            r_ang_vel_xy=r_ang_vel_xy, r_joint_vel=r_joint_vel,
            r_torque=r_torque, r_joint_arms=r_joint_arms,
            r_joint_torso=r_joint_torso, r_hip_yaw=r_hip_yaw,
            r_ankle_tau=r_ankle_tau, r_ankle_dev=r_ankle_dev,
            r_foot_slide=r_foot_slide, r_bad_contact=r_bad_contact,
            pelvis_z=pelvis_z, upright=float(rot[2, 2]),
        )

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self._model, self._data)

        # Restore default torso mass (undo previous episode's randomisation)
        self._model.body_mass[self._torso_id] = self._torso_mass_default

        # ── Domain randomisation — from isaacsimlab/alex_stand_env_cfg.py ───

        if self.random_init:
            # Mass randomisation: ±2 kg on torso (add_base_mass event)
            delta_mass = self.np_random.uniform(-MASS_RAND_KG, MASS_RAND_KG)
            self._model.body_mass[self._torso_id] = (
                self._torso_mass_default + delta_mass)

        # Start from the reference standing pose
        self._data.qpos[:] = self._stand_qpos.copy()
        self._data.ctrl[:] = self._stand_joint_q.copy()

        if self.random_init:
            # Joint position randomisation — reset_robot_joints position_range=±0.7
            self._data.qpos[7:] += self.np_random.uniform(
                -JOINT_RESET_RANGE, JOINT_RESET_RANGE,
                size=self._model.nq - 7,
            )
            # Base position: ±0.5 m x/y, ±0.01 m z (reset_base pose_range)
            self._data.qpos[0] += float(self.np_random.uniform(-BASE_RESET_XY, BASE_RESET_XY))
            self._data.qpos[1] += float(self.np_random.uniform(-BASE_RESET_XY, BASE_RESET_XY))
            self._data.qpos[2] += float(self.np_random.uniform(-BASE_RESET_Z,  BASE_RESET_Z))
            # Orientation: full yaw ±π, roll/pitch ±0.1 rad (reset_base pose_range)
            roll  = float(self.np_random.uniform(-BASE_RESET_ROLL_PITCH, BASE_RESET_ROLL_PITCH))
            pitch = float(self.np_random.uniform(-BASE_RESET_ROLL_PITCH, BASE_RESET_ROLL_PITCH))
            yaw   = float(self.np_random.uniform(-BASE_RESET_YAW, BASE_RESET_YAW))
            dq    = _quat_from_euler_xyz(roll, pitch, yaw)
            self._data.qpos[3:7] = _quat_mul(self._data.qpos[3:7].copy(), dq)

        self._data.qvel[:] = 0.0

        if self.random_init:
            # Joint velocity randomisation — reset_robot_joints velocity_range=±0.5
            # Applied after zeroing so the noise is not overwritten.
            self._data.qvel[6:] = self.np_random.uniform(
                -JOINT_RESET_VEL, JOINT_RESET_VEL,
                size=self._model.nv - 6,
            )

        mujoco.mj_forward(self._model, self._data)

        # Resample episode-level randomisation
        self._cur_delay = int(self.np_random.integers(
            MIN_DELAY_STEPS, MAX_DELAY_STEPS + 1))
        for _ in range(len(self._delay_buf)):
            self._delay_buf.append(self._stand_joint_q.copy())

        # Sample target height command (commands.base_height range)
        self._target_height = float(self.np_random.uniform(
            TARGET_HEIGHT_MIN, TARGET_HEIGHT_MAX))

        # Schedule first push disturbance
        push_steps = self.np_random.uniform(PUSH_INTERVAL_MIN, PUSH_INTERVAL_MAX)
        self._next_push_step = int(push_steps / CONTROL_DT)

        self._prev_action      = np.zeros(self._model.nu, dtype=np.float32)
        self._prev_joint_vel   = np.zeros(self._model.nv - 6, dtype=np.float64)
        self._step_count       = 0

        return self._get_obs(), {}

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        prev_action = self._prev_action.copy()

        # Map action → desired joint positions around stand pose
        q_des = self._stand_joint_q + action.astype(np.float64) * ACTION_SCALE
        q_des = np.clip(q_des, self._ctrl_lo, self._ctrl_hi)

        # Enqueue desired position into delay buffer
        self._delay_buf.append(q_des.copy())

        # Retrieve the delayed command (index 0 = MAX_DELAY_STEPS ago = most delayed)
        delay_idx = MAX_DELAY_STEPS - self._cur_delay
        delayed_q_des = self._delay_buf[delay_idx]

        # Apply push disturbance if scheduled (push_robot event, --push to enable)
        # Velocity ranges from isaacsimlab/alex_stand_env_cfg.py push_robot params.
        if self._enable_push and self._step_count == self._next_push_step and self._step_count > 0:
            self._data.qvel[0] += float(self.np_random.uniform(-PUSH_VEL_XY,         PUSH_VEL_XY))
            self._data.qvel[1] += float(self.np_random.uniform(-PUSH_VEL_XY,         PUSH_VEL_XY))
            self._data.qvel[2] += float(self.np_random.uniform(-PUSH_VEL_Z,          PUSH_VEL_Z))
            self._data.qvel[3] += float(self.np_random.uniform(-PUSH_VEL_ROLL_PITCH, PUSH_VEL_ROLL_PITCH))
            self._data.qvel[4] += float(self.np_random.uniform(-PUSH_VEL_ROLL_PITCH, PUSH_VEL_ROLL_PITCH))
            self._data.qvel[5] += float(self.np_random.uniform(-PUSH_VEL_YAW,        PUSH_VEL_YAW))
            # Schedule next push
            push_steps = self.np_random.uniform(PUSH_INTERVAL_MIN, PUSH_INTERVAL_MAX)
            self._next_push_step = self._step_count + int(push_steps / CONTROL_DT)

        # Decimation: run DECIMATION sim steps per control step
        for _ in range(DECIMATION):
            self._data.ctrl[:] = delayed_q_des
            mujoco.mj_step(self._model, self._data)

        self._step_count += 1

        obs          = self._get_obs()
        reward, info = self._compute_reward(action, prev_action)
        terminated   = self._is_terminated()
        truncated    = self._step_count >= MAX_STEPS

        if terminated:
            reward += FALL_PENALTY
            info["fall"] = True

        self._prev_action      = action.copy()
        self._prev_joint_vel   = self._data.qvel[6:].copy()

        return obs, reward, terminated, truncated, info

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer as _mv
                self._viewer = _mv.launch_passive(self._model, self._data)
            self._viewer.sync()
        elif self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self._model, height=480, width=640)
            renderer.update_scene(self._data)
            return renderer.render()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    total_timesteps: int = 10_000_000,
    n_envs: int = 8,
    pose: str = "isaaclab",
    enable_push: bool = True,
) -> None:
    """
    Train with SB3 PPO using hyperparameters from AlexStandingEnvPPORunnerCfg.

    RSL-RL → SB3 translation:
      actor/critic:    [128,128,128] ELU  (identical)
      learning_rate:   1e-3 adaptive  →  1e-3 fixed
                       (RSL-RL uses adaptive KL schedule; SB3 uses fixed LR;
                        1e-3 matches the isaacsimlab target value)
      gamma:           0.99           →  0.99
      gae_lambda:      0.95           →  0.95
      clip_param:      0.2            →  clip_range=0.2
      entropy_coef:    0.01           →  ent_coef=0.01
      max_grad_norm:   1.0            →  max_grad_norm=1.0
      num_learning_epochs: 5          →  n_epochs=5
      num_mini_batches:    4          →  batch_size = n_steps*n_envs // 4
      num_steps_per_env:   24 (RSL)   →  n_steps=2048 (same as original; more
                                          stable gradients with fewer CPU envs)
    """
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    MODELS_DIR.mkdir(exist_ok=True)

    n_steps    = 2048
    batch_size = (n_steps * n_envs) // 4   # 4 mini-batches

    env_kwargs      = {"random_init": True,  "pose": pose, "enable_push": enable_push}
    eval_env_kwargs = {"random_init": False, "pose": pose, "enable_push": False}

    try:
        vec_env = make_vec_env(
            AlexStandIsaacEnv, n_envs=n_envs,
            env_kwargs=env_kwargs,
            vec_env_cls=SubprocVecEnv,
        )
    except Exception:
        print("SubprocVecEnv failed, falling back to DummyVecEnv.")
        from stable_baselines3.common.vec_env import DummyVecEnv
        vec_env = make_vec_env(
            AlexStandIsaacEnv, n_envs=n_envs,
            env_kwargs=env_kwargs,
            vec_env_cls=DummyVecEnv,
        )

    vec_env = VecNormalize(
        vec_env, norm_obs=True, norm_reward=True,
        clip_obs=10.0, clip_reward=10.0, gamma=0.99,
    )

    eval_env = make_vec_env(AlexStandIsaacEnv, n_envs=1, env_kwargs=eval_env_kwargs)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            clip_obs=10.0, training=False)

    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=1e-3,            # 1e-3 — matches AlexStandingEnvPPORunnerCfg
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=5,                    # num_learning_epochs=5
        gamma=0.99,                    # gamma=0.99
        gae_lambda=0.95,               # lam=0.95
        clip_range=0.2,                # clip_param=0.2
        ent_coef=0.01,                 # entropy_coef=0.01
        vf_coef=1.0,                   # value_loss_coef=1.0
        max_grad_norm=1.0,             # max_grad_norm=1.0
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128, 128], vf=[128, 128, 128]),  # [128,128,128]
            activation_fn=torch.nn.ELU,  # activation="elu"
            log_std_init=-1.0,
            ortho_init=True,
        ),
        verbose=1,
        tensorboard_log=str(MODELS_DIR / "tensorboard"),
        seed=42,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=max(100_000 // n_envs, 1),
        save_path=str(MODELS_DIR / "checkpoints"),
        name_prefix="alex_stand_isaac",
        save_vecnormalize=True,
        verbose=1,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(MODELS_DIR / "best"),
        log_path=str(MODELS_DIR / "eval_logs"),
        eval_freq=max(50_000 // n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    print(f"\nTraining for {total_timesteps:,} timesteps across {n_envs} envs.")
    print(f"pose={pose}, push={'on' if enable_push else 'off'}")
    print(f"n_steps={n_steps}, batch_size={batch_size}, n_epochs=5, lr=1e-3\n")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[ckpt_cb, eval_cb],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted.")

    model.save(str(MODELS_DIR / "alex_stand_isaac_final"))
    vec_env.save(str(MODELS_DIR / "vec_normalize_final.pkl"))
    print(f"\nSaved to {MODELS_DIR}/")
    vec_env.close()
    eval_env.close()


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model_path: str, vec_norm_path: Optional[str] = None,
             n_episodes: int = 5) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    raw = Path(model_path).expanduser()
    if not raw.exists() and raw.suffix == "":
        raw = raw.with_suffix(".zip")

    env     = AlexStandIsaacEnv(render_mode="human", random_init=False)
    vec_env = DummyVecEnv([lambda: env])

    if vec_norm_path:
        vnp = Path(vec_norm_path).expanduser()
        if vnp.exists():
            vec_env = VecNormalize.load(str(vnp), vec_env)
            vec_env.training    = False
            vec_env.norm_reward = False
        else:
            print("VecNormalize stats not found — observations will not be normalised.")

    model = PPO.load(str(raw), env=vec_env)
    dt    = CONTROL_DT   # sleep to real-time

    for ep in range(1, n_episodes + 1):
        obs          = vec_env.reset()
        total_reward = 0.0
        info_last    = {}
        for step in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            total_reward += float(reward[0])
            info_last     = info[0]
            env.render()
            time.sleep(dt)
            if done[0]:
                break

        fell = info_last.get("fall", False)
        print(
            f"  Episode {ep:2d}  reward={total_reward:8.1f}  "
            f"pelvis_z={info_last.get('pelvis_z', 0.0):.3f}  "
            f"upright={info_last.get('upright', 0.0):.3f}  "
            f"steps={step+1:4d}  {'FELL' if fell else 'OK'}"
        )

    env.close()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--eval",      metavar="MODEL_PATH", default=None,
                    help="path to saved model to evaluate (skips training)")
    ap.add_argument("--vec-norm",  metavar="PKL_PATH",   default=None,
                    help="VecNormalize stats file for evaluation")
    ap.add_argument("--episodes",  type=int, default=5)
    ap.add_argument("--timesteps", type=int, default=10_000_000)
    ap.add_argument("--n-envs",    type=int, default=8)
    ap.add_argument(
        "--pose",
        choices=["isaaclab", "original", "zero"],
        default="isaaclab",
        help=(
            "isaaclab: bent-knee from isaacsimlab/ (default, harder); "
            "original: near-upright from alex-stand-ppo.py (easier, faster convergence); "
            "zero: all joints at 0 (sanity check)"
        ),
    )
    ap.add_argument(
        "--no-push", dest="enable_push", action="store_false",
        help="disable random push disturbances (push enabled by default, matching isaacsimlab)",
    )
    ap.set_defaults(enable_push=True)
    args = ap.parse_args()

    if args.eval:
        evaluate(args.eval, args.vec_norm, args.episodes)
    else:
        train(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            pose=args.pose,
            enable_push=args.enable_push,
        )
