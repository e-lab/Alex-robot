"""mediapipe_to_alex.py

Retargets MediaPipe Pose world keypoints (produced by action_collect_video.py)
to Alex robot tracking NPZ format for use with Mjlab.

Usage
-----
    python motions/mediapipe_to_alex.py <keypoints.npz> [output.npz] [options]

Coordinate systems
------------------
MediaPipe world:    Y-up,  X right (person's right),  Z toward camera.
Simulator (MuJoCo): Z-up,  X forward,                 Y left.
Transform applied:  sim_x = -mp_z,   sim_y = -mp_x,   sim_z = -mp_y

Joint sign conventions (Alex)
------------------------------
Hip/shoulder pitch _y : positive = flexion (limb forward).
Hip/shoulder roll  _x : positive = abduction (limb away from midline).
Hip yaw            _z : set to 0 (not recoverable from 2-D keypoints).
Knee/elbow         _y : positive = flexion (joint bends).
Spine              _z : signed yaw of thorax relative to pelvis.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.alex.env_cfgs import alex_v1_flat_tracking_env_cfg
from mjlab.utils.lab_api.math import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_mul,
    quat_slerp,
)

# ──────────────────────────────────────────────────────────────────────────────
# Robot constants
# ──────────────────────────────────────────────────────────────────────────────

ALEX_JOINT_ORDER = (
    "spine_z",
    "left_hip_x",
    "left_hip_z",
    "left_hip_y",
    "left_knee_y",
    "right_hip_x",
    "right_hip_z",
    "right_hip_y",
    "right_knee_y",
    "left_shoulder_y",
    "left_shoulder_x",
    "left_elbow_y",
    "right_shoulder_y",
    "right_shoulder_x",
    "right_elbow_y",
)

# Approximate Alex pelvis height above the ground when standing [m].
# Adjust if the model uses a different default.
ALEX_HIP_HEIGHT = 0.88

# Standing prior from mjlab.asset_zoo.robots.alex_V1_description.alex_constants.ALEX_INIT_STATE
STAND_PRIOR_JOINT_POS = {
    "left_hip_y": -0.30,
    "left_knee_y": 0.60,
    "right_hip_y": -0.30,
    "right_knee_y": 0.60,
    "left_shoulder_y": 0.20,
    "right_shoulder_y": 0.20,
    "left_elbow_y": -0.60,
    "right_elbow_y": -0.60,
}

# ──────────────────────────────────────────────────────────────────────────────
# MediaPipe landmark indices
# ──────────────────────────────────────────────────────────────────────────────

_L_SHOULDER  = 11
_R_SHOULDER  = 12
_L_ELBOW     = 13
_R_ELBOW     = 14
_L_WRIST     = 15
_R_WRIST     = 16
_L_HIP       = 23
_R_HIP       = 24
_L_KNEE      = 25
_R_KNEE      = 26
_L_ANKLE     = 27
_R_ANKLE     = 28

# ──────────────────────────────────────────────────────────────────────────────
# Geometry utilities
# ──────────────────────────────────────────────────────────────────────────────

def _norm(v: np.ndarray) -> np.ndarray:
    """Unit vector along last axis; safe for near-zero vectors."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-8)


def _angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle (rad) between unit vectors, computed along last axis."""
    return np.arccos(np.clip(np.einsum("...i,...i", a, b), -1.0, 1.0))


def _mp_to_sim(pts: np.ndarray) -> np.ndarray:
    """MediaPipe world → simulator (Z-up) coordinates.  Shape: (..., 3)."""
    return np.stack([-pts[..., 2],   # X_sim = -Z_mp  (forward)
                     -pts[..., 0],   # Y_sim = -X_mp  (left)
                     -pts[..., 1]],  # Z_sim = -Y_mp  (up)
                    axis=-1)


def _build_frame(right: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal frame from a 'right' (+X) direction vector (N, 3).

    Assumes Z-up world.
    Returns (right, forward, up) each (N, 3).
    Convention: right × forward = up  →  forward = up × right (Z-up right-hand).
    """
    N = right.shape[0]
    world_up = np.broadcast_to([[0.0, 0.0, 1.0]], (N, 3)).copy()
    # Orthogonalize world-up w.r.t. right to get a clean up vector.
    up = _norm(world_up - np.einsum("ni,ni->n", world_up, right)[:, None] * right)
    fwd = _norm(np.cross(up, right))   # Z × X = Y in standard Z-up right-hand frame
    return right, fwd, up


def _to_local(vec: np.ndarray,
              right: np.ndarray,
              fwd: np.ndarray,
              up: np.ndarray) -> np.ndarray:
    """Express world vectors in a local frame (X=right, Y=fwd, Z=up).
    Returns (N, 3) with components (local_x, local_y, local_z).
    """
    return np.stack([
        np.einsum("ni,ni->n", vec, right),
        np.einsum("ni,ni->n", vec, fwd),
        np.einsum("ni,ni->n", vec, up),
    ], axis=-1)


def _limb_pitch_roll(seg_local: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Pitch (Y) and roll (X) angles from a limb segment in the parent local frame.

    Frame convention: X=right, Y=forward, Z=up.
    Neutral hanging-down direction: local = (0, 0, -1).

    pitch_y > 0  →  limb forward (flexion).
    roll_x  > 0  →  limb abducted away from midline (right for right limbs,
                     left for left limbs).
    """
    lx, ly, lz = seg_local[:, 0], seg_local[:, 1], seg_local[:, 2]
    pitch_y = np.arctan2(ly, -lz)
    # Sign flip so abduction is always positive regardless of side.
    sign = -1.0 if side == "left" else 1.0
    roll_x = np.arctan2(sign * lx, -lz)
    return pitch_y, roll_x

# ──────────────────────────────────────────────────────────────────────────────
# Core retargeting
# ──────────────────────────────────────────────────────────────────────────────

def retarget_keypoints(wkp: np.ndarray) -> np.ndarray:
    """Map MediaPipe world keypoints to Alex motion data.

    Parameters
    ----------
    wkp : (N, 33, 3)
        World-space landmark positions in MediaPipe coordinates (Y-up).

    Returns
    -------
    motion_xyzw : (N, 7 + 15)
        Each row: [root_x, root_y, root_z, quat_x, quat_y, quat_z, quat_w,
                   *joint_angles_in_ALEX_JOINT_ORDER]
        Quaternion in XYZW format (MotionResampler will convert to WXYZ).
    """
    # ── 1. Coordinate transform to simulator frame ────────────────────────────
    sim = _mp_to_sim(wkp)  # (N, 33, 3)

    l_hip  = sim[:, _L_HIP];   r_hip  = sim[:, _R_HIP]
    l_knee = sim[:, _L_KNEE];  r_knee = sim[:, _R_KNEE]
    l_ank  = sim[:, _L_ANKLE]; r_ank  = sim[:, _R_ANKLE]
    l_sho  = sim[:, _L_SHOULDER]; r_sho = sim[:, _R_SHOULDER]
    l_elb  = sim[:, _L_ELBOW];    r_elb = sim[:, _R_ELBOW]
    l_wri  = sim[:, _L_WRIST];    r_wri = sim[:, _R_WRIST]

    N = wkp.shape[0]

    # ── 2. Pelvis frame (X=right, Y=forward, Z=up) ───────────────────────────
    pelvis_right, pelvis_fwd, pelvis_up = _build_frame(_norm(r_hip - l_hip))

    # ── 3. Root position ──────────────────────────────────────────────────────
    # MediaPipe world is centred at the hip midpoint each frame, so
    # the XY displacement is lost.  We fix XY=0 and set Z to the Alex
    # default standing hip height.  Adjust ALEX_HIP_HEIGHT if needed.
    pelvis_pos = (l_hip + r_hip) / 2.0          # ≈ (0, 0, 0) in world coords
    root_pos = pelvis_pos.copy()
    root_pos[:, 2] += ALEX_HIP_HEIGHT

    # ── 4. Root quaternion from pelvis rotation matrix ────────────────────────
    # Columns of R are the world-space directions of the local axes.
    R = np.stack([pelvis_right, pelvis_fwd, pelvis_up], axis=-1)  # (N, 3, 3)
    root_quat_xyzw = Rotation.from_matrix(R).as_quat()            # (N, 4) XYZW

    # ── 5. Leg joints ─────────────────────────────────────────────────────────
    l_femur = _norm(l_knee - l_hip)
    r_femur = _norm(r_knee - r_hip)
    l_tibia = _norm(l_ank  - l_knee)
    r_tibia = _norm(r_ank  - r_knee)

    # Knee flexion: angle between the two segments (0 = straight leg).
    l_knee_y = _angle_between(l_femur, l_tibia)
    r_knee_y = _angle_between(r_femur, r_tibia)

    # Hip pitch + roll: express femur in pelvis frame.
    l_fem_local = _to_local(l_femur, pelvis_right, pelvis_fwd, pelvis_up)
    r_fem_local = _to_local(r_femur, pelvis_right, pelvis_fwd, pelvis_up)

    l_hip_y, l_hip_x = _limb_pitch_roll(l_fem_local, side="left")
    r_hip_y, r_hip_x = _limb_pitch_roll(r_fem_local, side="right")

    # Hip yaw: not recoverable from 2-D pose keypoints → zero.
    l_hip_z = np.zeros(N)
    r_hip_z = np.zeros(N)

    # ── 6. Arm joints ─────────────────────────────────────────────────────────
    thorax_right, thorax_fwd, thorax_up = _build_frame(_norm(r_sho - l_sho))

    l_upper = _norm(l_elb - l_sho)
    r_upper = _norm(r_elb - r_sho)
    l_fore  = _norm(l_wri - l_elb)
    r_fore  = _norm(r_wri - r_elb)

    l_arm_local = _to_local(l_upper, thorax_right, thorax_fwd, thorax_up)
    r_arm_local = _to_local(r_upper, thorax_right, thorax_fwd, thorax_up)

    l_sho_y, l_sho_x = _limb_pitch_roll(l_arm_local, side="left")
    r_sho_y, r_sho_x = _limb_pitch_roll(r_arm_local, side="right")

    l_elbow_y = _angle_between(l_upper, l_fore)
    r_elbow_y = _angle_between(r_upper, r_fore)

    # ── 7. Spine yaw: signed angle from pelvis_fwd to thorax_fwd about Z ──────
    p_xy = pelvis_fwd[:, :2]
    t_xy = thorax_fwd[:, :2]
    p_xy = p_xy / np.maximum(np.linalg.norm(p_xy, axis=-1, keepdims=True), 1e-8)
    t_xy = t_xy / np.maximum(np.linalg.norm(t_xy, axis=-1, keepdims=True), 1e-8)
    sin_z = p_xy[:, 0] * t_xy[:, 1] - p_xy[:, 1] * t_xy[:, 0]   # 2-D cross
    cos_z = np.einsum("ni,ni->n", p_xy, t_xy)
    spine_z = np.arctan2(sin_z, cos_z)

    # ── 8. Assemble in ALEX_JOINT_ORDER ───────────────────────────────────────
    joint_angles = np.column_stack([
        spine_z,    # spine_z
        l_hip_x,   # left_hip_x
        l_hip_z,   # left_hip_z
        l_hip_y,   # left_hip_y
        l_knee_y,  # left_knee_y
        r_hip_x,   # right_hip_x
        r_hip_z,   # right_hip_z
        r_hip_y,   # right_hip_y
        r_knee_y,  # right_knee_y
        l_sho_y,   # left_shoulder_y
        l_sho_x,   # left_shoulder_x
        l_elbow_y, # left_elbow_y
        r_sho_y,   # right_shoulder_y
        r_sho_x,   # right_shoulder_x
        r_elbow_y, # right_elbow_y
    ])  # (N, 15)

    return np.concatenate([root_pos, root_quat_xyzw, joint_angles], axis=1)  # (N, 22)


def _joint_index_map() -> dict[str, int]:
    return {name: i for i, name in enumerate(ALEX_JOINT_ORDER)}


def _build_stand_prior_vector() -> np.ndarray:
    prior = np.zeros(len(ALEX_JOINT_ORDER), dtype=np.float32)
    jidx = _joint_index_map()
    for name, val in STAND_PRIOR_JOINT_POS.items():
        prior[jidx[name]] = np.float32(val)
    return prior


def _detect_active_arm(wkp_xyz: np.ndarray) -> str:
    """Choose active arm by larger wrist travel in MediaPipe world coordinates."""
    l_w = wkp_xyz[:, _L_WRIST, :]
    r_w = wkp_xyz[:, _R_WRIST, :]
    l_travel = float(np.linalg.norm(np.diff(l_w, axis=0), axis=1).sum()) if len(l_w) > 1 else 0.0
    r_travel = float(np.linalg.norm(np.diff(r_w, axis=0), axis=1).sum()) if len(r_w) > 1 else 0.0
    return "left" if l_travel >= r_travel else "right"


def _smooth_columns(x: np.ndarray, alpha: float) -> np.ndarray:
    if not (0.0 < alpha <= 1.0):
        return x
    y = x.copy()
    for t in range(1, y.shape[0]):
        y[t] = (1.0 - alpha) * y[t - 1] + alpha * y[t]
    return y


def _joint_limits() -> dict[str, tuple[float, float]]:
    return {
        "spine_z": (-0.523599, 0.523599),
        "left_hip_x": (-0.349066, 0.872665),
        "left_hip_z": (-0.349066, 1.0472),
        "left_hip_y": (-2.61799, 0.785398),
        "left_knee_y": (0.0, 2.445206),
        "right_hip_x": (-0.872665, 0.349066),
        "right_hip_z": (-1.0472, 0.349066),
        "right_hip_y": (-2.61799, 0.785398),
        "right_knee_y": (0.0, 2.445206),
        "left_shoulder_y": (-3.141592, 1.22173),
        "left_shoulder_x": (-0.349066, 2.79253),
        "left_elbow_y": (-2.35619, 0.174533),
        "right_shoulder_y": (-3.141592, 1.22173),
        "right_shoulder_x": (-2.79253, 0.349066),
        "right_elbow_y": (-2.35619, 0.174533),
    }


def _compute_joint_confidence(wkp: np.ndarray) -> np.ndarray:
    """Build per-joint confidence weights (N, 15) from MediaPipe visibility."""
    N = wkp.shape[0]
    if wkp.shape[-1] < 4:
        return np.ones((N, len(ALEX_JOINT_ORDER)), dtype=np.float32)
    vis = np.clip(wkp[:, :, 3].astype(np.float32), 0.0, 1.0)
    jidx = _joint_index_map()
    conf = np.ones((N, len(ALEX_JOINT_ORDER)), dtype=np.float32)

    def meanv(*ids: int) -> np.ndarray:
        return np.mean(vis[:, list(ids)], axis=1)

    conf[:, jidx["spine_z"]] = meanv(_L_HIP, _R_HIP, _L_SHOULDER, _R_SHOULDER)
    conf[:, jidx["left_hip_x"]] = meanv(_L_HIP, _L_KNEE)
    conf[:, jidx["left_hip_z"]] = meanv(_L_HIP, _L_KNEE)
    conf[:, jidx["left_hip_y"]] = meanv(_L_HIP, _L_KNEE)
    conf[:, jidx["left_knee_y"]] = meanv(_L_HIP, _L_KNEE, _L_ANKLE)
    conf[:, jidx["right_hip_x"]] = meanv(_R_HIP, _R_KNEE)
    conf[:, jidx["right_hip_z"]] = meanv(_R_HIP, _R_KNEE)
    conf[:, jidx["right_hip_y"]] = meanv(_R_HIP, _R_KNEE)
    conf[:, jidx["right_knee_y"]] = meanv(_R_HIP, _R_KNEE, _R_ANKLE)
    conf[:, jidx["left_shoulder_y"]] = meanv(_L_SHOULDER, _L_ELBOW)
    conf[:, jidx["left_shoulder_x"]] = meanv(_L_SHOULDER, _L_ELBOW)
    conf[:, jidx["left_elbow_y"]] = meanv(_L_SHOULDER, _L_ELBOW, _L_WRIST)
    conf[:, jidx["right_shoulder_y"]] = meanv(_R_SHOULDER, _R_ELBOW)
    conf[:, jidx["right_shoulder_x"]] = meanv(_R_SHOULDER, _R_ELBOW)
    conf[:, jidx["right_elbow_y"]] = meanv(_R_SHOULDER, _R_ELBOW, _R_WRIST)

    # Keep a minimum prior influence when visibility drops.
    return np.clip(conf, 0.15, 1.0)


def apply_calibrated_standing_deltas(
    motion_xyzw: np.ndarray,
    wkp: np.ndarray,
    neutral_window: int = 20,
    lower_body_gain: float = 0.45,
    upper_body_gain: float = 0.90,
    spine_gain: float = 0.50,
    confidence_power: float = 1.0,
    smooth_alpha: float = 0.2,
) -> np.ndarray:
    """Full-body delta retargeting around stand prior with neutral-frame calibration."""
    out = motion_xyzw.copy()
    N = out.shape[0]
    if N == 0:
        return out

    jidx = _joint_index_map()
    limits = _joint_limits()
    prior = _build_stand_prior_vector()
    raw_j = out[:, 7:].copy()
    nw = int(max(3, min(neutral_window, N)))
    neutral = np.median(raw_j[:nw], axis=0, keepdims=True)
    raw_delta = raw_j - neutral
    out_j = np.repeat(prior[None, :], N, axis=0)
    conf = _compute_joint_confidence(wkp) ** max(0.0, confidence_power)

    lower = ["left_hip_x", "left_hip_z", "left_hip_y", "left_knee_y",
             "right_hip_x", "right_hip_z", "right_hip_y", "right_knee_y"]
    upper = ["left_shoulder_y", "left_shoulder_x", "left_elbow_y",
             "right_shoulder_y", "right_shoulder_x", "right_elbow_y"]

    for n in lower:
        i = jidx[n]
        out_j[:, i] = prior[i] + lower_body_gain * conf[:, i] * raw_delta[:, i]

    for n in upper:
        i = jidx[n]
        out_j[:, i] = prior[i] + upper_body_gain * conf[:, i] * raw_delta[:, i]

    s = jidx["spine_z"]
    out_j[:, s] = prior[s] + spine_gain * conf[:, s] * raw_delta[:, s]

    # Clamp to model limits.
    for name, (lo, hi) in limits.items():
        i = jidx[name]
        out_j[:, i] = np.clip(out_j[:, i], lo, hi)

    out[:, 7:] = _smooth_columns(out_j, alpha=smooth_alpha).astype(np.float32)

    # Stabilize root to neutral standing frame to avoid monocular drift.
    out[:, :3] = out[:1, :3]
    out[:, 3:7] = out[:1, 3:7]
    return out

# ──────────────────────────────────────────────────────────────────────────────
# MotionResampler (identical to lafan_to_alex.py)
# ──────────────────────────────────────────────────────────────────────────────

class MotionResampler:
    def __init__(
        self,
        motion_xyzw: torch.Tensor,
        input_fps: float,
        output_fps: float,
        device: str,
    ):
        self.motion_xyzw = motion_xyzw.to(torch.float32).to(device)
        self.input_fps = float(input_fps)
        self.output_fps = float(output_fps)
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.device = device

        self._split_inputs()
        self._interpolate()
        self._compute_velocities()

    def _split_inputs(self) -> None:
        self.base_pos_input = self.motion_xyzw[:, :3]
        # CSV/input stores XYZW; convert to WXYZ for simulator utilities.
        self.base_quat_input = self.motion_xyzw[:, 3:7][:, [3, 0, 1, 2]]
        self.joint_pos_input = self.motion_xyzw[:, 7:]

        self.num_input_frames = int(self.motion_xyzw.shape[0])
        if self.num_input_frames < 2:
            raise ValueError("Input must contain at least 2 frames.")

        self.duration = (self.num_input_frames - 1) * self.input_dt

    def _compute_frame_blend(
        self, times: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phase = times / self.duration
        idx0 = (phase * (self.num_input_frames - 1)).floor().long()
        idx1 = torch.minimum(idx0 + 1, torch.tensor(self.num_input_frames - 1))
        blend = phase * (self.num_input_frames - 1) - idx0
        return idx0, idx1, blend

    def _interpolate(self) -> None:
        times = torch.arange(
            0.0, self.duration, self.output_dt,
            device=self.device, dtype=torch.float32,
        )
        if times.numel() == 0:
            times = torch.tensor([0.0], device=self.device)

        self.num_frames = int(times.shape[0])
        idx0, idx1, blend = self._compute_frame_blend(times)

        self.base_pos = self._lerp(self.base_pos_input[idx0], self.base_pos_input[idx1], blend)
        self.base_quat = self._slerp(self.base_quat_input[idx0], self.base_quat_input[idx1], blend)
        self.joint_pos = self._lerp(self.joint_pos_input[idx0], self.joint_pos_input[idx1], blend)

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        return a * (1.0 - blend.unsqueeze(-1)) + b * blend.unsqueeze(-1)

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(a)
        for i in range(a.shape[0]):
            out[i] = quat_slerp(a[i], b[i], float(blend[i]))
        return out

    def _so3_derivative(self, rotations_wxyz: torch.Tensor, dt: float) -> torch.Tensor:
        if rotations_wxyz.shape[0] < 3:
            return torch.zeros(rotations_wxyz.shape[0], 3, device=rotations_wxyz.device)
        q_prev, q_next = rotations_wxyz[:-2], rotations_wxyz[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        return torch.cat([omega[:1], omega, omega[-1:]], dim=0)

    def _compute_velocities(self) -> None:
        self.base_lin_vel = torch.gradient(self.base_pos, spacing=self.output_dt, dim=0)[0]
        self.joint_vel = torch.gradient(self.joint_pos, spacing=self.output_dt, dim=0)[0]
        self.base_ang_vel = self._so3_derivative(self.base_quat, self.output_dt)

# ──────────────────────────────────────────────────────────────────────────────
# Main conversion
# ──────────────────────────────────────────────────────────────────────────────

def convert_mediapipe_to_alex_npz(
    input_npz: Path,
    output_npz: Path,
    input_fps: float,
    output_fps: float,
    device: str,
    frame_range: tuple[int, int] | None,
    retarget_mode: str,
    neutral_window: int,
    lower_body_gain: float,
    upper_body_gain: float,
    spine_gain: float,
    confidence_power: float,
    smooth_alpha: float,
) -> None:
    # ── Load keypoints ────────────────────────────────────────────────────────
    data = np.load(input_npz, allow_pickle=True)
    wkp = data["world_keypoints"]  # (N, 33, 4)

    if frame_range is not None:
        s, e = frame_range
        if s < 0 or e > wkp.shape[0] or s >= e:
            raise ValueError(f"frame_range ({s}, {e}) out of bounds for {wkp.shape[0]} frames.")
        wkp = wkp[s:e]

    wkp_xyz = wkp[:, :, :3].astype(np.float64)  # drop visibility channel

    print(f"[INFO] Loaded {wkp_xyz.shape[0]} frames from '{input_npz}'")

    # ── Retarget: keypoints → root pose + joint angles ────────────────────────
    motion_xyzw = retarget_keypoints(wkp_xyz).astype(np.float32)  # (N, 22)
    if retarget_mode == "calibrated_delta":
        motion_xyzw = apply_calibrated_standing_deltas(
            motion_xyzw=motion_xyzw,
            wkp=wkp,
            neutral_window=neutral_window,
            lower_body_gain=lower_body_gain,
            upper_body_gain=upper_body_gain,
            spine_gain=spine_gain,
            confidence_power=confidence_power,
            smooth_alpha=smooth_alpha,
        )
        print(
            "[INFO] Applied calibrated full-body stand-delta retarget:"
            f" neutral_window={neutral_window}, lower_body_gain={lower_body_gain},"
            f" upper_body_gain={upper_body_gain}, spine_gain={spine_gain},"
            f" confidence_power={confidence_power},"
            f" smooth_alpha={smooth_alpha}"
        )

    # ── Sanity checks for common retarget failures ────────────────────────────
    # Columns in motion_xyzw after root pose:
    # 0 spine_z, 1 l_hip_x, 2 l_hip_z, 3 l_hip_y, 4 l_knee_y,
    # 5 r_hip_x, 6 r_hip_z, 7 r_hip_y, 8 r_knee_y, ...
    j = motion_xyzw[:, 7:]
    checks = [
        ("left_hip_x", j[:, 1], (-2.0, 2.0)),
        ("right_hip_x", j[:, 5], (-2.0, 2.0)),
        ("left_hip_y", j[:, 3], (-2.6, 1.2)),
        ("right_hip_y", j[:, 7], (-2.6, 1.2)),
        ("left_knee_y", j[:, 4], (0.0, 2.8)),
        ("right_knee_y", j[:, 8], (0.0, 2.8)),
    ]
    for name, arr, (lo, hi) in checks:
        amin = float(np.min(arr))
        amax = float(np.max(arr))
        if amin < lo - 1e-3 or amax > hi + 1e-3:
            print(
                f"[WARN] {name} out of expected range [{lo:.2f}, {hi:.2f}] "
                f"(min={amin:.3f}, max={amax:.3f})."
            )

    # ── Resample to output FPS ────────────────────────────────────────────────
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, falling back to CPU.")
        device = "cpu"

    resampled = MotionResampler(
        motion_xyzw=torch.from_numpy(motion_xyzw),
        input_fps=input_fps,
        output_fps=output_fps,
        device=device,
    )

    # ── Simulate to capture full body state ───────────────────────────────────
    sim_cfg = SimulationCfg()
    sim_cfg.mujoco.timestep = 1.0 / output_fps

    scene = Scene(alex_v1_flat_tracking_env_cfg().scene, device=device)
    model = scene.compile()
    sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)

    robot: Entity = scene["robot"]
    robot_joint_indexes = robot.find_joints(ALEX_JOINT_ORDER, preserve_order=True)[0]

    log: dict[str, list | np.ndarray] = {
        "fps": np.array([output_fps], dtype=np.float32),
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    scene.reset()

    for i in range(resampled.num_frames):
        root_states = robot.data.default_root_state.clone()
        root_states[:, 0:3]  = resampled.base_pos[i : i + 1]
        root_states[:, :2]  += scene.env_origins[:, :2]
        root_states[:, 3:7]  = resampled.base_quat[i : i + 1]
        root_states[:, 7:10] = resampled.base_lin_vel[i : i + 1]
        root_states[:, 10:]  = resampled.base_ang_vel[i : i + 1]
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = resampled.joint_pos[i : i + 1]
        joint_vel[:, robot_joint_indexes] = resampled.joint_vel[i : i + 1]
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.forward()
        scene.update(sim.mj_model.opt.timestep)

        log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_link_pos_w[0].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_link_quat_w[0].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_link_lin_vel_w[0].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_link_ang_vel_w[0].cpu().numpy().copy())

    for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
              "body_lin_vel_w", "body_ang_vel_w"):
        log[k] = np.stack(log[k], axis=0)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_npz, **log)

    print(f"[INFO] Input NPZ:  {input_npz}")
    print(f"[INFO] Output NPZ: {output_npz}")
    print(f"[INFO] Frames: {log['joint_pos'].shape[0]}, fps: {output_fps}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retarget MediaPipe keypoint NPZ to Alex tracking NPZ."
    )
    parser.add_argument(
        "input_npz",
        type=Path,
        help="Path to the mediapipe keypoints NPZ (from action_collect_video.py).",
    )
    parser.add_argument(
        "output_npz",
        nargs="?",
        type=Path,
        default=None,
        help="Output path. Default: motions/<input_stem>_alex_tracking.npz",
    )
    parser.add_argument(
        "--input-fps", "--input_fps",
        dest="input_fps",
        type=float,
        default=30.0,
        help="FPS of the source video / keypoints (default: 30).",
    )
    parser.add_argument(
        "--output-fps", "--output_fps",
        dest="output_fps",
        type=float,
        default=50.0,
        help="FPS for the output tracking data (default: 50).",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--retarget-mode",
        choices=("calibrated_delta", "direct"),
        default="calibrated_delta",
        help="Retarget mode: calibrated full-body stand-prior deltas (recommended) or direct mapping.",
    )
    parser.add_argument(
        "--neutral-window",
        type=int,
        default=20,
        help="Number of initial frames used to calibrate neutral standing baseline.",
    )
    parser.add_argument(
        "--lower-body-gain",
        type=float,
        default=0.45,
        help="Delta gain for hips/knees in calibrated_delta mode.",
    )
    parser.add_argument(
        "--upper-body-gain",
        type=float,
        default=0.90,
        help="Delta gain for shoulders/elbows in calibrated_delta mode.",
    )
    parser.add_argument(
        "--spine-gain",
        type=float,
        default=0.50,
        help="Delta gain for spine yaw in calibrated_delta mode.",
    )
    parser.add_argument(
        "--confidence-power",
        type=float,
        default=1.0,
        help="Exponent for visibility confidence weighting (0 disables, 1 linear).",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.2,
        help="Exponential smoothing alpha for joint deltas in calibrated_delta mode.",
    )
    parser.add_argument(
        "--frame-range", "--frame_range",
        dest="frame_range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="0-indexed half-open frame slice [START, END) to process.",
    )
    args = parser.parse_args()

    motions_dir = Path(__file__).parent
    output_npz = args.output_npz or (
        motions_dir / f"{args.input_npz.stem}_alex_tracking.npz"
    )
    frame_range = tuple(args.frame_range) if args.frame_range is not None else None

    convert_mediapipe_to_alex_npz(
        input_npz=args.input_npz,
        output_npz=output_npz,
        input_fps=args.input_fps,
        output_fps=args.output_fps,
        device=args.device,
        frame_range=frame_range,
        retarget_mode=args.retarget_mode,
        neutral_window=args.neutral_window,
        lower_body_gain=args.lower_body_gain,
        upper_body_gain=args.upper_body_gain,
        spine_gain=args.spine_gain,
        confidence_power=args.confidence_power,
        smooth_alpha=args.smooth_alpha,
    )


if __name__ == "__main__":
    main()
