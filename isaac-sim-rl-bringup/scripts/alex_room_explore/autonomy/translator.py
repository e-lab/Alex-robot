"""FSM mode → ``_cmd = [vx, vy, yaw_rate, standing_flag]`` translator.

This is the single new piece of logic the Phase-1 plan introduces (see
``PLAN/autonomous_navigation_plan.md``):

    (target_xyz, robot_pose, FSM mode) -> _cmd = [vx, vy, yaw_rate, standing]

Per FSM mode (Phase-1 table from the plan):

    SEARCH    : vx=0, vy=0, yaw_rate=search_yaw, standing=0
    APPROACH  : vx=walk_speed if |heading_err| < heading_walk_deg else 0
                yaw_rate=clip(heading_kp * heading_err, +/- yaw_max)
                standing=0
    ARRIVED   : vx=0, vy=0, yaw_rate=0, standing=1
    FALLEN    : vx=0, vy=0, yaw_rate=0, standing=1

All output components are hard-clamped to ``GaitLimits`` (the policy training
distribution) regardless of caller-supplied gains.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


# ── Math helpers ─────────────────────────────────────────────────────────────
def wrap_to_pi(angle_rad: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def heading_error(
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    target_x: float,
    target_y: float,
) -> float:
    """Yaw delta (rad, wrapped) the robot needs to rotate to face the target."""
    dx = target_x - robot_x
    dy = target_y - robot_y
    return wrap_to_pi(math.atan2(dy, dx) - robot_yaw)


def forward_distance(
    robot_x: float,
    robot_y: float,
    target_x: float,
    target_y: float,
) -> float:
    """2D Euclidean distance robot -> target."""
    return math.hypot(target_x - robot_x, target_y - robot_y)


# ── Gait limits ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class GaitLimits:
    """Hard caps from the Alex ONNX policy's training distribution.

    Per ``PLAN/autonomous_navigation_plan.md``: vx<=0.4, vy<=0.3, yaw_rate<=0.4.
    Anything beyond these is undefined behaviour for the policy and is the most
    likely cause of falls when the cam-robot controller is ported verbatim.
    """

    vx_max: float = 0.4
    vy_max: float = 0.3
    yaw_rate_max: float = 0.4

    def clamp(self, vx: float, vy: float, yaw_rate: float) -> Tuple[float, float, float]:
        vx = max(-self.vx_max, min(self.vx_max, vx))
        vy = max(-self.vy_max, min(self.vy_max, vy))
        yaw_rate = max(-self.yaw_rate_max, min(self.yaw_rate_max, yaw_rate))
        return vx, vy, yaw_rate


# ── Translator ───────────────────────────────────────────────────────────────
def fsm_mode_to_cmd(
    mode: str,
    *,
    heading_err_rad: float = 0.0,
    walk_speed: float = 0.30,
    search_yaw: float = 0.30,
    heading_kp: float = 0.8,
    yaw_max: float = 0.40,
    heading_walk_deg: float = 30.0,
    limits: GaitLimits = GaitLimits(),
) -> Tuple[float, float, float, float]:
    """Translate an FSM mode + heading error into the 4-float ``_cmd`` array.

    Parameters
    ----------
    mode
        One of ``"search"``, ``"approach"``, ``"arrived"``, ``"fallen"``.
        Unknown modes are treated as ``"arrived"`` (safe stop).
    heading_err_rad
        Wrapped heading error to the goal (only used in APPROACH).
    walk_speed
        Forward velocity command in APPROACH (m/s). Will be clamped by limits.
    search_yaw
        Yaw rate in SEARCH (rad/s). Sign is preserved; clamped by limits.
    heading_kp
        Proportional gain on heading_err -> yaw_rate in APPROACH (rad/s per rad).
    yaw_max
        Soft cap on APPROACH yaw_rate before the gait-limit clamp.
    heading_walk_deg
        Below this absolute heading error (deg), APPROACH commands forward
        velocity. Above, the robot turns in place first.

    Returns
    -------
    (vx, vy, yaw_rate, standing_flag) — all floats. Standing flag is 0.0 or 1.0.
    """
    if mode == "search":
        vx, vy, yaw_rate = 0.0, 0.0, float(search_yaw)
        standing = 0.0
    elif mode == "approach":
        # Yaw: P-controller toward goal heading, soft-capped at yaw_max.
        yaw_rate = max(-yaw_max, min(yaw_max, heading_kp * heading_err_rad))
        # Forward: only walk when roughly facing target.
        if abs(heading_err_rad) < math.radians(heading_walk_deg):
            vx = float(walk_speed)
        else:
            vx = 0.0
        vy = 0.0
        standing = 0.0
    elif mode == "fallen":
        vx, vy, yaw_rate = 0.0, 0.0, 0.0
        standing = 1.0
    else:  # "arrived" or anything unrecognised -> safe stop
        vx, vy, yaw_rate = 0.0, 0.0, 0.0
        standing = 1.0

    vx, vy, yaw_rate = limits.clamp(vx, vy, yaw_rate)
    return vx, vy, yaw_rate, standing
