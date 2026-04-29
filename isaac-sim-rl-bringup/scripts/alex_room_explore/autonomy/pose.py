"""Pose helpers — yaw extraction from quaternion + Phase-1 fall monitor.

Pure functions / a tiny stateful class. No Isaac, no torch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence


def yaw_from_quat(quat_wxyz: Sequence[float]) -> float:
    """Yaw (Z-axis rotation) extracted from a (w, x, y, z) quaternion.

    Matches the convention used by Isaac's ``robot.data.root_quat_w``: scalar-first.
    Returns yaw in radians wrapped to [-pi, pi].
    """
    w, x, y, z = (float(v) for v in quat_wxyz)
    # Standard ZYX -> yaw: atan2(2(wz + xy), 1 - 2(y^2 + z^2))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


@dataclass
class FallMonitor:
    """Phase-1 fall stub.

    Latches True the first tick the robot's base height drops below
    ``fall_height_m`` *or* the projected-gravity tilt magnitude exceeds
    ``fall_tilt_norm``. Once latched, ``update`` always returns True until
    ``reset()`` is called (Phase 4 recovery will reset it after standing).

    Tilt check is optional — pass ``proj_grav_xy=None`` to skip it (height-only).
    """

    fall_height_m: float = 0.5
    fall_tilt_norm: float = 0.7
    fallen: bool = False

    def update(
        self,
        root_z: float,
        proj_grav_xy: Optional[Sequence[float]] = None,
    ) -> bool:
        """Return True iff the robot is currently considered fallen.

        proj_grav_xy: the (x, y) components of gravity expressed in the robot's
        base frame. With the robot upright, gravity points along -Z in the base
        frame, so x and y are ~0. Tilt magnitude = hypot(x, y); a value of 0.7
        corresponds to ~45 deg tilt.
        """
        if self.fallen:
            return True

        if root_z < self.fall_height_m:
            self.fallen = True
            return True

        if proj_grav_xy is not None:
            gx = float(proj_grav_xy[0])
            gy = float(proj_grav_xy[1])
            if math.hypot(gx, gy) > self.fall_tilt_norm:
                self.fallen = True
                return True

        return False

    def reset(self) -> None:
        self.fallen = False
