# Copyright (c) 2026, Patrizio Acquadro.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command helpers for the SAKE EZGripper used by Alex Purdue."""

from __future__ import annotations

import math
from numbers import Real
from typing import Literal

EZGripperSide = Literal["left", "right"]
EZGRIPPER_CLOSED_ANGLE_RAD = 1.30


def alex_purdue_ezgripper_targets(
    position: Real, side: EZGripperSide
) -> dict[str, float]:
    """Map normalized closed=0/open=1 position to leader-joint radians."""

    if isinstance(position, bool) or not isinstance(position, Real):
        raise TypeError("EZGripper position must be a real scalar")
    normalized = float(position)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError("EZGripper position must be finite and within [0, 1]")
    if side not in ("left", "right"):
        raise ValueError("EZGripper side must be 'left' or 'right'")
    angle_rad = EZGRIPPER_CLOSED_ANGLE_RAD * (1.0 - normalized)
    return {
        f"{side}_ezgripper_knuckle_palm_l1_1": angle_rad,
        f"{side}_ezgripper_knuckle_l1_l2_1": angle_rad,
    }


__all__ = [
    "EZGripperSide",
    "alex_purdue_ezgripper_targets",
]
