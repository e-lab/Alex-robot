"""Search / Approach / Arrived / Fallen FSM controller.

Ported from ``cam_room_explore_isaac.py::_step_autonomous`` but stripped of all
direct robot-pose mutation. The cam controller wrote ``robot.x/y/yaw``
directly; this one only reads pose and produces a 4-float command tuple. The
caller writes that tuple to the global ``_cmd`` consumed by the Alex ONNX
policy obs builder.

Phase-1 only — no SAM3 perception path; the goal source is plumbed in by the
caller via :class:`autonomy.goal.GoalState` (currently set once via
``set_fixed`` in ``fixed_xyz`` mode).

Phase 2+ extensions go into :class:`autonomy.goal.GoalState` (lock-on,
staleness) without changing this module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

from .goal import GoalState
from .translator import (
    GaitLimits,
    forward_distance,
    fsm_mode_to_cmd,
    heading_error,
)


# ── Mode names ───────────────────────────────────────────────────────────────
class FSMMode:
    SEARCH = "search"
    APPROACH = "approach"
    ARRIVED = "arrived"
    FALLEN = "fallen"


# ── Tunables (mirror configs/autonomy/approach.yaml) ─────────────────────────
@dataclass(frozen=True)
class FSMParams:
    """Subset of AutonomyConfig that affects FSM logic / actuation."""

    stop_dist: float = 1.0
    walk_speed: float = 0.30
    search_yaw: float = 0.30
    heading_kp: float = 0.8
    yaw_max: float = 0.40
    heading_walk_deg: float = 30.0
    stale_s: float = 5.0
    limits: GaitLimits = field(default_factory=GaitLimits)


# ── Controller ───────────────────────────────────────────────────────────────
class FSMController:
    """Stateful FSM that emits ``_cmd = (vx, vy, yaw_rate, standing)`` per tick.

    Usage::

        fsm = FSMController(params)
        goal = GoalState(); goal.set_fixed((3.0, 0.0, 0.0))
        cmd = fsm.step(robot_x, robot_y, robot_yaw, goal=goal, fallen=False)
        _cmd[:] = cmd

    The optional ``on_transition`` callback fires whenever ``mode`` changes
    (used by the main script to print state changes / log to Rerun).
    """

    def __init__(
        self,
        params: FSMParams,
        on_transition: Optional[Callable[[str, str, dict], None]] = None,
    ) -> None:
        self.params = params
        self.on_transition = on_transition
        self.mode: str = FSMMode.SEARCH
        self.arrived_printed: bool = False
        self.last_dist: Optional[float] = None
        self.last_heading_err: Optional[float] = None

    # ------------------------------------------------------------- compute
    # Hysteresis on the arrival boundary: enter ARRIVED at
    # ``dist < stop_dist``, leave only when we've drifted ``ARRIVE_HYST_M``
    # past that. Prevents tick-to-tick flapping when the robot ends up
    # sitting right on the ``stop_dist`` boundary (e.g. ``dist = 1.000m``
    # vs ``stop_dist = 1.000m`` — sub-mm gait jitter flips the decision).
    ARRIVE_HYST_M: float = 0.10

    def _decide_mode(
        self,
        goal: GoalState,
        forward_dist: Optional[float],
        fallen: bool,
    ) -> str:
        """Pick the next FSM mode given the current observation."""
        if fallen:
            return FSMMode.FALLEN
        if not goal.is_fresh(self.params.stale_s):
            return FSMMode.SEARCH
        if forward_dist is None:
            return FSMMode.APPROACH
        # Hysteresis: tighter threshold to enter ARRIVED, looser to leave.
        if self.mode == FSMMode.ARRIVED:
            if forward_dist > self.params.stop_dist + self.ARRIVE_HYST_M:
                return FSMMode.APPROACH
            return FSMMode.ARRIVED
        if forward_dist < self.params.stop_dist:
            return FSMMode.ARRIVED
        return FSMMode.APPROACH

    def step(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        *,
        goal: GoalState,
        fallen: bool = False,
    ) -> Tuple[float, float, float, float]:
        """One controller tick. Returns ``(vx, vy, yaw_rate, standing)``."""
        # --- Compute goal-relative geometry (only when we have a goal)
        if goal.xyz is not None:
            tx, ty, _tz = goal.xyz
            dist = forward_distance(robot_x, robot_y, tx, ty)
            err = heading_error(robot_x, robot_y, robot_yaw, tx, ty)
        else:
            dist = None
            err = None
        self.last_dist = dist
        self.last_heading_err = err

        # --- Mode transition
        new_mode = self._decide_mode(goal, dist, fallen)
        if new_mode != self.mode:
            old = self.mode
            self.mode = new_mode
            if new_mode != FSMMode.ARRIVED:
                self.arrived_printed = False
            if self.on_transition is not None:
                self.on_transition(
                    old,
                    new_mode,
                    {
                        "forward_dist": dist,
                        "heading_err_rad": err,
                        "heading_err_deg": (math.degrees(err) if err is not None else None),
                    },
                )

        # --- Actuation -> _cmd via the pure translator
        return fsm_mode_to_cmd(
            self.mode,
            heading_err_rad=(err if err is not None else 0.0),
            walk_speed=self.params.walk_speed,
            search_yaw=self.params.search_yaw,
            heading_kp=self.params.heading_kp,
            yaw_max=self.params.yaw_max,
            heading_walk_deg=self.params.heading_walk_deg,
            limits=self.params.limits,
        )

    # ---------------------------------------------------------- reset hooks
    def reset(self) -> None:
        """Called when switching target / re-arming after fall."""
        self.mode = FSMMode.SEARCH
        self.arrived_printed = False
        self.last_dist = None
        self.last_heading_err = None
