"""Phase-1 autonomy package for the Alex ONNX walking policy.

Pure-logic modules only (no Isaac imports, no rerun): keeps the FSM and the
_cmd translator unit-testable without spinning up the simulator.

Phase 1 scope (per PLAN/autonomous_navigation_plan.md):
- pose:        yaw_from_quat, FallMonitor
- translator:  fsm_mode_to_cmd  (vx, vy, yaw_rate, standing) clamped to gait limits
- fsm:         FSMController    (search / approach / arrived / fallen)
- goal:        GoalState        (fixed_xyz mode; Phase 2 will extend with SAM3 lock/stale)
"""
from .pose import yaw_from_quat, FallMonitor
from .translator import fsm_mode_to_cmd, GaitLimits, wrap_to_pi, heading_error
from .fsm import FSMController, FSMMode
from .goal import GoalState

__all__ = [
    "yaw_from_quat",
    "FallMonitor",
    "fsm_mode_to_cmd",
    "GaitLimits",
    "wrap_to_pi",
    "heading_error",
    "FSMController",
    "FSMMode",
    "GoalState",
]
