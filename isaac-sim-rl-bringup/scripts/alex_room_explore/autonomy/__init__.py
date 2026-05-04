"""Autonomy package for the Alex ONNX walking policy.

Pure-logic modules — and one Isaac adapter (``perception.py``).

Phase 1 (PLAN/autonomous_navigation_plan.md):
- pose:          yaw_from_quat, FallMonitor
- translator:    fsm_mode_to_cmd (vx, vy, yaw_rate, standing) clamped to gait limits
- fsm:           FSMController   (search / approach / arrived / fallen)
- goal:          GoalState       (fixed_xyz preset)

Phase 2:
- target_picker: pick_goal_for_target  (label + lock_conf + min_observations)
- goal:          + GoalState.update_from_object  (latches once score >= lock_conf)
- perception:    get_head_cam_pose_K, read_rgb_depth   (Isaac-coupled adapter)

Phase 3.5 (active):
- obstacle:      forward_cone_distance — used by the emergency brake only.
                 The deliberative planner (USD-derived occupancy + A*) is the
                 primary path-around-obstacle mechanism. Reactive cone
                 steering was tried in Phase 3 and failed against wall-
                 shaped obstacles; see docs/phase3_retrospective.md.

Phase 4 (active):
- recovery:      RecoveryAgent (IDLE / STANDING / ROTATING / FAILED),
                 StuckMonitor (rolling-window stall detector),
                 YawTracker (closed-loop 90° rotation).

Scene-graph machinery (vendored from sravani-scenegraph-demo) lives in the
top-level ``scene_graph/`` package; we don't re-export it here.
"""
from .pose import yaw_from_quat, FallMonitor
from .translator import fsm_mode_to_cmd, GaitLimits, wrap_to_pi, heading_error
from .fsm import FSMController, FSMMode
from .goal import GoalState
from .target_picker import pick_goal_for_target
from .obstacle import forward_cone_distance
from .recovery import RecoveryAgent, RecoveryState, StuckMonitor, YawTracker

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
    "pick_goal_for_target",
    "forward_cone_distance",
    "RecoveryAgent",
    "RecoveryState",
    "StuckMonitor",
    "YawTracker",
]
