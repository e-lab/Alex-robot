"""Goal state container.

Phase 1: ``fixed_xyz`` source — goal set once at startup, never changes.
Phase 2: ``approach`` source — goal updated from SAM3-detected ObjectNodes
via :meth:`GoalState.update_from_object`. Lock-on triggers when the
detection's confidence reaches ``lock_conf``; once locked, further updates
are ignored (the goal latches even if SAM3 stops seeing the target — fixes
the cam-script's mid-approach oscillation when the object fills view and
mask quality drops).

The FSM consumes only ``GoalState`` — it never knows whether the goal came
from a config constant or live perception.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class GoalState:
    """Latest known target XYZ + freshness metadata.

    Attributes
    ----------
    xyz
        World-frame (x, y, z) of the goal, or None if unknown.
    last_update_t
        ``time.time()`` of the last update. Used by Phase 2 to detect staleness.
    score
        Detection confidence in [0, 1] (Phase 2). 1.0 for ``fixed_xyz`` mode.
    locked
        True once a high-confidence detection latches the goal (Phase 2).
        Always True for ``fixed_xyz`` mode (it never goes stale).
    """

    xyz: Optional[Tuple[float, float, float]] = None
    last_update_t: float = 0.0
    score: float = 0.0
    locked: bool = False

    def set_fixed(self, xyz: Tuple[float, float, float]) -> None:
        """Phase-1 helper: set the goal once and treat it as locked-forever."""
        self.xyz = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        self.last_update_t = time.time()
        self.score = 1.0
        self.locked = True

    def clear(self) -> None:
        self.xyz = None
        self.last_update_t = 0.0
        self.score = 0.0
        self.locked = False

    def is_fresh(self, stale_s: float, now: Optional[float] = None) -> bool:
        """Goal is considered actionable if locked, OR seen within stale_s."""
        if self.xyz is None:
            return False
        if self.locked:
            return True
        now = time.time() if now is None else now
        return (now - self.last_update_t) < stale_s

    def update_from_object(self, obj, *, lock_conf: float = 0.6) -> None:
        """Update goal state from a SceneGraph ObjectNode (Phase 2).

        Once the goal is locked (``self.locked == True``), subsequent updates
        are silently ignored — the goal latches even if a different (or
        higher-scoring) object appears later in the same label class. To
        retarget, call :meth:`clear` first.

        ``obj`` is duck-typed: must have ``.position_xyz`` (length-3 sequence)
        and ``.confidence`` (float). The vendored
        ``scene_graph.graph.node_types.ObjectNode`` satisfies this.
        """
        if self.locked:
            return
        x, y, z = obj.position_xyz[0], obj.position_xyz[1], obj.position_xyz[2]
        self.xyz = (float(x), float(y), float(z))
        self.score = float(obj.confidence)
        self.last_update_t = time.time()
        self.locked = self.score >= lock_conf
