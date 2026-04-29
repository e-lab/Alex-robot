"""Goal state container.

Phase 1: only the ``fixed_xyz`` source is implemented — the goal is set once at
startup and never changes.

Phase 2 will extend this class with: SAM3 detection updates, score, freshness
(stale_s), and lock-on (score >= lock_conf). Keeping the surface stable now so
the FSM doesn't have to change in Phase 2.
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
