"""Tests for autonomy.goal.GoalState (Phase 1: fixed_xyz path only)."""
from __future__ import annotations

import time

from autonomy.goal import GoalState


class TestGoalState:
    def test_default_no_goal(self):
        g = GoalState()
        assert g.xyz is None
        assert g.locked is False
        assert g.score == 0.0
        assert g.is_fresh(stale_s=5.0) is False

    def test_set_fixed_locks_and_is_fresh(self):
        g = GoalState()
        g.set_fixed((3.0, 0.5, 0.0))
        assert g.xyz == (3.0, 0.5, 0.0)
        assert g.locked is True
        assert g.score == 1.0
        # Locked goals never go stale.
        assert g.is_fresh(stale_s=0.001) is True
        assert g.is_fresh(stale_s=5.0, now=time.time() + 1e6) is True

    def test_unlocked_goal_freshness(self):
        g = GoalState()
        g.xyz = (1.0, 2.0, 0.0)
        g.last_update_t = 100.0
        g.locked = False
        # now = 102.0, stale_s = 5 -> fresh
        assert g.is_fresh(stale_s=5.0, now=102.0) is True
        # now = 110.0, stale_s = 5 -> stale
        assert g.is_fresh(stale_s=5.0, now=110.0) is False

    def test_clear(self):
        g = GoalState()
        g.set_fixed((1.0, 2.0, 3.0))
        g.clear()
        assert g.xyz is None
        assert g.locked is False
        assert g.score == 0.0
        assert g.last_update_t == 0.0

    def test_set_fixed_coerces_to_floats(self):
        g = GoalState()
        g.set_fixed((1, 2, 3))   # ints
        assert g.xyz == (1.0, 2.0, 3.0)
        assert all(isinstance(v, float) for v in g.xyz)
