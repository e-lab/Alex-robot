"""Tests for autonomy.goal.GoalState.update_from_object (Phase 2)."""
from __future__ import annotations

import pytest

from autonomy.goal import GoalState
from scene_graph.graph.node_types import ObjectNode


def _obj(label="oven", *, confidence=0.8, xyz=(2.0, 1.0, 0.7), n_obs=5):
    return ObjectNode(
        id=f"{label}_1",
        label=label,
        position_xyz=list(xyz),
        bbox_min_xyz=[xyz[0] - 0.1, xyz[1] - 0.1, xyz[2] - 0.1],
        bbox_max_xyz=[xyz[0] + 0.1, xyz[1] + 0.1, xyz[2] + 0.1],
        confidence=confidence,
        n_observations=n_obs,
    )


class TestUpdateFromObject:
    def test_writes_xyz_score_timestamp(self):
        g = GoalState()
        g.update_from_object(_obj(xyz=(3.0, -1.5, 0.5), confidence=0.55))
        assert g.xyz == (3.0, -1.5, 0.5)
        assert g.score == pytest.approx(0.55)
        assert g.last_update_t > 0
        # 0.55 < default lock_conf=0.6 -> not yet locked
        assert g.locked is False

    def test_locks_when_score_meets_threshold(self):
        g = GoalState()
        g.update_from_object(_obj(confidence=0.6))
        assert g.locked is True

    def test_locks_above_threshold(self):
        g = GoalState()
        g.update_from_object(_obj(confidence=0.95))
        assert g.locked is True

    def test_locked_goal_ignores_subsequent_updates(self):
        g = GoalState()
        # First update: locks at score 0.9, xyz (1,0,0)
        g.update_from_object(_obj(confidence=0.9, xyz=(1.0, 0.0, 0.0)))
        assert g.locked is True
        original_t = g.last_update_t
        # Second update with different xyz + higher confidence is ignored.
        g.update_from_object(_obj(confidence=0.99, xyz=(5.0, 5.0, 0.0)))
        assert g.xyz == (1.0, 0.0, 0.0)        # unchanged
        assert g.score == pytest.approx(0.9)    # unchanged
        assert g.last_update_t == original_t   # timestamp not bumped

    def test_unlocked_goal_can_be_re_updated(self):
        g = GoalState()
        # Below lock threshold — goal updates but stays unlocked
        g.update_from_object(_obj(confidence=0.4, xyz=(1.0, 0.0, 0.0)))
        assert g.xyz == (1.0, 0.0, 0.0)
        assert g.locked is False
        # Subsequent better detection can move + lock the goal
        g.update_from_object(_obj(confidence=0.8, xyz=(2.0, 2.0, 0.0)))
        assert g.xyz == (2.0, 2.0, 0.0)
        assert g.locked is True

    def test_clear_unlocks_and_clears(self):
        g = GoalState()
        g.update_from_object(_obj(confidence=0.9))
        assert g.locked is True
        g.clear()
        assert g.locked is False
        assert g.xyz is None
        # After clear, update_from_object works again
        g.update_from_object(_obj(confidence=0.9, xyz=(3.0, 3.0, 0.0)))
        assert g.xyz == (3.0, 3.0, 0.0)
        assert g.locked is True

    def test_custom_lock_conf(self):
        g = GoalState()
        # Default 0.6 would reject, but caller can pass a tighter threshold
        g.update_from_object(_obj(confidence=0.7), lock_conf=0.8)
        assert g.locked is False
        g.update_from_object(_obj(confidence=0.85), lock_conf=0.8)
        assert g.locked is True

    def test_update_from_object_coerces_to_floats(self):
        g = GoalState()
        # ObjectNode may carry numpy scalars or ints; xyz should be plain floats.
        import numpy as np
        obj = _obj(xyz=(np.float32(1.0), np.int64(2), 3))
        g.update_from_object(obj)
        assert g.xyz == (1.0, 2.0, 3.0)
        assert all(isinstance(v, float) for v in g.xyz)
