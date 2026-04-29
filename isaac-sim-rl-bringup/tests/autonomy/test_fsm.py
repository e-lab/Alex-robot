"""Tests for autonomy.fsm.FSMController.

Covers mode transitions (search/approach/arrived/fallen) and the cmd produced
in each state. Stays pure: no Isaac, no torch.
"""
from __future__ import annotations

import math

import pytest

from autonomy.fsm import FSMController, FSMMode, FSMParams
from autonomy.goal import GoalState
from autonomy.translator import GaitLimits


def _params(**overrides) -> FSMParams:
    base = dict(
        stop_dist=1.0,
        walk_speed=0.30,
        search_yaw=0.30,
        heading_kp=0.8,
        yaw_max=0.40,
        heading_walk_deg=30.0,
        limits=GaitLimits(),
    )
    base.update(overrides)
    return FSMParams(**base)


class TestFSMModeDecision:
    def test_no_goal_mode_search(self):
        fsm = FSMController(_params())
        cmd = fsm.step(0, 0, 0, goal=GoalState(), fallen=False)
        assert fsm.mode == FSMMode.SEARCH
        # Search yaws only
        assert cmd[0] == 0.0 and cmd[1] == 0.0
        assert cmd[2] == pytest.approx(0.30)

    def test_fallen_overrides_everything(self):
        fsm = FSMController(_params())
        goal = GoalState(); goal.set_fixed((3.0, 0.0, 0.0))
        cmd = fsm.step(0, 0, 0, goal=goal, fallen=True)
        assert fsm.mode == FSMMode.FALLEN
        assert cmd == (0.0, 0.0, 0.0, 1.0)

    def test_locked_goal_far_approach(self):
        fsm = FSMController(_params())
        goal = GoalState(); goal.set_fixed((3.0, 0.0, 0.0))
        cmd = fsm.step(0, 0, 0, goal=goal, fallen=False)
        assert fsm.mode == FSMMode.APPROACH
        # Robot at origin facing target -> heading_err=0 -> walk forward
        assert cmd[0] == pytest.approx(0.30)
        assert cmd[3] == 0.0

    def test_within_stop_dist_arrives(self):
        fsm = FSMController(_params(stop_dist=1.0))
        goal = GoalState(); goal.set_fixed((0.5, 0.0, 0.0))   # 0.5 m away < stop_dist
        cmd = fsm.step(0, 0, 0, goal=goal, fallen=False)
        assert fsm.mode == FSMMode.ARRIVED
        assert cmd == (0.0, 0.0, 0.0, 1.0)

    def test_unlocked_stale_goal_search(self):
        fsm = FSMController(_params(stale_s=1.0))
        goal = GoalState()
        goal.xyz = (3.0, 0.0, 0.0)
        goal.last_update_t = 0.0       # ancient
        goal.locked = False
        cmd = fsm.step(0, 0, 0, goal=goal, fallen=False)
        assert fsm.mode == FSMMode.SEARCH
        # Goal is_fresh sees current wall-clock time; at any sane runtime, the
        # goal's last_update_t=0 is way past stale_s=1.
        assert cmd[2] == pytest.approx(0.30)


class TestFSMTransitions:
    def test_transition_callback_fires(self):
        events = []
        fsm = FSMController(_params(), on_transition=lambda o, n, info: events.append((o, n)))
        goal = GoalState()
        # First tick -> SEARCH (no transition since initial mode=SEARCH already)
        fsm.step(0, 0, 0, goal=goal)
        assert events == []
        # Now lock a goal far away -> SEARCH -> APPROACH
        goal.set_fixed((3.0, 0.0, 0.0))
        fsm.step(0, 0, 0, goal=goal)
        assert events[-1] == (FSMMode.SEARCH, FSMMode.APPROACH)
        # Move close -> APPROACH -> ARRIVED
        fsm.step(2.5, 0, 0, goal=goal)        # dist=0.5 < stop_dist=1.0
        assert events[-1] == (FSMMode.APPROACH, FSMMode.ARRIVED)

    def test_reset_returns_to_search(self):
        fsm = FSMController(_params())
        goal = GoalState(); goal.set_fixed((3.0, 0.0, 0.0))
        fsm.step(0, 0, 0, goal=goal)
        assert fsm.mode == FSMMode.APPROACH
        fsm.reset()
        assert fsm.mode == FSMMode.SEARCH
        assert fsm.last_dist is None
        assert fsm.last_heading_err is None


class TestFSMHeading:
    def test_target_to_left_yaws_left(self):
        # Robot at origin facing +x; goal at (1, 1) -> heading +45° -> approach turns left
        fsm = FSMController(_params())
        goal = GoalState(); goal.set_fixed((1.0, 1.0, 0.0))
        # dist = sqrt(2) > stop_dist, so APPROACH
        cmd = fsm.step(0, 0, 0, goal=goal)
        assert fsm.mode == FSMMode.APPROACH
        assert cmd[2] > 0.0   # positive yaw_rate

    def test_target_behind_does_not_walk_forward(self):
        # Robot facing +x, goal at (-3, 0): heading_err = ±π → walk_forward suppressed
        fsm = FSMController(_params())
        goal = GoalState(); goal.set_fixed((-3.0, 0.0, 0.0))
        cmd = fsm.step(0, 0, 0, goal=goal)
        assert fsm.mode == FSMMode.APPROACH
        assert cmd[0] == 0.0
        # yaw_rate at the soft cap (yaw_max=0.4)
        assert abs(cmd[2]) == pytest.approx(0.4)

    def test_last_dist_and_heading_recorded(self):
        fsm = FSMController(_params())
        goal = GoalState(); goal.set_fixed((4.0, 3.0, 0.0))
        fsm.step(0.0, 0.0, 0.0, goal=goal)
        assert fsm.last_dist == pytest.approx(5.0)
        assert fsm.last_heading_err == pytest.approx(math.atan2(3.0, 4.0))
