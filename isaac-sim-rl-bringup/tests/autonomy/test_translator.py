"""Tests for autonomy.translator (heading math, gait limits, FSM-mode → cmd)."""
from __future__ import annotations

import math

import pytest

from autonomy.translator import (
    GaitLimits,
    forward_distance,
    fsm_mode_to_cmd,
    heading_error,
    wrap_to_pi,
)


# ── wrap_to_pi ───────────────────────────────────────────────────────────────
class TestWrapToPi:
    @pytest.mark.parametrize("inp,expected", [
        (0.0, 0.0),
        (math.pi, -math.pi),                  # +pi wraps to -pi by this convention
        (-math.pi, -math.pi),
        (math.pi + 0.1, -math.pi + 0.1),
        (3 * math.pi, -math.pi),              # +3pi wraps to -pi
        (-3 * math.pi, -math.pi),
        (math.pi / 4, math.pi / 4),
    ])
    def test_wraps(self, inp, expected):
        assert wrap_to_pi(inp) == pytest.approx(expected, abs=1e-9)


# ── heading_error ────────────────────────────────────────────────────────────
class TestHeadingError:
    def test_target_in_front_facing_target_zero_err(self):
        # Robot at origin looking +x, goal at (1, 0): err = 0
        assert heading_error(0, 0, 0.0, 1.0, 0.0) == pytest.approx(0.0)

    def test_target_to_left_yaw_zero_positive_err(self):
        # Robot at origin facing +x; goal at (0, 1) is +90° away
        assert heading_error(0, 0, 0.0, 0.0, 1.0) == pytest.approx(math.pi / 2)

    def test_target_behind_180_err_wraps(self):
        # Robot facing +x, goal at (-1, 0): err is +pi or -pi (both valid)
        err = heading_error(0, 0, 0.0, -1.0, 0.0)
        assert math.isclose(abs(err), math.pi, rel_tol=1e-6)

    def test_robot_already_yawed(self):
        # Robot at origin yawed +90°, goal at (0, 1): now exactly in front -> err 0
        assert heading_error(0, 0, math.pi / 2, 0.0, 1.0) == pytest.approx(0.0, abs=1e-9)

    def test_wrap_property(self):
        # Whatever yaw and target, returned err is always in [-pi, pi]
        err = heading_error(0, 0, 5.0 * math.pi, 0.3, -0.2)
        assert -math.pi - 1e-9 <= err <= math.pi + 1e-9


# ── forward_distance ─────────────────────────────────────────────────────────
class TestForwardDistance:
    def test_basic(self):
        assert forward_distance(0, 0, 3, 4) == pytest.approx(5.0)

    def test_zero(self):
        assert forward_distance(2, 2, 2, 2) == pytest.approx(0.0)


# ── GaitLimits ───────────────────────────────────────────────────────────────
class TestGaitLimits:
    def test_within_limits_unchanged(self):
        L = GaitLimits()
        assert L.clamp(0.2, 0.1, 0.3) == (0.2, 0.1, 0.3)

    def test_clamp_high(self):
        L = GaitLimits()
        # vx > 0.4, vy > 0.3, yaw_rate > 0.4 — all should clamp
        assert L.clamp(1.0, 1.0, 1.0) == (0.4, 0.3, 0.4)

    def test_clamp_low(self):
        L = GaitLimits()
        assert L.clamp(-1.0, -1.0, -1.0) == (-0.4, -0.3, -0.4)

    def test_custom_limits(self):
        L = GaitLimits(vx_max=0.2, vy_max=0.1, yaw_rate_max=0.05)
        assert L.clamp(0.5, 0.2, 0.1) == (0.2, 0.1, 0.05)


# ── fsm_mode_to_cmd ──────────────────────────────────────────────────────────
class TestModeToCmd:
    # ----- SEARCH
    def test_search_yaws_only(self):
        vx, vy, yawr, st = fsm_mode_to_cmd("search", search_yaw=0.30)
        assert (vx, vy, st) == (0.0, 0.0, 0.0)
        assert yawr == pytest.approx(0.30)

    def test_search_yaw_clamped_to_gait_limit(self):
        # search_yaw=0.9 exceeds default yaw_rate_max=0.4
        vx, vy, yawr, st = fsm_mode_to_cmd("search", search_yaw=0.9)
        assert yawr == pytest.approx(0.4)
        assert (vx, vy, st) == (0.0, 0.0, 0.0)

    def test_search_negative_yaw_preserved(self):
        _, _, yawr, _ = fsm_mode_to_cmd("search", search_yaw=-0.30)
        assert yawr == pytest.approx(-0.30)

    # ----- APPROACH
    def test_approach_facing_target_walks_forward(self):
        vx, vy, yawr, st = fsm_mode_to_cmd(
            "approach", heading_err_rad=0.0, walk_speed=0.30,
            heading_kp=0.8, yaw_max=0.4,
        )
        assert vx == pytest.approx(0.30)
        assert vy == 0.0
        assert yawr == pytest.approx(0.0)
        assert st == 0.0

    def test_approach_high_heading_err_does_not_walk_forward(self):
        # |err| = 60° > heading_walk_deg=30 -> vx=0
        vx, _, yawr, _ = fsm_mode_to_cmd(
            "approach", heading_err_rad=math.radians(60),
            walk_speed=0.30, heading_kp=0.8, yaw_max=0.4,
            heading_walk_deg=30.0,
        )
        assert vx == 0.0
        # yaw_rate = 0.8 * 60deg = 0.838 rad/s, soft-cap yaw_max=0.4
        assert yawr == pytest.approx(0.4)

    def test_approach_yaw_kp_applies_then_yaw_max_caps(self):
        # err=0.1 rad, kp=0.8 -> yaw_rate = 0.08, well under yaw_max=0.4
        vx, _, yawr, _ = fsm_mode_to_cmd(
            "approach", heading_err_rad=0.1, walk_speed=0.30,
            heading_kp=0.8, yaw_max=0.4,
        )
        assert yawr == pytest.approx(0.08)
        assert vx == pytest.approx(0.30)   # |err|=0.1 rad ~ 5.7° < 30°

    def test_approach_negative_heading_err_negative_yaw(self):
        _, _, yawr, _ = fsm_mode_to_cmd(
            "approach", heading_err_rad=-0.5, walk_speed=0.30,
            heading_kp=0.8, yaw_max=0.4,
        )
        # 0.8 * -0.5 = -0.4, exactly the cap
        assert yawr == pytest.approx(-0.4)

    def test_approach_walk_speed_clamped_by_gait_limit(self):
        # walk_speed=1.0 > vx_max=0.4 -> clamped to 0.4
        vx, _, _, _ = fsm_mode_to_cmd("approach", heading_err_rad=0.0, walk_speed=1.0)
        assert vx == pytest.approx(0.4)

    # ----- ARRIVED / FALLEN
    def test_arrived_zero_cmd_standing_on(self):
        assert fsm_mode_to_cmd("arrived") == (0.0, 0.0, 0.0, 1.0)

    def test_fallen_zero_cmd_standing_on(self):
        assert fsm_mode_to_cmd("fallen") == (0.0, 0.0, 0.0, 1.0)

    # ----- Unknown mode
    def test_unknown_mode_safe_stop(self):
        assert fsm_mode_to_cmd("garbage") == (0.0, 0.0, 0.0, 1.0)
