"""Tests for the Phase 4 recovery agent + stuck monitor + yaw tracker.

Pure-logic — no Isaac, no torch, no time.sleep. Wall-clock is injected
via the optional ``now`` parameter on every method (mirrors
``GoalState.is_fresh(now=...)``). Tests run in milliseconds.
"""
from __future__ import annotations

import math

import pytest

from autonomy.pose import FallMonitor
from autonomy.recovery import (
    RecoveryAgent,
    RecoveryState,
    StuckMonitor,
    YawTracker,
)


# ── RecoveryAgent: state machine + cmd output ────────────────────────────────
class TestRecoveryAgent:
    def test_initial_state_idle(self):
        a = RecoveryAgent()
        assert a.state is RecoveryState.IDLE
        assert a.attempts_used == 0

    def test_begin_standing_transitions_to_standing(self):
        a = RecoveryAgent()
        a.begin_standing(now=100.0)
        assert a.state is RecoveryState.STANDING

    def test_standing_done_at_stand_duration(self):
        a = RecoveryAgent(stand_duration_s=3.0)
        a.begin_standing(now=100.0)
        assert a.is_standing_done(now=102.9) is False
        assert a.is_standing_done(now=103.0) is True
        assert a.is_standing_done(now=103.5) is True

    def test_is_standing_done_false_when_not_standing(self):
        a = RecoveryAgent()
        # IDLE → never "done" because we never started.
        assert a.is_standing_done(now=999.0) is False

    def test_standing_cmd_is_zero_with_standing_flag(self):
        a = RecoveryAgent()
        a.begin_standing(now=0.0)
        assert a.cmd() == (0.0, 0.0, 0.0, 1.0)

    def test_succeed_clears_attempts_back_to_idle(self):
        a = RecoveryAgent()
        a.begin_standing(now=0.0)
        a.attempts_used = 1
        a.succeed()
        assert a.state is RecoveryState.IDLE
        assert a.attempts_used == 0

    def test_fail_attempt_increments_and_returns_to_idle(self):
        a = RecoveryAgent(max_attempts=2)
        a.begin_standing(now=0.0)
        a.fail_attempt()
        assert a.state is RecoveryState.IDLE
        assert a.attempts_used == 1

    def test_fail_attempt_after_max_transitions_to_failed(self):
        a = RecoveryAgent(max_attempts=2)
        a.begin_standing(now=0.0)
        a.fail_attempt()
        a.begin_standing(now=10.0)
        a.fail_attempt()
        assert a.state is RecoveryState.FAILED
        assert a.attempts_used == 2

    def test_failed_cmd_is_safe_stop(self):
        a = RecoveryAgent(max_attempts=1)
        a.begin_standing(now=0.0)
        a.fail_attempt()
        assert a.state is RecoveryState.FAILED
        assert a.cmd() == (0.0, 0.0, 0.0, 1.0)

    def test_idle_cmd_is_safe_stop(self):
        # Defensive: if cmd() is ever called from IDLE, return a safe stop.
        a = RecoveryAgent()
        assert a.cmd() == (0.0, 0.0, 0.0, 1.0)

    def test_begin_rotation_from_idle(self):
        a = RecoveryAgent()
        a.begin_rotation()
        assert a.state is RecoveryState.ROTATING

    def test_rotation_does_not_consume_fall_attempt(self):
        a = RecoveryAgent(max_attempts=2)
        a.begin_rotation()
        a.succeed()
        assert a.attempts_used == 0

    def test_rotation_cmd_yaws_only(self):
        a = RecoveryAgent(rotation_yaw_rate=0.4)
        a.begin_rotation()
        cmd = a.cmd(rotation_sign=+1.0)
        assert cmd[0] == 0.0
        assert cmd[1] == 0.0
        assert cmd[2] == pytest.approx(+0.4)
        assert cmd[3] == 0.0

    def test_rotation_cmd_negative_sign_yaws_other_way(self):
        a = RecoveryAgent(rotation_yaw_rate=0.4)
        a.begin_rotation()
        cmd = a.cmd(rotation_sign=-1.0)
        assert cmd[2] == pytest.approx(-0.4)


# ── YawTracker: closed-loop 90° rotation ─────────────────────────────────────
class TestYawTracker:
    def test_initial_delta_is_zero(self):
        t = YawTracker()
        t.start(yaw_now=0.0)
        assert t.delta == pytest.approx(0.0)

    def test_delta_accumulates_through_pi(self):
        t = YawTracker(target_rad=math.pi)
        t.start(yaw_now=0.0)
        # Rotate from 0 → π/3 → 2π/3 → π. delta should track the running sum.
        assert t.update(yaw_now=math.pi / 3) is False
        assert t.delta == pytest.approx(math.pi / 3, abs=1e-9)
        t.update(yaw_now=2 * math.pi / 3)
        assert t.delta == pytest.approx(2 * math.pi / 3, abs=1e-9)
        # Reaching exactly π should fire done (|delta| >= |target|).
        done = t.update(yaw_now=math.pi)
        assert done is True

    def test_delta_handles_wrap_around_positive(self):
        """Rotate +90° starting from yaw=+170° → wraps through ±180° to +260° (-100°)."""
        t = YawTracker(target_rad=math.pi / 2)
        # yaw0 = 170° = +2.967 rad
        t.start(yaw_now=math.radians(170.0))
        # Now rotate +30°. yaw_now would naturally wrap: 200° → -160° = -2.793 rad.
        # Tracker should see this as +30° not -330°.
        t.update(yaw_now=math.radians(-160.0))   # i.e. 200° wrapped
        assert t.delta == pytest.approx(math.radians(30.0), abs=1e-3)

    def test_delta_handles_wrap_around_negative(self):
        """Rotate -90° starting from yaw=-170° → wraps to -260° (+100°)."""
        t = YawTracker(target_rad=-math.pi / 2)
        t.start(yaw_now=math.radians(-170.0))
        # Rotate -30°: yaw_now goes -200° → +160°.
        t.update(yaw_now=math.radians(160.0))
        assert t.delta == pytest.approx(math.radians(-30.0), abs=1e-3)

    def test_done_at_quarter_turn(self):
        t = YawTracker(target_rad=math.pi / 2)
        t.start(yaw_now=0.0)
        # Just shy of 90°: not done.
        assert t.update(yaw_now=math.pi / 2 - 0.01) is False
        # At exactly 90°: done.
        assert t.update(yaw_now=math.pi / 2) is True

    def test_negative_target_supported(self):
        t = YawTracker(target_rad=-math.pi / 2)
        t.start(yaw_now=0.0)
        # Rotate the wrong way → not done.
        assert t.update(yaw_now=+0.5) is False
        # Now rotate -π/2: |delta| = π/2 ≥ π/2 → done.
        # Need to drive yaw down past zero to negative.
        t.update(yaw_now=0.0)         # back to zero
        t.update(yaw_now=-math.pi / 4)
        assert t.update(yaw_now=-math.pi / 2) is True


# ── StuckMonitor: rolling-window stuck detection ─────────────────────────────
class TestStuckMonitor:
    def test_initial_not_stuck(self):
        m = StuckMonitor()
        assert m.stuck is False

    def test_idle_when_inactive_doesnt_latch(self):
        """Even with no displacement, inactive samples don't accumulate."""
        m = StuckMonitor(window_s=5.0, min_disp_m=0.2)
        for k in range(20):
            m.update(0.0, 0.0, active=False, now=float(k) * 0.5)
        assert m.stuck is False

    def test_active_with_motion_doesnt_latch(self):
        """Robot moves 0.5m per sample: clearly making progress."""
        m = StuckMonitor(window_s=5.0, min_disp_m=0.2)
        for k in range(15):
            m.update(0.5 * k, 0.0, active=True, now=float(k) * 0.5)
        assert m.stuck is False

    def test_active_no_motion_latches_after_window(self):
        """Stationary for ≥ window → stuck latches True."""
        m = StuckMonitor(window_s=5.0, min_disp_m=0.2)
        for k in range(15):
            now = float(k) * 0.5     # 0, 0.5, 1.0, ..., 7.0 (spans ≥ 5 s)
            m.update(1.0, 2.0, active=True, now=now)
        assert m.stuck is True

    def test_threshold_is_strict_less_than(self):
        """Total displacement of exactly min_disp_m is NOT stuck."""
        m = StuckMonitor(window_s=5.0, min_disp_m=0.2)
        # Sample 0: at (0,0) at t=0.
        # Sample N: at (0.2, 0) at t=6.
        # Spans > window with displacement == 0.2 → not stuck (strict).
        m.update(0.0, 0.0, active=True, now=0.0)
        m.update(0.05, 0.0, active=True, now=2.0)
        m.update(0.10, 0.0, active=True, now=4.0)
        m.update(0.20, 0.0, active=True, now=6.0)
        assert m.stuck is False

    def test_inactive_clears_buffer(self):
        """An active=False sample mid-window resets the rolling buffer."""
        m = StuckMonitor(window_s=5.0, min_disp_m=0.2)
        # Half a window of stationary active samples.
        m.update(0.0, 0.0, active=True, now=0.0)
        m.update(0.0, 0.0, active=True, now=2.0)
        # Inactive sample drops buffer.
        m.update(0.0, 0.0, active=False, now=3.0)
        # Single stationary active sample again — too short to latch.
        m.update(0.0, 0.0, active=True, now=4.0)
        assert m.stuck is False
        # Need a fresh full window of stationary active samples to latch.
        for now in [4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
            m.update(0.0, 0.0, active=True, now=now)
        assert m.stuck is True

    def test_reset_clears_latch_and_buffer(self):
        m = StuckMonitor(window_s=5.0, min_disp_m=0.2)
        for k in range(15):
            m.update(0.0, 0.0, active=True, now=float(k) * 0.5)
        assert m.stuck is True
        m.reset()
        assert m.stuck is False
        # And immediately reading a stationary sample doesn't re-latch:
        # the buffer is empty so we don't span a window.
        m.update(0.0, 0.0, active=True, now=10.0)
        assert m.stuck is False

    def test_window_uses_injected_now(self):
        """No call to time.time(): everything uses the injected now."""
        m = StuckMonitor(window_s=5.0, min_disp_m=0.2)
        m.update(0.0, 0.0, active=True, now=1000.0)
        m.update(0.0, 0.0, active=True, now=1006.0)
        # Two samples spanning > window with no motion → stuck.
        assert m.stuck is True


# ── Integration: agent + monitor + FallMonitor working together ──────────────
class TestRecoveryIntegrationScenarios:
    def test_fall_then_successful_recovery(self):
        """A fall latches, agent stands for 3s, robot is upright at end →
        succeed, attempts reset to 0."""
        fall = FallMonitor(fall_height_m=0.5, fall_tilt_norm=0.7)
        agent = RecoveryAgent(stand_duration_s=3.0, max_attempts=2)

        # Tick 0: robot falls.
        assert fall.update(root_z=0.20, proj_grav_xy=(0.0, 0.0)) is True
        agent.begin_standing(now=100.0)

        # 3 seconds elapse...
        assert agent.is_standing_done(now=103.0) is True

        # Recovery check: reset latch, re-test pose.
        fall.reset()
        re_fallen = fall.update(root_z=0.93, proj_grav_xy=(0.0, 0.0))
        assert re_fallen is False

        agent.succeed()
        assert agent.state is RecoveryState.IDLE
        assert agent.attempts_used == 0

    def test_fall_repeated_failure_eventually_aborts(self):
        """Two stand windows both end with the robot still fallen → FAILED."""
        fall = FallMonitor(fall_height_m=0.5, fall_tilt_norm=0.7)
        agent = RecoveryAgent(stand_duration_s=3.0, max_attempts=2)

        # First fall + first attempt.
        fall.update(root_z=0.20)
        agent.begin_standing(now=0.0)
        assert agent.is_standing_done(now=3.0) is True
        fall.reset()
        re_fallen = fall.update(root_z=0.20)        # still on the ground
        assert re_fallen is True
        agent.fail_attempt()
        assert agent.state is RecoveryState.IDLE
        assert agent.attempts_used == 1

        # Second attempt.
        agent.begin_standing(now=10.0)
        assert agent.is_standing_done(now=13.0) is True
        fall.reset()
        re_fallen = fall.update(root_z=0.20)
        assert re_fallen is True
        agent.fail_attempt()
        assert agent.state is RecoveryState.FAILED
        assert agent.cmd() == (0.0, 0.0, 0.0, 1.0)

    def test_stuck_then_rotation_then_clear(self):
        """Stuck latches → caller starts rotation → yaw tracker reaches π/2 →
        succeed → subsequent moving samples don't re-latch."""
        stuck = StuckMonitor(window_s=5.0, min_disp_m=0.2)
        agent = RecoveryAgent()
        yaw = YawTracker(target_rad=math.pi / 2)

        # Stuck: feed stationary samples for > 5s.
        for k in range(15):
            stuck.update(2.0, 3.0, active=True, now=float(k) * 0.5)
        assert stuck.stuck is True

        # Begin rotation.
        agent.begin_rotation()
        yaw.start(yaw_now=0.0)
        assert agent.state is RecoveryState.ROTATING
        assert yaw.update(yaw_now=math.pi / 4) is False

        # Reach target.
        assert yaw.update(yaw_now=math.pi / 2) is True

        # Caller cleans up.
        agent.succeed()
        stuck.reset()
        assert agent.state is RecoveryState.IDLE
        assert stuck.stuck is False

        # Subsequent moving samples don't re-latch.
        for k in range(15):
            stuck.update(2.0 + 0.5 * k, 3.0, active=True,
                         now=20.0 + float(k) * 0.5)
        assert stuck.stuck is False
