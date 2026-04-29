"""Tests for autonomy.pose (yaw_from_quat, FallMonitor)."""
from __future__ import annotations

import math

import pytest

from autonomy.pose import FallMonitor, yaw_from_quat


# ── yaw_from_quat ────────────────────────────────────────────────────────────
class TestYawFromQuat:
    def test_identity_is_zero_yaw(self):
        # Scalar-first: (w=1, x=0, y=0, z=0) is identity
        assert yaw_from_quat((1.0, 0.0, 0.0, 0.0)) == pytest.approx(0.0)

    def test_90_deg_yaw(self):
        # Yaw +90° about Z: (w, 0, 0, sin(45°))
        c = math.cos(math.pi / 4)
        s = math.sin(math.pi / 4)
        assert yaw_from_quat((c, 0.0, 0.0, s)) == pytest.approx(math.pi / 2, rel=1e-6)

    def test_180_deg_yaw_wraps_into_range(self):
        # Yaw 180° -> atan2 returns +pi (or -pi); both are valid wraps
        result = yaw_from_quat((0.0, 0.0, 0.0, 1.0))
        assert math.isclose(abs(result), math.pi, rel_tol=1e-6)

    def test_negative_yaw(self):
        c = math.cos(-math.pi / 6)
        s = math.sin(-math.pi / 6)
        result = yaw_from_quat((c, 0.0, 0.0, s))
        assert result == pytest.approx(-math.pi / 3, rel=1e-6)

    def test_pitched_quat_still_extracts_yaw(self):
        # Combined yaw 30° about Z * pitch 15° about Y. Yaw output should still be ~30°.
        yaw, pitch = math.radians(30), math.radians(15)
        # quat_yaw  = (cos(yaw/2), 0, 0, sin(yaw/2))
        # quat_pitch = (cos(pitch/2), 0, sin(pitch/2), 0)
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        # q_total = q_yaw * q_pitch (Hamilton product, scalar-first)
        w = cy * cp
        x = cy * sp * 0 + sy * 0 - sy * sp * 0  # simplified branches that involve x-component zero terms
        # Easier: use a direct multiplication for (w1,x1,y1,z1) * (w2,x2,y2,z2)
        w1, x1, y1, z1 = cy, 0.0, 0.0, sy
        w2, x2, y2, z2 = cp, 0.0, sp, 0.0
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        # Allow small error since pitch couples slightly into the yaw extraction at large angles.
        assert yaw_from_quat((w, x, y, z)) == pytest.approx(yaw, abs=1e-3)


# ── FallMonitor ──────────────────────────────────────────────────────────────
class TestFallMonitor:
    def test_upright_robot_not_fallen(self):
        fm = FallMonitor()
        assert fm.update(root_z=0.93, proj_grav_xy=(0.01, -0.02)) is False
        assert fm.fallen is False

    def test_low_height_triggers_fall(self):
        fm = FallMonitor(fall_height_m=0.5)
        assert fm.update(root_z=0.30, proj_grav_xy=(0.0, 0.0)) is True
        assert fm.fallen is True

    def test_high_tilt_triggers_fall(self):
        fm = FallMonitor(fall_tilt_norm=0.7)
        # Tilt magnitude = hypot(0.6, 0.5) ≈ 0.78 > 0.7
        assert fm.update(root_z=0.93, proj_grav_xy=(0.6, 0.5)) is True

    def test_tilt_under_threshold_not_fallen(self):
        fm = FallMonitor(fall_tilt_norm=0.7)
        # hypot(0.4, 0.3) = 0.5 < 0.7
        assert fm.update(root_z=0.93, proj_grav_xy=(0.4, 0.3)) is False

    def test_fall_latches(self):
        fm = FallMonitor(fall_height_m=0.5)
        fm.update(root_z=0.30)            # fall
        # Even if robot magically goes back up, the monitor stays latched
        assert fm.update(root_z=1.50) is True

    def test_reset_clears_latch(self):
        fm = FallMonitor(fall_height_m=0.5)
        fm.update(root_z=0.30)
        assert fm.fallen is True
        fm.reset()
        assert fm.fallen is False
        assert fm.update(root_z=0.93) is False

    def test_proj_grav_optional(self):
        fm = FallMonitor()
        # Passing None for proj_grav_xy should not crash.
        assert fm.update(root_z=0.93, proj_grav_xy=None) is False
