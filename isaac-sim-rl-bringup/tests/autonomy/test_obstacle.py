"""Tests for the forward-cone obstacle distance estimator.

The function takes a depth image + intrinsics and returns the median of the
lowest 20% of depth values inside a central cone (±h_deg, ±v_deg). It is the
single pure-numpy primitive Phase 3 adds; the wiring into the FSM lives in
the main script.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from autonomy.obstacle import forward_cone_distance


# Standard 640x480 pinhole, fx=fy~733 → ~46° horizontal FoV, ~36° vertical.
# Matches the head camera config in alex_onnx_walking_policy.py.
H, W = 480, 640
FX = FY = 732.99927
CX, CY = 320, 240   # int so we can use them as slice bounds in tests
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float32)


def _full_depth(z: float) -> np.ndarray:
    return np.full((H, W), z, dtype=np.float32)


# ── Behaviour ────────────────────────────────────────────────────────────────
def test_uniform_depth_returns_that_depth():
    """Constant 2 m wall in front → cone distance = 2.0."""
    depth = _full_depth(2.0)
    d = forward_cone_distance(depth, K, h_deg=20.0, v_deg=10.0, q_lowest=0.2)
    assert d == pytest.approx(2.0, abs=1e-3)


def test_returns_inf_for_empty_cone():
    """All-NaN depth → no valid pixels → +inf (caller treats as 'clear')."""
    depth = np.full((H, W), np.nan, dtype=np.float32)
    d = forward_cone_distance(depth, K, h_deg=20.0, v_deg=10.0, q_lowest=0.2)
    assert math.isinf(d) and d > 0


def test_close_obstacle_dominates_through_lowest_quantile():
    """Near pillar (1.0 m) covering most of the centre + far background (8.0 m).

    The lowest-20% percentile picks pixels from the pillar, so the returned
    cone distance is ~1.0 m, not the average ~4.5 m a naïve mean would give.
    """
    depth = _full_depth(8.0)
    # Wide pillar (~halves the cone width) at 1.0 m. The cone covers
    # u ∈ [cx − fx·tan20°, cx + fx·tan20°] ≈ ±267 px around cx=320,
    # so a ±150 px pillar is well over 50 % of the cone — the lowest 20 %
    # quantile is unambiguously inside the pillar.
    depth[:, CX - 150 : CX + 150] = 1.0
    d = forward_cone_distance(depth, K, h_deg=20.0, v_deg=10.0, q_lowest=0.2)
    assert d == pytest.approx(1.0, abs=0.05)


def test_far_clutter_outside_cone_is_ignored():
    """Close objects outside the ±h cone must not affect the result."""
    depth = _full_depth(5.0)
    # Paint a near pillar at the LEFT edge (well outside ±20° cone)
    depth[:, 0:50] = 0.3
    d = forward_cone_distance(depth, K, h_deg=20.0, v_deg=10.0, q_lowest=0.2)
    # Should still report ~5.0 m because 0.3 m strip is outside the cone
    assert d == pytest.approx(5.0, abs=0.05)


def test_invalid_depth_pixels_are_dropped():
    """Pixels with depth ≤ 0 or non-finite are excluded from the percentile."""
    depth = _full_depth(2.0)
    # Inject some 0-depth and inf-depth garbage in the centre
    depth[200:280, 300:340] = 0.0
    depth[100:120, 300:340] = np.inf
    d = forward_cone_distance(depth, K, h_deg=20.0, v_deg=10.0, q_lowest=0.2)
    # Valid pixels remain at 2.0 m → median of lowest 20% is still 2.0
    assert d == pytest.approx(2.0, abs=1e-3)


def test_max_depth_clip_drops_far_pixels():
    """Pixels beyond max_depth are not considered (treats clip-far as 'no data')."""
    depth = _full_depth(15.0)  # well past max_depth=10
    # A near 1.5 m blob in the centre
    depth[220:260, 310:330] = 1.5
    d = forward_cone_distance(
        depth, K, h_deg=20.0, v_deg=10.0, q_lowest=0.2, max_depth=10.0
    )
    # Only the near blob is valid → result ~1.5 m
    assert d == pytest.approx(1.5, abs=0.05)


# ── Cone geometry ────────────────────────────────────────────────────────────
def test_cone_width_matches_h_deg_via_intrinsics():
    """Setting h_deg=10° must select a narrower pixel band than h_deg=20°.

    Indirectly checks the cone-to-pixel conversion uses fx,cx correctly.
    Place a near strip in the annular band that lies inside ±20° but
    outside ±10°: the wide cone should see ~0.5 m, the narrow cone 5.0 m.
    """
    depth = _full_depth(5.0)
    u_inner = int(FX * math.tan(math.radians(11.0)))   # outside ±10°
    u_outer = int(FX * math.tan(math.radians(19.0)))   # inside  ±20°
    depth[:, CX + u_inner : CX + u_outer] = 0.5
    depth[:, CX - u_outer : CX - u_inner] = 0.5         # symmetric on the left
    d_wide   = forward_cone_distance(depth, K, h_deg=20.0, v_deg=10.0, q_lowest=0.2)
    d_narrow = forward_cone_distance(depth, K, h_deg=10.0, v_deg=10.0, q_lowest=0.2)
    assert d_wide   < 2.0
    assert d_narrow > 4.5


def test_v_cone_filters_floor_pixels():
    """A near floor strip below the v-cone must not be reported as obstacle."""
    depth = _full_depth(5.0)
    # Bottom 30 rows = floor at 0.5 m
    depth[H - 30 : H, :] = 0.5
    d = forward_cone_distance(depth, K, h_deg=20.0, v_deg=10.0, q_lowest=0.2)
    # 30 rows is well outside ±10° vertical cone → result stays at 5 m
    assert d == pytest.approx(5.0, abs=0.05)


# ── Robustness ───────────────────────────────────────────────────────────────
def test_depth_3d_shape_is_squeezed():
    """Some Isaac sensor outputs come as (H, W, 1); function must accept that."""
    depth = _full_depth(2.5)[:, :, None]   # (H, W, 1)
    d = forward_cone_distance(depth, K, h_deg=20.0, v_deg=10.0, q_lowest=0.2)
    assert d == pytest.approx(2.5, abs=1e-3)


def test_q_lowest_extreme_values():
    """q_lowest must be clamped to (0, 1] so the percentile call is well-defined."""
    depth = _full_depth(2.0)
    # q=1.0 → use all pixels → median is 2.0
    d_all = forward_cone_distance(depth, K, h_deg=20.0, v_deg=10.0, q_lowest=1.0)
    assert d_all == pytest.approx(2.0, abs=1e-3)
    # q=0 should not crash; treat as "use the single nearest pixel"
    d_min = forward_cone_distance(depth, K, h_deg=20.0, v_deg=10.0, q_lowest=0.0)
    assert d_min == pytest.approx(2.0, abs=1e-3)
