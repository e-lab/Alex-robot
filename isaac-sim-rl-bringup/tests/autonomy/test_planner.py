"""Tests for the A* path planner (Phase 3.5b).

The planner consumes a 2D occupancy grid + GridFrame (output of Phase 3.5a)
and a (start, goal) world XY pair, and returns a list of world-XY
waypoints around the obstacles. Used by the FSM to drive Alex toward
goals via deliberative planning instead of reactive cone steering.

Pure numpy + scipy + heapq — no Isaac, no pxr.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pytest

from autonomy.planner import plan_path
from autonomy.usd_occupancy import GridFrame


# ── Helpers ──────────────────────────────────────────────────────────────────
def _empty_grid(w: int = 40, h: int = 40, res: float = 0.1) -> tuple[np.ndarray, GridFrame]:
    """All-clear ``(h, w)`` grid centred on the world origin."""
    occ = np.zeros((h, w), dtype=bool)
    gf = GridFrame(
        origin_x=-w * res / 2.0,
        origin_y=-h * res / 2.0,
        resolution_m=res,
        width=w,
        height=h,
    )
    return occ, gf


def _path_endpoints_match(path, start_xy, goal_xy, *, tol_m: float):
    """Assert that path[0] ≈ start and path[-1] ≈ goal."""
    assert path is not None and len(path) >= 2
    sx, sy = start_xy
    gx, gy = goal_xy
    assert math.hypot(path[0][0] - sx, path[0][1] - sy) <= tol_m
    assert math.hypot(path[-1][0] - gx, path[-1][1] - gy) <= tol_m


def _path_is_collision_free(path, occ, gf, *, samples_per_seg: int = 10):
    """Walk along each segment, sample at ``samples_per_seg`` interior
    points, and assert every sample lands on a free cell.
    """
    for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
        for s in range(samples_per_seg + 1):
            t = s / samples_per_seg
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            ix, iy = gf.world_to_grid(x, y)
            assert gf.in_bounds(ix, iy), f"sample ({x},{y}) out of bounds"
            assert not occ[iy, ix], f"sample ({x},{y}) lands on obstacle"


# ── Empty grid: shortest path is a straight line ─────────────────────────────
def test_clear_grid_returns_two_point_path_after_smoothing():
    """Empty grid + smoothing → path is just [start, goal]."""
    occ, gf = _empty_grid()
    path = plan_path((-1.5, -1.5), (1.5, 1.5), occ, gf, inflation_m=0.0, smooth=True)
    assert path is not None
    assert len(path) == 2
    _path_endpoints_match(path, (-1.5, -1.5), (1.5, 1.5), tol_m=gf.resolution_m)


def test_clear_grid_unsmoothed_has_intermediate_waypoints():
    """Without smoothing, A* returns one waypoint per cell on the path,
    so the count grows with the diagonal length."""
    occ, gf = _empty_grid()
    path = plan_path((-1.5, -1.5), (1.5, 1.5), occ, gf, inflation_m=0.0, smooth=False)
    assert path is not None
    # Diagonal of length 30 cells → expect roughly that many waypoints
    assert len(path) >= 25


def test_start_equals_goal_returns_single_point():
    """Trivial case: start == goal → one-point path. Caller's waypoint
    follower advances past it immediately."""
    occ, gf = _empty_grid()
    path = plan_path((0.0, 0.0), (0.0, 0.0), occ, gf, inflation_m=0.0)
    assert path is not None
    assert len(path) >= 1
    assert math.hypot(path[0][0], path[0][1]) <= gf.resolution_m


# ── Detour around a single obstacle ──────────────────────────────────────────
def test_obstacle_in_path_forces_detour():
    """A vertical wall between start and goal forces the path to bend
    around it. The output is collision-free."""
    occ, gf = _empty_grid(w=60, h=40, res=0.1)
    # Vertical wall at x=0 from y=-1 to y=+1 (cells)
    ix_wall, _ = gf.world_to_grid(0.0, 0.0)
    iy_lo, _ = gf.world_to_grid(0.0, -1.0)[::-1] if False else gf.world_to_grid(0.0, -1.0)
    iy_hi, _ = gf.world_to_grid(0.0,  1.0)[::-1] if False else gf.world_to_grid(0.0,  1.0)
    # ``world_to_grid`` returns (ix, iy); we want iy here.
    _, iy_lo = gf.world_to_grid(0.0, -1.0)
    _, iy_hi = gf.world_to_grid(0.0,  1.0)
    occ[iy_lo : iy_hi + 1, ix_wall - 1 : ix_wall + 2] = True

    start = (-2.0, 0.0)
    goal  = ( 2.0, 0.0)
    path = plan_path(start, goal, occ, gf, inflation_m=0.0)
    _path_endpoints_match(path, start, goal, tol_m=gf.resolution_m)
    _path_is_collision_free(path, occ, gf)


# ── Unreachable ──────────────────────────────────────────────────────────────
def test_unreachable_goal_returns_none():
    """Goal walled in on all sides → planner returns None."""
    occ, gf = _empty_grid(w=40, h=40, res=0.1)
    # Build a closed box around the goal at (1.0, 0.0).
    ix_g, iy_g = gf.world_to_grid(1.0, 0.0)
    # 6-cell-wide ring (~0.6 m) — wide enough to block 0.4 m inflation too
    for d in range(-3, 4):
        occ[iy_g + 3, ix_g + d] = True   # top wall
        occ[iy_g - 3, ix_g + d] = True   # bottom
        occ[iy_g + d, ix_g - 3] = True   # left
        occ[iy_g + d, ix_g + 3] = True   # right
    path = plan_path((-1.0, 0.0), (1.0, 0.0), occ, gf, inflation_m=0.0)
    assert path is None


def test_completely_blocked_grid_returns_none():
    occ, gf = _empty_grid(w=20, h=20, res=0.1)
    occ[:] = True
    path = plan_path((-0.5, -0.5), (0.5, 0.5), occ, gf, inflation_m=0.0)
    assert path is None


# ── Inflation ────────────────────────────────────────────────────────────────
def test_inflation_enlarges_obstacles():
    """A narrow gap between two boxes that would normally be passable
    becomes blocked when inflation_m is large enough that the inflated
    boxes overlap."""
    occ, gf = _empty_grid(w=40, h=40, res=0.1)
    # Two boxes with a 0.5m gap between them (5 cells)
    _, iy_mid = gf.world_to_grid(0.0, 0.0)
    ix_left,  _ = gf.world_to_grid(-0.5, 0.0)
    ix_right, _ = gf.world_to_grid(+0.5, 0.0)
    # Each box: 1.0m wide × 0.4m tall
    occ[iy_mid - 2 : iy_mid + 3, ix_left  - 5 : ix_left  + 1] = True
    occ[iy_mid - 2 : iy_mid + 3, ix_right     : ix_right + 6] = True

    start = (-1.5, -1.0)
    goal  = ( 1.5,  1.0)
    # Without inflation: path threads the gap.
    p_thin = plan_path(start, goal, occ, gf, inflation_m=0.0)
    assert p_thin is not None
    # With inflation ≥ 0.30 m (3 cells per side > half the gap): path
    # has to go *around* the boxes instead of through.
    p_inflated = plan_path(start, goal, occ, gf, inflation_m=0.30)
    assert p_inflated is not None
    # Heuristic check: the inflated path is longer (more waypoints when
    # detouring), and at no point passes through the 0.5 m gap region.
    _path_is_collision_free(p_thin, occ, gf)
    _path_is_collision_free(p_inflated, occ, gf)


def test_inflation_blocks_goal_in_narrow_corridor():
    """Goal sits on a free cell, but inflation grows obstacles until no
    free cell exists adjacent to the goal → planner returns None."""
    occ, gf = _empty_grid(w=20, h=20, res=0.1)
    # Three-wide corridor; obstruct everything outside ix in [9,10,11]
    ix_g, iy_g = gf.world_to_grid(0.0, 0.0)
    occ[:, : ix_g - 1] = True
    occ[:, ix_g + 2 :] = True
    # 0.5 m inflation = 5 cells per side > corridor half-width (1 cell)
    path = plan_path((0.0, -0.8), (0.0, 0.8), occ, gf, inflation_m=0.50)
    assert path is None


# ── Snap-to-free-cell on start / goal ────────────────────────────────────────
def test_goal_inside_obstacle_snaps_to_nearest_free():
    """Common in practice: the goal XYZ from SAM3 is on the stove itself,
    which is an obstacle. The planner should plan to the **nearest free
    cell** instead of returning None."""
    occ, gf = _empty_grid(w=40, h=40, res=0.1)
    # Box at the centre, goal *inside* it.
    occ[18:22, 18:22] = True
    start = (-1.5, -1.5)
    goal  = ( 0.0,  0.0)
    path = plan_path(start, goal, occ, gf, inflation_m=0.0)
    assert path is not None
    # Last waypoint must be a *free* cell adjacent to the box.
    px, py = path[-1]
    ix, iy = gf.world_to_grid(px, py)
    assert not occ[iy, ix]
    # And reasonably close to the requested goal.
    assert math.hypot(px - goal[0], py - goal[1]) < 0.5


def test_start_inside_obstacle_snaps_to_nearest_free():
    """Symmetric: spawn on (or just inside) an obstacle still produces a path."""
    occ, gf = _empty_grid(w=40, h=40, res=0.1)
    # Big box covering the spawn at (-1.5, -1.5)
    ix_s, iy_s = gf.world_to_grid(-1.5, -1.5)
    occ[iy_s - 1 : iy_s + 2, ix_s - 1 : ix_s + 2] = True
    path = plan_path((-1.5, -1.5), (1.0, 1.0), occ, gf, inflation_m=0.0)
    assert path is not None
    px, py = path[0]
    ix, iy = gf.world_to_grid(px, py)
    assert not occ[iy, ix]


def test_goal_far_from_any_free_cell_returns_none():
    """If the snap radius is exhausted before finding a free cell, return None."""
    occ, gf = _empty_grid(w=20, h=20, res=0.1)
    occ[:] = True
    # The default snap radius (e.g. 1.0 m = 10 cells) won't find anything
    path = plan_path((0.0, 0.0), (0.5, 0.5), occ, gf, inflation_m=0.0)
    assert path is None


# ── Smoothing ────────────────────────────────────────────────────────────────
def test_smoothing_reduces_waypoint_count_on_clear_corridor():
    """Smoothing on a clear corridor reduces a many-cell A* path to two
    waypoints (start + goal)."""
    occ, gf = _empty_grid(w=60, h=20, res=0.1)
    p_unsmoothed = plan_path((-2.5, 0.0), (2.5, 0.0), occ, gf, smooth=False)
    p_smoothed   = plan_path((-2.5, 0.0), (2.5, 0.0), occ, gf, smooth=True)
    assert p_smoothed is not None and p_unsmoothed is not None
    assert len(p_smoothed) < len(p_unsmoothed)
    assert len(p_smoothed) == 2


def test_smoothing_preserves_collision_freedom():
    """Smoothing must not cut a corner *through* an obstacle."""
    occ, gf = _empty_grid(w=40, h=40, res=0.1)
    # Diagonal wall — 5 cells thick on the line y = -x
    for ix in range(15, 25):
        for off in range(-2, 3):
            iy = (40 - 1) - ix + off
            if 0 <= iy < 40:
                occ[iy, ix] = True
    path = plan_path((-1.5, -1.5), (1.5, 1.5), occ, gf, inflation_m=0.0, smooth=True)
    assert path is not None
    _path_is_collision_free(path, occ, gf)
