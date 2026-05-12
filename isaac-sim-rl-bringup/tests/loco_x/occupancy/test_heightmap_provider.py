"""LA-0b.2 head-cam integration tests for :class:`HeightMapProvider`.

The fold-in math is already pinned by LA-0b.1's synthetic harness; this
file tests the integration surface that LA-0b.2 introduces:

* :func:`depth_to_world_points` → :meth:`HeightMapProvider.update`
  end-to-end via a synthetic depth image,
* the path-invalidation watchdog (D10): scan a planned path and report
  the first cell that flipped to OBSTACLE,
* the path-staleness query: report the max staleness across a planned
  path so the agent observation can surface it (D10),
* the Protocol-level contracts that the LA-0b.1 tests don't already
  cover (init state, inflation-shaped queries, grid dtype).

No Isaac is required — depth images and poses are fabricated in code,
mirroring how the LA-0b.1 synthetic harness fabricated point clouds.
This lets the integration surface be exercised without spinning up
the simulator.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import pytest

from loco_x.occupancy import CellState
from loco_x.occupancy.backproject import (
    CameraIntrinsics,
    CameraPose,
    depth_to_world_points,
)
from loco_x.occupancy.heightmap_provider import HeightMapProvider
from loco_x.occupancy.synthetic import (
    Pose,
    PointCloud,
    add_gaussian_noise,
    box,
    flat_floor,
    merge,
    outlier_spike,
)


# ── Common config ───────────────────────────────────────────────────────────
GRID = dict(
    origin_xy=(-5.0, -5.0),
    size=(10.0, 10.0),
    cell_size_m=0.05,
    traversable_threshold_m=0.05,
    consistency_n=3,
    obs_window_s=1.0,
    stale_s=60.0,
    path_freshness_s=15.0,
)


def _provider() -> HeightMapProvider:
    return HeightMapProvider(**GRID)


def _intr(width: int = 32, height: int = 24, hfov_deg: float = 60.0) -> CameraIntrinsics:
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return CameraIntrinsics(
        width=width, height=height, fx=fx, fy=fx,
        cx=(width - 1) / 2.0, cy=(height - 1) / 2.0,
    )


# ── 1. Init state: all UNKNOWN ──────────────────────────────────────────────
def test_init_empty_map_is_all_unknown() -> None:
    """Fresh provider with no observations: every cell is UNKNOWN."""
    p = _provider()
    grid = p.grid_for_planner()
    expected = np.full(
        (int(GRID["size"][1] / GRID["cell_size_m"]),
         int(GRID["size"][0] / GRID["cell_size_m"])),
        int(CellState.UNKNOWN),
        dtype=np.int8,
    )
    np.testing.assert_array_equal(grid, expected)
    assert p.visited_fraction() == 0.0


# ── 2. Grid dtype is int8 in CellState values ──────────────────────────────
def test_grid_for_planner_returns_int8_grid_in_cellstate_values() -> None:
    """The planner expects an int8 grid whose values are CellState
    integers. Lock the dtype + value set."""
    p = _provider()
    cloud = flat_floor(n_points=20_000, seed=0)
    for k in range(3):
        p.update(point_cloud=cloud, pose=Pose(), now=float(k) * 0.1)
    grid = p.grid_for_planner()
    assert grid.dtype == np.int8
    unique_states = set(np.unique(grid).tolist())
    valid_states = {
        int(CellState.UNKNOWN),
        int(CellState.FREE),
        int(CellState.OBSTACLE),
    }
    assert unique_states.issubset(valid_states), f"unexpected values: {unique_states}"


# ── 3. End-to-end: synthetic depth → world points → heightmap ──────────────
def test_depth_image_through_backprojection_into_provider() -> None:
    """A synthetic depth image goes through ``depth_to_world_points``
    and gets folded into the provider via :meth:`update`. Verifies the
    integration surface end-to-end at the API level — no Isaac, but
    every byte travels the same path it would in the real pipeline."""
    p = _provider()
    intr = _intr(width=32, height=24, hfov_deg=60.0)
    # Camera at (0, 0, 0.5), looking forward (yaw=0) and slightly down
    # (pitch=-30°). Synthetic depth: a 2 m "wall" everywhere.
    pose = CameraPose(
        xy=(0.0, 0.0), z=0.5,
        yaw_rad=0.0, pitch_rad=math.radians(-30),
    )
    depth = np.full((intr.height, intr.width), 2.0, dtype=np.float32)

    # Three identical frames to satisfy the consistency gate.
    for k in range(3):
        cloud = depth_to_world_points(depth, intr, pose, timestamp=k * 0.1)
        p.update(point_cloud=cloud, pose=Pose(xy=pose.xy, z=pose.z),
                 now=float(k) * 0.1)

    # The wall lands at world XY along the camera's forward direction,
    # roughly 1.7 m in front of the camera (2 m depth * cos(30°)).
    # Cells near that lateral arc should be promoted away from UNKNOWN.
    grid = p.grid_for_planner()
    n_seen = int((grid != int(CellState.UNKNOWN)).sum())
    assert n_seen > 0, "expected at least some cells flipped away from UNKNOWN"


# ── 4. Path-invalidation watchdog: FREE→OBSTACLE flips trigger ─────────────
def test_path_invalidation_event_fires_on_free_to_obstacle_under_path() -> None:
    """A planned path through cells classified FREE. After a new tall
    obstacle is observed on one of those cells, the watchdog reports
    that cell as the first blockage along the path."""
    p = _provider()
    floor = flat_floor(n_points=200_000, seed=11)
    for k in range(3):
        p.update(point_cloud=floor, pose=Pose(), now=float(k) * 0.1)

    # Synthetic path along y=0 from x=-2 to x=+2.
    path: List[Tuple[float, float]] = [(x, 0.0) for x in np.linspace(-2.0, 2.0, 9)]
    # No obstacle yet → no invalidation.
    assert p.path_invalidated_by_new_obstacle(path) is None

    # Drop a tall obstacle right on a path cell.
    obstacle_xy = (1.0, 0.0)
    bx = box(
        xy_min=(obstacle_xy[0] - 0.05, obstacle_xy[1] - 0.05),
        xy_max=(obstacle_xy[0] + 0.05, obstacle_xy[1] + 0.05),
        z_min=0.0, z_max=0.5, n_points=200, seed=12,
    )
    # Three obstacle-bearing frames → promote that cell to OBSTACLE.
    for k in range(3):
        p.update(point_cloud=bx, pose=Pose(), now=1.0 + float(k) * 0.1)

    blocker = p.path_invalidated_by_new_obstacle(path)
    assert blocker is not None
    # Within one cell of where we placed it.
    assert abs(blocker[0] - obstacle_xy[0]) < GRID["cell_size_m"] * 1.5
    assert abs(blocker[1] - obstacle_xy[1]) < GRID["cell_size_m"] * 1.5


def test_no_event_when_obstacle_appears_off_path() -> None:
    """An obstacle that appears *next to* the path but not *on* it
    must NOT trigger the watchdog. The follower keeps walking."""
    p = _provider()
    floor = flat_floor(n_points=200_000, seed=13)
    for k in range(3):
        p.update(point_cloud=floor, pose=Pose(), now=float(k) * 0.1)

    path: List[Tuple[float, float]] = [(x, 0.0) for x in np.linspace(-2.0, 2.0, 9)]
    # Obstacle at y=2 — well off the path that runs along y=0.
    bx = box(xy_min=(0.95, 1.95), xy_max=(1.05, 2.05),
             z_min=0.0, z_max=0.5, n_points=200, seed=14)
    for k in range(3):
        p.update(point_cloud=bx, pose=Pose(), now=1.0 + float(k) * 0.1)

    assert p.path_invalidated_by_new_obstacle(path) is None


# ── 5. Path-staleness query (D10) ──────────────────────────────────────────
def test_path_staleness_query_returns_seconds_on_each_waypoint() -> None:
    """``max_path_staleness`` returns ``(staleness_s, world_xy)`` for
    the worst (oldest) cell on the planned path. Used by the agent
    observation builder to surface ``path_staleness: max NNs at ...``.

    Setup: observe two adjacent regions of the floor at different
    times, build a path of waypoints that all lie inside *observed*
    cells (so the staleness on every waypoint is finite), then query.
    The cell from the earlier observation must be the reported worst.

    A path that goes through never-observed cells would correctly
    report ``+inf`` staleness for those cells — the agent observation
    interprets that as "go peek before you walk". Tested separately
    below.
    """
    p = _provider()
    # Observe the "old" region (x around -1) at t=0 with a wide patch
    # that covers the path cells from (-1.5, 0) through (-0.5, 0).
    old_patch_pts = []
    for dx in np.linspace(-1.5, -0.5, 41):
        for dy in np.linspace(-0.2, 0.2, 9):
            old_patch_pts.append([dx, dy, 0.0])
    old_patch = PointCloud(points=np.array(old_patch_pts), timestamp=0.0)
    for k in range(3):
        p.update(point_cloud=old_patch, pose=Pose(), now=float(k) * 0.001)

    # Then observe the "new" region (x around +1) at t=10 s.
    new_patch_pts = []
    for dx in np.linspace(0.5, 1.5, 41):
        for dy in np.linspace(-0.2, 0.2, 9):
            new_patch_pts.append([dx, dy, 0.0])
    new_patch = PointCloud(points=np.array(new_patch_pts), timestamp=10.0)
    for k in range(3):
        p.update(point_cloud=new_patch, pose=Pose(),
                 now=10.0 + float(k) * 0.001)

    # Path only includes cells inside the *observed* regions; no
    # never-seen cells in between.
    path = [(-1.0, 0.0), (-0.5, 0.0), (0.5, 0.0), (1.0, 0.0)]
    worst_age, worst_xy = p.max_path_staleness(path)
    # The cell at (-1, 0) was observed at t=0; now ~10 s → staleness ~10 s.
    # The cell at (+1, 0) is fresh.
    assert 9.0 < worst_age < 11.0, f"got {worst_age}"
    assert worst_xy is not None
    assert worst_xy[0] < 0.0, f"got worst at {worst_xy}"


def test_path_staleness_returns_inf_for_never_observed_cells() -> None:
    """When the path crosses a never-observed cell, staleness is +inf
    and the watchdog reports that cell as the worst. The agent
    observation interprets +inf as "go peek before you walk"."""
    p = _provider()
    floor = flat_floor(n_points=200_000, seed=18)
    for k in range(3):
        p.update(point_cloud=floor, pose=Pose(), now=float(k) * 0.1)
    # Path includes a cell well outside the grid (never observable).
    path = [(0.0, 0.0), (100.0, 100.0)]
    worst_age, worst_xy = p.max_path_staleness(path)
    assert worst_age == float("inf")
    assert worst_xy == (100.0, 100.0)


# ── 6. Inflation consistency with the USD provider ─────────────────────────
def test_inflation_marks_neighbors_obstacle() -> None:
    """When the planner inflates the height-map grid, the resulting
    blocked region must include the cell containing the obstacle plus
    its immediate neighbors — same shape the legacy USD planner sees.

    This isn't testing the planner itself (that's covered by
    Phase 1-4); it's confirming the provider's classification gives
    the planner a grid in the *shape* it expects: cells crossing the
    threshold are OBSTACLE on a single-cell footprint, not a smeared
    multi-cell blob. (The blob comes from the planner's inflation
    step, not from the provider.)"""
    p = _provider()
    floor = flat_floor(n_points=200_000, seed=15)
    bx = box(xy_min=(0.95, -0.05), xy_max=(1.05, 0.05),
             z_min=0.0, z_max=0.5, n_points=200, seed=16)
    for k in range(3):
        p.update(point_cloud=merge(floor, bx),
                 pose=Pose(), now=float(k) * 0.1)
    grid = p.grid_for_planner()

    # Identify the obstacle cell. Confirm immediate neighbors are FREE
    # in the *provider* output — inflation is the planner's job, not
    # the provider's.
    ix, iy = p._world_to_grid(1.0, 0.0)
    assert grid[iy, ix] == int(CellState.OBSTACLE)
    # One cell away (still inside the obstacle's 0.10 m XY footprint
    # given our 0.05 m cell size, so technically still OBSTACLE).
    # Two cells away from the obstacle center is clear floor → FREE.
    assert grid[iy, ix + 3] == int(CellState.FREE)
    assert grid[iy + 3, ix] == int(CellState.FREE)


# ── 7. Visited fraction grows monotonically ────────────────────────────────
def test_visited_fraction_grows_monotonically_during_observation() -> None:
    """As more cells get observed, ``visited_fraction`` only goes up
    (until staleness decay kicks in, well outside this test)."""
    p = _provider()
    fractions = []

    # Observe progressively wider patches centred on origin.
    for i, half in enumerate((0.5, 1.0, 1.5, 2.0)):
        pts = []
        n = 31
        for dx in np.linspace(-half, half, n):
            for dy in np.linspace(-half, half, n):
                pts.append([dx, dy, 0.0])
        cloud = PointCloud(points=np.array(pts), timestamp=float(i) * 0.1)
        for k in range(3):
            p.update(point_cloud=cloud, pose=Pose(),
                     now=float(i) * 0.1 + float(k) * 0.01)
        fractions.append(p.visited_fraction())

    # Strictly non-decreasing.
    for a, b in zip(fractions, fractions[1:]):
        assert b >= a, f"visited_fraction regressed: {fractions}"


# ── 8. Re-promotion after contradictory burst ──────────────────────────────
def test_obstacle_can_be_re_promoted_to_free_after_being_cleared() -> None:
    """If a once-OBSTACLE cell stops seeing tall points and consistently
    sees low ones, the consistency gate flips it back to FREE.

    This is the "obstacle was a person who walked away" case: real-world
    dynamic obstacles need to clear once they leave. The latched state
    is updated on the next consistent burst."""
    p = _provider()
    # Step 1: promote (1, 0) to OBSTACLE via 3 frames of a tall object.
    bx = box(xy_min=(0.95, -0.05), xy_max=(1.05, 0.05),
             z_min=0.0, z_max=0.5, n_points=200, seed=17)
    for k in range(3):
        p.update(point_cloud=bx, pose=Pose(), now=float(k) * 0.1)
    assert p.query((1.0, 0.0)) == CellState.OBSTACLE

    # Step 2: from now on, three consistent frames of just floor below
    # the threshold at the same cell. (Within obs_window_s so the
    # gate has a fresh consistent set.)
    # Use a denser patch so the cell reliably gets >=5 points/frame.
    floor_pts = []
    for dx in np.linspace(-0.05, 0.05, 11):
        for dy in np.linspace(-0.05, 0.05, 11):
            floor_pts.append([1.0 + dx, 0.0 + dy, 0.0])
    cloud = PointCloud(points=np.array(floor_pts), timestamp=1.0)
    for k in range(3):
        p.update(point_cloud=cloud, pose=Pose(),
                 now=1.0 + float(k) * 0.1)
    assert p.query((1.0, 0.0)) == CellState.FREE
