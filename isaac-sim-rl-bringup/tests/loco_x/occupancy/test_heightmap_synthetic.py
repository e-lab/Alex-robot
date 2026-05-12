"""LA-0b.1 synthetic point-cloud harness for :class:`HeightMapProvider`.

Mandatory gate before LA-0b.2 (head-cam integration). Each test
generates its point cloud in code, runs in <1 s, uses ``now``-injection
(D8) so no real time elapses, and asserts a deterministic outcome.

No Isaac, no GPU. The provider's CPU-fallback path must satisfy all 10
contracts here before any depth-back-projection / camera-model code is
wired up. If we don't pin the *fold-in math* now, real-camera bugs will
hide behind sensor-noise bugs in LA-0b.2.

A note on point density: with a 5 cm cell size, the default 10x10 m
grid has 40,000 cells. A uniformly random cloud of 10k points would
leave most cells unobserved (~0.25 expected hits/cell), which would
make many test assertions race-dependent on the RNG. So the floor
clouds in these tests sample 200k points (~5 expected hits/cell) —
enough to reliably cover the specific cells each test inspects.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest

from loco_x.occupancy import CellState
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
# Tight 10x10 m provider with 5 cm cells matches the plan defaults
# (D10: cell_size_m=0.05, traversable_threshold_m=0.05).
GRID = dict(
    origin_xy=(-5.0, -5.0),
    size=(10.0, 10.0),
    cell_size_m=0.05,
    traversable_threshold_m=0.05,
    consistency_n=3,           # N>=3 observations before FREE/OBSTACLE flip
    obs_window_s=1.0,          # consistency-gate temporal window
    stale_s=60.0,
    path_freshness_s=15.0,
)


def _provider() -> HeightMapProvider:
    return HeightMapProvider(**GRID)


def _cell_state(provider: HeightMapProvider, xy: Tuple[float, float]) -> CellState:
    """query() helper; provider's contract returns CellState."""
    return provider.query(xy)


# ── 1. Flat floor → all FREE ────────────────────────────────────────────────
def test_flat_floor_all_free() -> None:
    """Dense floor cloud → after N>=3 consistent observations every
    observed cell is FREE; none OBSTACLE. Unobserved cells stay
    UNKNOWN (the boundary case)."""
    p = _provider()
    cloud = flat_floor(n_points=200_000, seed=0)
    for k in range(3):
        # Same cloud three times → consistency gate satisfied; no noise
        # means each frame is "identical enough" to count toward the gate.
        p.update(point_cloud=cloud, pose=Pose(), now=float(k) * 0.1)

    grid = p.grid_for_planner()
    # No OBSTACLE cells anywhere — floor is below traversable_threshold.
    assert (grid == int(CellState.OBSTACLE)).sum() == 0
    # Sampled center cells are FREE (we observed enough to flip them).
    for xy in [(0.0, 0.0), (2.0, 2.0), (-3.0, 4.0)]:
        assert _cell_state(p, xy) == CellState.FREE, f"{xy}"


# ── 2. Tall box → OBSTACLE under footprint, FREE around it ─────────────────
def test_tall_box_marks_obstacle() -> None:
    """Floor + a 0.3x0.3x0.6 m box centered at (+2, +1). The cells
    inside the box footprint must be OBSTACLE; the cells just outside
    (still on flat floor) must be FREE."""
    p = _provider()
    floor = flat_floor(n_points=200_000, seed=1)
    bx = box(xy_min=(1.85, 0.85), xy_max=(2.15, 1.15),
             z_min=0.0, z_max=0.6, n_points=500, seed=2)
    cloud = merge(floor, bx)

    for k in range(3):
        p.update(point_cloud=cloud, pose=Pose(), now=float(k) * 0.1)

    # Center of the box → OBSTACLE.
    assert _cell_state(p, (2.0, 1.0)) == CellState.OBSTACLE
    # 30 cm away from the box footprint → FREE.
    assert _cell_state(p, (2.0, 1.6)) == CellState.FREE
    assert _cell_state(p, (1.4, 1.0)) == CellState.FREE


# ── 3. Low cable (2 cm) → still FREE (below threshold) ─────────────────────
def test_low_cable_stays_free() -> None:
    """A 0.02 m thin object on the floor must NOT flip its cell to
    OBSTACLE — the current gait can scrape over it. This is the test
    that justifies setting traversable_threshold_m=0.05."""
    p = _provider()
    floor = flat_floor(n_points=200_000, seed=3)
    cable = box(xy_min=(0.95, -0.025), xy_max=(1.05, 0.025),
                z_min=0.0, z_max=0.02, n_points=200, seed=4)
    cloud = merge(floor, cable)

    for k in range(3):
        p.update(point_cloud=cloud, pose=Pose(), now=float(k) * 0.1)

    # Cable cell remains FREE — max-height 2 cm < threshold 5 cm.
    assert _cell_state(p, (1.0, 0.0)) == CellState.FREE


# ── 4. Low furniture base (8 cm) → OBSTACLE (above threshold) ──────────────
def test_low_furniture_base_marks_obstacle() -> None:
    """A 0.08 m coffee-table foot must classify as OBSTACLE. Confirms
    the current gait cannot step over an 8 cm obstacle, so the height
    map correctly treats it as one."""
    p = _provider()
    floor = flat_floor(n_points=200_000, seed=5)
    base = box(xy_min=(-1.05, -1.05), xy_max=(-0.95, -0.95),
               z_min=0.0, z_max=0.08, n_points=200, seed=6)
    cloud = merge(floor, base)

    for k in range(3):
        p.update(point_cloud=cloud, pose=Pose(), now=float(k) * 0.1)

    assert _cell_state(p, (-1.0, -1.0)) == CellState.OBSTACLE


# ── 5. Camera motion fills cumulative visited cells ─────────────────────────
def test_camera_motion_fills_in_visited_cells() -> None:
    """Drive a synthetic "frustum" through the scene at 1 m/s. Each
    frame is a small patch of points (the visible cone). Cells inside
    the cumulative visited region become FREE; cells far away that we
    never observed stay UNKNOWN.

    We don't model a real camera here — we just hand the provider a
    point cloud that *would* be visible from each pose. The provider's
    job is to fold them in; the back-projection is LA-0b.2."""
    p = _provider()
    # Camera walks east from (-3, 0) to (+3, 0), one frame per 0.5 m.
    poses = [Pose(xy=(x, 0.0)) for x in np.linspace(-3.0, 3.0, 13)]
    for i, pose in enumerate(poses):
        # "Visible patch": dense flat-floor cloud in a 1 m radius
        # around the pose. Dense enough to reliably cover every cell
        # in the patch (~5 expected hits/cell).
        patch_pts = []
        for dx in np.linspace(-0.9, 0.9, 37):
            for dy in np.linspace(-0.9, 0.9, 37):
                patch_pts.append([pose.xy[0] + dx, pose.xy[1] + dy, 0.0])
        patch = PointCloud(points=np.array(patch_pts), timestamp=i * 0.1)
        # Fold the same patch 3x to satisfy the consistency gate at each pose.
        for k in range(3):
            p.update(point_cloud=patch, pose=pose, now=i * 0.1 + k * 0.01)

    # Cells on the walked corridor are FREE.
    assert _cell_state(p, (0.0, 0.0)) == CellState.FREE
    assert _cell_state(p, (-2.5, 0.0)) == CellState.FREE
    assert _cell_state(p, (2.5, 0.0)) == CellState.FREE
    # Cells far from any pose stay UNKNOWN.
    assert _cell_state(p, (0.0, 4.5)) == CellState.UNKNOWN
    assert _cell_state(p, (-4.5, -4.5)) == CellState.UNKNOWN


# ── 6. Noisy depth + consistency gate → still-correct labels ───────────────
def test_noisy_depth_with_consistency_gate() -> None:
    """Repeat 5 noisy frames (σ=0.02 m). With N>=3 required, the FREE
    / OBSTACLE labels match the noise-free baseline. No phantom
    obstacles spawn from individual noisy points."""
    p = _provider()
    floor = flat_floor(n_points=200_000, seed=7)
    bx = box(xy_min=(1.85, 0.85), xy_max=(2.15, 1.15),
             z_min=0.0, z_max=0.6, n_points=500, seed=8)
    base = merge(floor, bx)
    for k in range(5):
        noisy = add_gaussian_noise(base, sigma_m=0.02, seed=100 + k)
        p.update(point_cloud=noisy, pose=Pose(), now=float(k) * 0.1)

    # Box still OBSTACLE.
    assert _cell_state(p, (2.0, 1.0)) == CellState.OBSTACLE
    # Floor still FREE (no phantom-obstacle pop from σ=0.02 m noise:
    # individual cells might see a 2-3 cm spike but stay under 5 cm).
    assert _cell_state(p, (0.0, 0.0)) == CellState.FREE
    assert _cell_state(p, (-2.0, 2.0)) == CellState.FREE


# ── 7. Single-frame outlier → NOT an obstacle (gate rejects it) ────────────
def test_single_noisy_frame_does_not_create_phantom() -> None:
    """One frame with three reflection-artifact points at z=10 m.
    The consistency gate (N>=3 within obs_window_s) rejects them, so
    that cell stays UNKNOWN (we've seen it < N times) or, if other
    floor points covered it, FREE — never OBSTACLE."""
    p = _provider()
    bad = outlier_spike(xy=(3.0, -2.0), z=10.0, n_points=3, timestamp=0.0)
    p.update(point_cloud=bad, pose=Pose(), now=0.0)

    # Single frame → consistency gate (N=3) not met yet; cell is
    # UNKNOWN or FREE depending on whether floor support exists.
    # Either way: never OBSTACLE.
    state = _cell_state(p, (3.0, -2.0))
    assert state != CellState.OBSTACLE


# ── 8. Staleness decay (now-injection, no real sleeping) ───────────────────
def test_staleness_decay_in_synthetic_time() -> None:
    """Observe a cell, advance ``now`` past ``stale_s``, query again.
    Cell reverts to UNKNOWN. Uses D8 ``now``-injection — no real time
    elapses."""
    p = _provider()
    cloud = flat_floor(n_points=200_000, seed=9)
    for k in range(3):
        p.update(point_cloud=cloud, pose=Pose(), now=float(k) * 0.1)

    # Cell is FREE at t=0.3 s.
    assert _cell_state(p, (0.0, 0.0)) == CellState.FREE

    # Advance synthetic time past the 60 s stale window without any
    # new observations.
    p.advance_time_to(now=120.0)
    assert _cell_state(p, (0.0, 0.0)) == CellState.UNKNOWN


# ── 9. Path-freshness window is shorter than global staleness ──────────────
def test_path_freshness_window_shorter_than_global() -> None:
    """After ``path_freshness_s`` but before ``stale_s``, cells still
    classify as FREE globally, but ``path_staleness()`` reports them
    as stale-on-path. The path is NOT invalidated (D10) — staleness is
    a *signal*, not an action."""
    p = _provider()
    cloud = flat_floor(n_points=200_000, seed=10)
    for k in range(3):
        p.update(point_cloud=cloud, pose=Pose(), now=float(k) * 0.1)

    cell_xy = (1.0, 1.0)
    # Advance to 30 s — past path_freshness_s (15) but well under
    # stale_s (60).
    p.advance_time_to(now=30.0)

    assert _cell_state(p, cell_xy) == CellState.FREE
    staleness = p.staleness(cell_xy)
    # Cell hasn't been seen in ~30 s — beyond path_freshness_s.
    assert staleness > GRID["path_freshness_s"]
    # But cell is still globally fresh.
    assert staleness < GRID["stale_s"]


# ── 10. Drive-through stamping overrides low-variance floor ────────────────
def test_drive_through_overrides_low_variance_floor() -> None:
    """Stamp a cell FREE via ``drive_through()`` (the robot physically
    walked through it). Subsequent noisy observations of that cell
    don't downgrade it."""
    p = _provider()
    cell_xy = (0.5, 0.5)
    p.drive_through(world_xy=cell_xy, now=0.0)
    assert _cell_state(p, cell_xy) == CellState.FREE

    # Now hit it with a noisy "tall" outlier on a single frame.
    bad = outlier_spike(xy=cell_xy, z=2.0, n_points=3, timestamp=1.0)
    p.update(point_cloud=bad, pose=Pose(), now=1.0)

    # Cell stayed FREE — drive-through carries high confidence and
    # a single noisy frame can't override it.
    assert _cell_state(p, cell_xy) == CellState.FREE
