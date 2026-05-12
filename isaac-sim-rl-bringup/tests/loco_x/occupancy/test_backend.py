"""Tests for the :class:`OccupancyBackend` facade (LA-0b.2).

The backend is the single integration point between the autonomy
script and the two occupancy provider impls. These tests cover:

* provider selection from a stub Hydra config (USD vs. heightmap),
* the ``step_perception`` fold-in path with a synthetic depth image,
* the watchdog query (``maybe_invalidate_path``) under three cases:
  empty path, all-clear path, path containing a new obstacle.

No Isaac required — the USD-provider construction path is exercised
in ``test_usd_provider.py``. Tests here use a stub provider to verify
the backend dispatches correctly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from loco_x.occupancy import CellState
from loco_x.occupancy.backend import OccupancyBackend, WatchdogReport
from loco_x.occupancy.backproject import CameraIntrinsics, CameraPose
from loco_x.occupancy.heightmap_provider import HeightMapProvider
from loco_x.occupancy.synthetic import Pose, PointCloud, box, flat_floor, merge


# ── Stub Hydra config (no omegaconf dependency in tests) ───────────────────
@dataclass
class _StubScene:
    name: str = "room"
    hm_bounds_xy: Optional[Tuple[float, float, float, float]] = None

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class _StubOcc:
    provider: str
    cell_size_m: float = 0.05
    traversable_threshold_m: float = 0.05
    stale_s: float = 60.0
    path_freshness_s: float = 15.0

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class _StubCfg:
    scene: _StubScene
    occupancy: _StubOcc


# ── Provider selection ─────────────────────────────────────────────────────
def test_from_cfg_heightmap_uses_heightmap_provider() -> None:
    """provider='heightmap' → backend wraps a HeightMapProvider."""
    cfg = _StubCfg(
        scene=_StubScene(hm_bounds_xy=(-5.0, -5.0, 5.0, 5.0)),
        occupancy=_StubOcc(provider="heightmap"),
    )
    backend = OccupancyBackend.from_cfg(cfg)
    assert isinstance(backend.provider, HeightMapProvider)


def test_from_cfg_unknown_raises() -> None:
    """Unknown provider name surfaces a clear error rather than
    silently picking a default."""
    cfg = _StubCfg(
        scene=_StubScene(),
        occupancy=_StubOcc(provider="hand-rolled-octomap"),
    )
    with pytest.raises(ValueError, match="unknown occupancy provider"):
        OccupancyBackend.from_cfg(cfg)


def test_from_cfg_usd_without_stage_raises() -> None:
    """The USD provider requires an open Usd.Stage. Forgetting the
    ``stage=`` kwarg surfaces a clear error."""
    cfg = _StubCfg(scene=_StubScene(), occupancy=_StubOcc(provider="usd"))
    with pytest.raises(ValueError, match="usd provider requires"):
        OccupancyBackend.from_cfg(cfg)


# ── step_perception end-to-end ─────────────────────────────────────────────
def test_step_perception_without_depth_keeps_clock_advancing() -> None:
    """No live camera yet (chest_depth=None, no head): the backend must
    keep the provider's clock advancing via ``advance_time_to`` so
    staleness queries still work, and the drive-through stamp records
    "the robot fit here" even without a camera frame."""
    cfg = _StubCfg(
        scene=_StubScene(hm_bounds_xy=(-5.0, -5.0, 5.0, 5.0)),
        occupancy=_StubOcc(provider="heightmap"),
    )
    backend = OccupancyBackend.from_cfg(cfg)
    backend.step_perception(
        chest_depth=None, chest_pose=None,
        robot_xy=(0.0, 0.0), now=42.0,
    )
    # Drive-through stamp → cell is FREE with fresh last_seen.
    assert backend.provider.query((0.0, 0.0)) == CellState.FREE
    assert backend.provider.staleness((0.0, 0.0)) == 0.0
    # A cell *not* on the drive-through path is still never-observed.
    assert math.isinf(backend.provider.staleness((3.0, 3.0)))


def test_step_perception_chest_only_folds_into_heightmap() -> None:
    """Chest-cam only (the primary, always-on case): depth + pose +
    intrinsics fold into the height map. After 3 ticks the consistency
    gate fires on observed cells."""
    cfg = _StubCfg(
        scene=_StubScene(hm_bounds_xy=(-5.0, -5.0, 5.0, 5.0)),
        occupancy=_StubOcc(provider="heightmap"),
    )
    intr = CameraIntrinsics(
        width=64, height=48,
        fx=64 / (2 * math.tan(math.radians(45))),
        fy=64 / (2 * math.tan(math.radians(45))),
        cx=31.5, cy=23.5,
    )
    backend = OccupancyBackend.from_cfg(cfg)
    backend.chest_intrinsics = intr
    backend.depth_stride = 1

    chest_pose = CameraPose(
        xy=(0.0, 0.0), z=1.0, yaw_rad=0.0,
        # Chest-cam mounted pitched 30° downward (matches Phase 1-4
        # CHEST_CAM_PITCH_DEG).
        pitch_rad=math.radians(-30),
    )
    depth = np.full((intr.height, intr.width), 2.0, dtype=np.float32)
    for k in range(3):
        backend.step_perception(
            chest_depth=depth, chest_pose=chest_pose,
            robot_xy=(0.0, 0.0), now=float(k) * 0.1,
        )

    # Some cells along the camera's line of sight must have been
    # promoted away from UNKNOWN.
    grid = backend.provider.grid_for_planner()
    assert int((grid != int(CellState.UNKNOWN)).sum()) > 0


# ── LA-0b.3: chest + head dual-stream fold-in ──────────────────────────────
def test_step_perception_chest_plus_head_folds_both_streams() -> None:
    """Both cameras feeding the same provider: cells covered by chest
    *or* head should end up observed; cells covered by both still
    classify consistently. Verifies the LA-0b.3 multi-camera fold-in
    is just additive — the consistency gate handles both streams the
    same way."""
    cfg = _StubCfg(
        scene=_StubScene(hm_bounds_xy=(-5.0, -5.0, 5.0, 5.0)),
        occupancy=_StubOcc(provider="heightmap"),
    )
    intr = CameraIntrinsics(
        width=64, height=48,
        fx=64 / (2 * math.tan(math.radians(45))),
        fy=64 / (2 * math.tan(math.radians(45))),
        cx=31.5, cy=23.5,
    )
    backend = OccupancyBackend.from_cfg(cfg)
    backend.chest_intrinsics = intr
    backend.head_intrinsics = intr   # same intrinsics for simplicity
    backend.depth_stride = 1

    # Chest sees ~1.7 m ahead at z≈0 (after the 30° pitch). Mock with
    # a uniform 1.5 m depth — back-projection puts the points slightly
    # forward and on the floor.
    chest_pose = CameraPose(
        xy=(0.0, 0.0), z=1.0, yaw_rad=0.0,
        pitch_rad=math.radians(-30),
    )
    chest_depth = np.full((intr.height, intr.width), 1.5, dtype=np.float32)

    # Head looks further out, slightly down (~10° pitch). 3 m depth
    # → points land further from the robot.
    head_pose = CameraPose(
        xy=(0.0, 0.0), z=1.5, yaw_rad=0.0,
        pitch_rad=math.radians(-10),
    )
    head_depth = np.full((intr.height, intr.width), 3.0, dtype=np.float32)

    for k in range(3):
        backend.step_perception(
            chest_depth=chest_depth, chest_pose=chest_pose,
            head_depth=head_depth, head_pose=head_pose,
            robot_xy=(0.0, 0.0), now=float(k) * 0.1,
        )

    # Total observed cells must exceed what chest alone produces.
    grid_both = backend.provider.grid_for_planner()
    n_both = int((grid_both != int(CellState.UNKNOWN)).sum())

    # Chest-only baseline.
    backend_baseline = OccupancyBackend.from_cfg(cfg)
    backend_baseline.chest_intrinsics = intr
    backend_baseline.depth_stride = 1
    for k in range(3):
        backend_baseline.step_perception(
            chest_depth=chest_depth, chest_pose=chest_pose,
            robot_xy=(0.0, 0.0), now=float(k) * 0.1,
        )
    grid_chest = backend_baseline.provider.grid_for_planner()
    n_chest = int((grid_chest != int(CellState.UNKNOWN)).sum())

    # Folding in head extends coverage strictly.
    assert n_both > n_chest, (
        f"head-cam stream didn't add cells: chest_only={n_chest}, "
        f"chest+head={n_both}"
    )


def test_step_perception_head_only_without_chest_is_supported() -> None:
    """A pathological case: chest stream is offline (no depth this
    tick) but the head-cam is firing. The backend must still fold the
    head feed into the height map rather than dropping the tick
    silently. Lets a hardware-side chest-cam outage degrade gracefully
    instead of stopping the map cold."""
    cfg = _StubCfg(
        scene=_StubScene(hm_bounds_xy=(-5.0, -5.0, 5.0, 5.0)),
        occupancy=_StubOcc(provider="heightmap"),
    )
    intr = CameraIntrinsics(
        width=64, height=48,
        fx=64 / (2 * math.tan(math.radians(45))),
        fy=64 / (2 * math.tan(math.radians(45))),
        cx=31.5, cy=23.5,
    )
    backend = OccupancyBackend.from_cfg(cfg)
    backend.head_intrinsics = intr
    backend.depth_stride = 1

    head_pose = CameraPose(
        xy=(0.0, 0.0), z=1.5, yaw_rad=0.0,
        pitch_rad=math.radians(-10),
    )
    head_depth = np.full((intr.height, intr.width), 2.5, dtype=np.float32)

    for k in range(3):
        backend.step_perception(
            chest_depth=None, chest_pose=None,
            head_depth=head_depth, head_pose=head_pose,
            robot_xy=(0.0, 0.0), now=float(k) * 0.1,
        )

    grid = backend.provider.grid_for_planner()
    assert int((grid != int(CellState.UNKNOWN)).sum()) > 0


# ── Watchdog query ─────────────────────────────────────────────────────────
def test_maybe_invalidate_empty_path_returns_default() -> None:
    """Empty or None path → no blocker, zero staleness."""
    cfg = _StubCfg(
        scene=_StubScene(hm_bounds_xy=(-5.0, -5.0, 5.0, 5.0)),
        occupancy=_StubOcc(provider="heightmap"),
    )
    backend = OccupancyBackend.from_cfg(cfg)
    rep = backend.maybe_invalidate_path([])
    assert rep == WatchdogReport(
        blocker_xy=None, max_staleness_s=0.0, stalest_xy=None
    )
    rep = backend.maybe_invalidate_path(None)
    assert rep.blocker_xy is None


def test_maybe_invalidate_clear_path() -> None:
    """A path through all-FREE cells: no blocker."""
    cfg = _StubCfg(
        scene=_StubScene(hm_bounds_xy=(-5.0, -5.0, 5.0, 5.0)),
        occupancy=_StubOcc(provider="heightmap"),
    )
    backend = OccupancyBackend.from_cfg(cfg)
    floor = flat_floor(n_points=200_000, seed=20)
    for k in range(3):
        backend.provider.update(point_cloud=floor, pose=Pose(),
                                now=float(k) * 0.1)
    path = [(x, 0.0) for x in np.linspace(-2.0, 2.0, 9)]
    rep = backend.maybe_invalidate_path(path)
    assert rep.blocker_xy is None


def test_maybe_invalidate_blocked_path_reports_first_obstacle() -> None:
    """A new obstacle appears under a planned path: the watchdog
    returns its XY (the first cell along the path that flipped)."""
    cfg = _StubCfg(
        scene=_StubScene(hm_bounds_xy=(-5.0, -5.0, 5.0, 5.0)),
        occupancy=_StubOcc(provider="heightmap"),
    )
    backend = OccupancyBackend.from_cfg(cfg)
    floor = flat_floor(n_points=200_000, seed=21)
    for k in range(3):
        backend.provider.update(point_cloud=floor, pose=Pose(),
                                now=float(k) * 0.1)
    bx = box(xy_min=(0.95, -0.05), xy_max=(1.05, 0.05),
             z_min=0.0, z_max=0.5, n_points=200, seed=22)
    for k in range(3):
        backend.provider.update(point_cloud=bx, pose=Pose(),
                                now=1.0 + float(k) * 0.1)
    path = [(x, 0.0) for x in np.linspace(-2.0, 2.0, 9)]
    rep = backend.maybe_invalidate_path(path)
    assert rep.blocker_xy is not None
    assert abs(rep.blocker_xy[0] - 1.0) < 0.06
    assert abs(rep.blocker_xy[1] - 0.0) < 0.06
