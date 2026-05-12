"""Tests for the USD-backed OccupancyProvider (LA-0a).

LA-0a is a pure refactor: ``UsdOccupancyProvider`` wraps the existing
``autonomy.usd_occupancy.occupancy_from_usd`` rasteriser behind the
``OccupancyProvider`` Protocol. Behavior is bit-identical to Phase 1-4 —
the grid the planner consumes, the path it produces, and the timing all
match what was there before.

These tests assert that contract:
* the wrapped grid equals the legacy ``occupancy_from_usd`` output cell-
  for-cell,
* ``update()`` is a no-op (USD is static ground truth),
* ``frontier_cells()`` returns ``[]`` (everything is FREE or OBSTACLE;
  there is no UNKNOWN),
* ``query()`` only ever returns FREE or OBSTACLE — never UNKNOWN
  (except for points outside the grid, which return UNKNOWN by
  graceful-degradation contract),
* planner output from ``UsdOccupancyProvider.grid_for_planner()`` is
  identical to planner output from the legacy grid.

All tests build their stage in-memory via ``Sdf.Layer.CreateAnonymous()``.
No Isaac, no on-disk USD. Pure pxr + numpy.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest

pxr = pytest.importorskip("pxr")
from pxr import Gf, Sdf, Usd, UsdGeom  # noqa: E402

from autonomy.planner import plan_path  # noqa: E402
from autonomy.usd_occupancy import occupancy_from_usd  # noqa: E402
from loco_x.occupancy import (  # noqa: E402
    CellState,
    OccupancyProvider,
    UsdOccupancyProvider,
)


# ── Fixtures ────────────────────────────────────────────────────────────────
def _make_stage() -> "Usd.Stage":
    layer = Sdf.Layer.CreateAnonymous(".usda")
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    return stage


def _add_box(
    stage: "Usd.Stage",
    path: str,
    *,
    xy_min: Tuple[float, float],
    xy_max: Tuple[float, float],
    z_min: float,
    z_max: float,
) -> None:
    """Add a translated/scaled unit cube whose world AABB matches the rect."""
    cube = UsdGeom.Cube.Define(stage, path)
    # Unit cube has extent (-1, +1) along each axis by default → size 2.
    sx = (xy_max[0] - xy_min[0]) / 2.0
    sy = (xy_max[1] - xy_min[1]) / 2.0
    sz = (z_max - z_min) / 2.0
    cx = (xy_max[0] + xy_min[0]) / 2.0
    cy = (xy_max[1] + xy_min[1]) / 2.0
    cz = (z_max + z_min) / 2.0
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, cz))
    xf.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))


def _build_stage_with_two_boxes() -> "Usd.Stage":
    stage = _make_stage()
    _add_box(stage, "/World/box_a",
             xy_min=(-1.0, -1.0), xy_max=(1.0, 1.0),
             z_min=0.2, z_max=1.0)
    _add_box(stage, "/World/box_b",
             xy_min=(2.0, 2.0), xy_max=(3.0, 3.0),
             z_min=0.2, z_max=1.0)
    return stage


# ── Interface contracts ─────────────────────────────────────────────────────
def test_usd_occupancy_provider_satisfies_protocol() -> None:
    """``UsdOccupancyProvider`` must be a structural ``OccupancyProvider``.

    Uses ``isinstance`` against a ``@runtime_checkable`` Protocol so a
    missing-method regression surfaces here instead of at the first
    runtime call from the agent.
    """
    stage = _build_stage_with_two_boxes()
    provider = UsdOccupancyProvider.from_stage(stage, use_collision_api=False)
    assert isinstance(provider, OccupancyProvider)


def test_cellstate_enum_values_are_stable() -> None:
    """Downstream code (planner, agent observation) treats CellState as an
    integer cell label. Lock the integer values so a re-order doesn't
    silently flip semantics."""
    assert int(CellState.UNKNOWN) == 0
    assert int(CellState.FREE) == 1
    assert int(CellState.OBSTACLE) == 2


# ── Bit-identical with legacy rasteriser ────────────────────────────────────
def test_usd_provider_grid_matches_legacy_usd_occupancy() -> None:
    """``UsdOccupancyProvider.grid_for_planner()`` returns the inflated-
    obstacle grid identical to the legacy ``occupancy_from_usd`` output
    (FREE/OBSTACLE only, encoded as ``CellState`` integers).

    This is the "bit-identical" canary for LA-0a — if it fails, we've
    drifted from Phase 1-4."""
    stage = _build_stage_with_two_boxes()
    legacy_occ, legacy_gf = occupancy_from_usd(
        stage,
        z_band=(0.10, 1.50),
        resolution_m=0.05,
        use_collision_api=False,
    )
    provider = UsdOccupancyProvider.from_stage(
        stage,
        z_band=(0.10, 1.50),
        resolution_m=0.05,
        use_collision_api=False,
    )

    grid = provider.grid_for_planner()
    assert grid.shape == legacy_occ.shape
    expected = np.where(legacy_occ, int(CellState.OBSTACLE), int(CellState.FREE))
    np.testing.assert_array_equal(grid, expected)

    # Frame metadata must agree too.
    assert provider.resolution_m() == legacy_gf.resolution_m
    assert provider.origin_xy() == (legacy_gf.origin_x, legacy_gf.origin_y)


def test_usd_provider_update_is_noop() -> None:
    """USD is static ground truth; ``update()`` must not change the grid."""
    stage = _build_stage_with_two_boxes()
    provider = UsdOccupancyProvider.from_stage(stage, use_collision_api=False)
    before = provider.grid_for_planner().copy()
    # Call update() with junk RGBD/pose — the provider must ignore it.
    provider.update(rgbd=None, head_cam_pose=None)
    after = provider.grid_for_planner()
    np.testing.assert_array_equal(before, after)


def test_usd_provider_frontier_cells_returns_empty() -> None:
    """USD provider has no UNKNOWN cells, so there are no frontiers."""
    stage = _build_stage_with_two_boxes()
    provider = UsdOccupancyProvider.from_stage(stage, use_collision_api=False)
    assert provider.frontier_cells(from_xy=(0.0, 0.0)) == []


def test_usd_provider_query_returns_free_or_obstacle_never_unknown() -> None:
    """``query(world_xy)`` returns FREE or OBSTACLE inside the grid —
    never UNKNOWN. (Out-of-bounds is a separate case; see next test.)"""
    stage = _build_stage_with_two_boxes()
    provider = UsdOccupancyProvider.from_stage(stage, use_collision_api=False)
    samples = [
        (0.0, 0.0),    # inside box_a → OBSTACLE
        (2.5, 2.5),    # inside box_b → OBSTACLE
        (0.0, 3.0),    # empty space → FREE
        (-0.4, -0.4),  # near origin, inside box_a → OBSTACLE
    ]
    seen = set()
    for xy in samples:
        state = provider.query(xy)
        assert state in (CellState.FREE, CellState.OBSTACLE), (
            f"query({xy}) returned {state}, expected FREE or OBSTACLE"
        )
        seen.add(state)
    # Sanity: at least one of each across the sample set.
    assert CellState.FREE in seen
    assert CellState.OBSTACLE in seen


def test_usd_provider_query_out_of_bounds_is_unknown() -> None:
    """A query *outside* the grid returns UNKNOWN — the one case where
    USD admits UNKNOWN (graceful degradation; ``query`` never raises)."""
    stage = _build_stage_with_two_boxes()
    provider = UsdOccupancyProvider.from_stage(stage, use_collision_api=False)
    assert provider.query((1000.0, 1000.0)) == CellState.UNKNOWN


# ── Planner-paths bit-identical ─────────────────────────────────────────────
def test_planner_paths_identical_to_phase_1_4_demo() -> None:
    """A* on the provider's grid must produce the same path it would have
    produced on the legacy boolean grid (within the same provider config).

    The Phase 1-4 ``plan_path`` consumes a boolean ``occ`` + ``GridFrame``;
    LA-0a's contract is that the provider produces an identical boolean
    view via ``grid == CellState.OBSTACLE``. Subsequent phases (LA-0c)
    will add a four-tier cost; the bit-identical test below pins the
    LA-0a behavior so we know what LA-0c diverges from."""
    stage = _build_stage_with_two_boxes()
    legacy_occ, legacy_gf = occupancy_from_usd(
        stage,
        resolution_m=0.05,
        use_collision_api=False,
    )
    provider = UsdOccupancyProvider.from_stage(
        stage, resolution_m=0.05, use_collision_api=False,
    )

    # Start and goal must both fall *inside* the auto-fit grid (which
    # for the two-box scene spans ~[-1.5, +3.5] each axis after the
    # default 0.5 m pad). Going from one free corner to the opposite
    # produces a 4-waypoint path that has to route around box_a.
    start = (-1.3, 1.3)
    goal = (1.3, -1.3)

    legacy_path = plan_path(start, goal, legacy_occ, legacy_gf, inflation_m=0.10)
    provider_bool_occ = (provider.grid_for_planner() == int(CellState.OBSTACLE))
    new_path = plan_path(
        start, goal, provider_bool_occ, legacy_gf, inflation_m=0.10,
    )

    assert legacy_path is not None and new_path is not None
    assert len(legacy_path) == len(new_path)
    for (a_x, a_y), (b_x, b_y) in zip(legacy_path, new_path):
        assert abs(a_x - b_x) < 1e-9 and abs(a_y - b_y) < 1e-9


def test_visited_fraction_for_usd_provider_equals_free_proportion() -> None:
    """USD provider has no UNKNOWN cells → ``visited_fraction`` reduces
    to the FREE proportion of in-bounds cells. We assert the weaker
    invariant: it equals ``count(FREE) / count(all cells)`` and contains
    no UNKNOWN cells anywhere (so the formula is well-defined)."""
    stage = _build_stage_with_two_boxes()
    provider = UsdOccupancyProvider.from_stage(stage, use_collision_api=False)
    grid = provider.grid_for_planner()
    assert (grid == int(CellState.UNKNOWN)).sum() == 0
    expected = float((grid == int(CellState.FREE)).sum()) / float(grid.size)
    assert abs(provider.visited_fraction() - expected) < 1e-9
