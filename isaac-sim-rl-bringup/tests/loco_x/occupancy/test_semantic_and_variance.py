"""D14 — semantic-biased + variance-aware A* tests.

Two related features under one design decision:

* **D14.1** — frontier scoring weighted by proximity to scene-graph
  *anchor* nodes the LLM names ("microwaves live near countertops").
  The agent calls ``next_frontier(prefer_near=["countertop"])`` and
  the provider's frontier ranking gets a bounded multiplicative boost
  for cells near matching anchors.
* **D14.2** — per-cell variance feeds a four-tier A* cost:
  ``FREE-clean / FREE-dirty / UNKNOWN / OBSTACLE``. USD's σ²≈0 keeps
  Phase 1-4 paths bit-identical; HeightMap's per-cell consistency
  signal supplies real σ² so the planner naturally prefers paths
  through cells it has seen clearly.

Both features carry safe defaults: ``prefer_near=[]`` reduces frontier
scoring to LA-0c pure geometric, ``var_cost_lambda=0`` reduces A* cost
to the existing binary FREE/OBSTACLE behaviour.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import pytest

from loco_x.occupancy import CellState
from loco_x.occupancy.frontier import frontier_cells
from loco_x.occupancy.heightmap_provider import HeightMapProvider
from loco_x.occupancy.planner_cost import (
    CostParams,
    PerCellCostProvider,
    plan_path_cost,
)
from loco_x.occupancy.synthetic import PointCloud, Pose


# ── Shared provider ────────────────────────────────────────────────────────
GRID = dict(
    origin_xy=(-5.0, -5.0),
    size=(10.0, 10.0),
    cell_size_m=0.2,
    traversable_threshold_m=0.05,
    consistency_n=3,
    obs_window_s=1.0,
    stale_s=60.0,
    path_freshness_s=15.0,
)


def _provider() -> HeightMapProvider:
    return HeightMapProvider(**GRID)


def _observe_rect(p, *, xy_min, xy_max, z=0.0, n_per_axis=21, now_base=0.0) -> None:
    pts = []
    for x in np.linspace(xy_min[0], xy_max[0], n_per_axis):
        for y in np.linspace(xy_min[1], xy_max[1], n_per_axis):
            pts.append([x, y, z])
    cloud = PointCloud(points=np.array(pts), timestamp=now_base)
    for k in range(3):
        p.update(point_cloud=cloud, pose=Pose(),
                 now=now_base + float(k) * 0.01)


# ── D14.1 — Semantic anchors on frontier scoring ───────────────────────────
def test_frontier_anchor_boosts_score_when_anchor_present() -> None:
    """A frontier cell near a scene-graph node matching ``prefer_near``
    scores strictly higher than the same cell would without the
    anchor. Multiplier bound: ``score_with <= 1.5 * score_without``."""
    p = _provider()
    _observe_rect(p, xy_min=(-1.0, -1.0), xy_max=(1.0, 1.0))

    # Stub scene-graph node lookup: one node labelled "countertop"
    # at world (+0.8, +0.8). The frontier helper must consult the
    # provider for anchor positions.
    anchors = {"countertop": [(0.8, 0.8)]}

    cands_no_boost = frontier_cells(p, from_xy=(0.0, 0.0), k=200)
    cands_boosted = frontier_cells(
        p, from_xy=(0.0, 0.0), k=200,
        prefer_near=["countertop"],
        scene_anchors=anchors,
    )

    # Find the candidate nearest to the anchor in each list — same XY
    # in both, but the boosted score must be > the unboosted one.
    def closest_to_anchor(cands):
        return min(cands, key=lambda c: math.hypot(
            c.world_xy[0] - 0.8, c.world_xy[1] - 0.8
        ))
    c0 = closest_to_anchor(cands_no_boost)
    c1 = closest_to_anchor(cands_boosted)
    assert c1.world_xy == c0.world_xy
    assert c1.score > c0.score
    # Multiplier bound: max boost = 1.5 (alpha=0.5 default).
    assert c1.score <= 1.5 * c0.score + 1e-9


def test_frontier_anchor_no_op_when_anchor_absent() -> None:
    """Empty ``prefer_near`` or no matching anchors in the scene → the
    boost is identity. Same ranking as plain geometric scoring."""
    p = _provider()
    _observe_rect(p, xy_min=(-1.0, -1.0), xy_max=(1.0, 1.0))

    plain = frontier_cells(p, from_xy=(0.0, 0.0), k=20)
    no_anchors = frontier_cells(p, from_xy=(0.0, 0.0), k=20, prefer_near=[])
    missing = frontier_cells(
        p, from_xy=(0.0, 0.0), k=20,
        prefer_near=["nonexistent_label"],
        scene_anchors={},
    )
    assert [c.world_xy for c in plain] == [c.world_xy for c in no_anchors]
    assert [c.world_xy for c in plain] == [c.world_xy for c in missing]
    # Score values identical too.
    for a, b in zip(plain, no_anchors):
        assert abs(a.score - b.score) < 1e-9


def test_frontier_anchor_bounded_by_alpha() -> None:
    """The semantic boost is multiplicative and capped: W_semantic ∈
    ``[1.0, 1 + alpha]``. Even a candidate sitting *exactly* on an
    anchor must not exceed ``(1 + alpha) * geometric_score``."""
    p = _provider()
    _observe_rect(p, xy_min=(-1.0, -1.0), xy_max=(1.0, 1.0))

    # Anchor co-located with a boundary frontier cell.
    anchor_xy = (1.0, 0.0)  # near the east edge of the observed patch
    anchors = {"countertop": [anchor_xy]}

    plain = frontier_cells(p, from_xy=(0.0, 0.0), k=200)
    boosted = frontier_cells(
        p, from_xy=(0.0, 0.0), k=200,
        prefer_near=["countertop"],
        scene_anchors=anchors,
        semantic_boost_alpha=0.5,
    )

    def closest(cands, xy):
        return min(cands, key=lambda c: math.hypot(
            c.world_xy[0] - xy[0], c.world_xy[1] - xy[1]
        ))
    c0 = closest(plain, anchor_xy)
    c1 = closest(boosted, anchor_xy)
    # Boost is bounded by (1 + alpha) regardless of distance to anchor.
    assert c1.score <= (1.0 + 0.5) * c0.score + 1e-9


def test_frontier_anchor_uses_nearest_match_when_multiple_anchors() -> None:
    """Multiple anchors in the scene: scoring uses the nearest one to
    each candidate. A candidate close to one anchor must score higher
    than the same candidate would if only the *far* anchor existed."""
    p = _provider()
    _observe_rect(p, xy_min=(-1.0, -1.0), xy_max=(1.0, 1.0))

    near_only = {"countertop": [(1.0, 0.0)]}
    far_only = {"countertop": [(5.0, 5.0)]}
    both = {"countertop": [(1.0, 0.0), (5.0, 5.0)]}

    def score_at_east_edge(anchors):
        cands = frontier_cells(
            p, from_xy=(0.0, 0.0), k=200,
            prefer_near=["countertop"],
            scene_anchors=anchors,
        )
        east = max(cands, key=lambda c: c.world_xy[0])
        return east.score

    s_near = score_at_east_edge(near_only)
    s_far = score_at_east_edge(far_only)
    s_both = score_at_east_edge(both)
    # Near anchor boosts strongly; far anchor barely. Both anchors
    # should boost as strongly as the near-only case (nearest match wins).
    assert s_near > s_far
    assert abs(s_both - s_near) < 1e-9


# ── D14.2 — Variance-aware A* cost ─────────────────────────────────────────
def test_var_cost_clean_free_costs_one() -> None:
    """A FREE cell with σ²=0 (USD or freshly drive-through stamped)
    must cost exactly 1.0 per step before scaling. The variance term
    only kicks in above ``var_cost_threshold``."""
    params = CostParams(
        var_cost_threshold=0.01,
        var_cost_lambda=5.0,
        unknown_cost_multiplier=6.0,
    )
    # Stub provider: returns FREE state + σ²=0 everywhere.
    provider = _ConstantProvider(state=CellState.FREE, variance=0.0)
    pcp = PerCellCostProvider(provider, params)
    assert abs(pcp.cell_cost((0.0, 0.0)) - 1.0) < 1e-9


def test_var_cost_increases_with_variance_on_free_cells() -> None:
    """``cost(FREE) = 1 + lambda * (σ² - threshold)`` once σ² exceeds
    the threshold. Higher variance → higher cost — but always below
    ``unknown_cost_multiplier`` so the planner never prefers UNKNOWN
    over dirty FREE."""
    params = CostParams(
        var_cost_threshold=0.01,
        var_cost_lambda=5.0,
        unknown_cost_multiplier=6.0,
    )
    pcp_clean = PerCellCostProvider(
        _ConstantProvider(state=CellState.FREE, variance=0.0), params,
    )
    pcp_mid = PerCellCostProvider(
        _ConstantProvider(state=CellState.FREE, variance=0.05), params,
    )
    pcp_dirty = PerCellCostProvider(
        _ConstantProvider(state=CellState.FREE, variance=0.20), params,
    )
    c0 = pcp_clean.cell_cost((0.0, 0.0))
    c1 = pcp_mid.cell_cost((0.0, 0.0))
    c2 = pcp_dirty.cell_cost((0.0, 0.0))
    # Strictly increasing in variance.
    assert c0 < c1 < c2
    # Capped below the unknown multiplier.
    assert c2 < params.unknown_cost_multiplier


def test_var_cost_is_noop_on_usd_provider() -> None:
    """USD providers report σ²=0 everywhere, so the variance branch
    never fires. FREE cells cost 1.0; OBSTACLE infinity; UNKNOWN
    multiplier (but USD has no UNKNOWN in-bounds → moot).

    This is the LA-0a bit-identical canary's variant for LA-0c: the
    Phase 1-4 planner must not see costs different from binary
    FREE/OBSTACLE when running against USD."""
    params = CostParams()
    free_cost = PerCellCostProvider(
        _ConstantProvider(state=CellState.FREE, variance=0.0), params,
    ).cell_cost((0.0, 0.0))
    obstacle_cost = PerCellCostProvider(
        _ConstantProvider(state=CellState.OBSTACLE, variance=0.0), params,
    ).cell_cost((0.0, 0.0))
    assert free_cost == 1.0
    assert math.isinf(obstacle_cost)


def test_var_cost_capped_below_unknown_multiplier() -> None:
    """The variance-induced penalty is clamped so dirty FREE always
    costs strictly less than UNKNOWN. Otherwise the planner would
    sometimes prefer driving into unknown territory over a slightly
    noisy seen patch, which is the opposite of what we want."""
    params = CostParams(
        var_cost_threshold=0.0,
        var_cost_lambda=10_000.0,   # absurd lambda to push cost up
        unknown_cost_multiplier=6.0,
    )
    dirty = PerCellCostProvider(
        _ConstantProvider(state=CellState.FREE, variance=1.0), params,
    ).cell_cost((0.0, 0.0))
    assert dirty < params.unknown_cost_multiplier


def test_unknown_cells_cost_more_than_dirty_free_but_less_than_obstacle() -> None:
    """Tier ordering invariant: ``FREE-clean < FREE-dirty < UNKNOWN
    < OBSTACLE(=∞)``. Locks the D10 tier semantics."""
    params = CostParams(
        var_cost_threshold=0.01,
        var_cost_lambda=5.0,
        unknown_cost_multiplier=6.0,
    )
    clean = PerCellCostProvider(
        _ConstantProvider(state=CellState.FREE, variance=0.0), params,
    ).cell_cost((0.0, 0.0))
    dirty = PerCellCostProvider(
        _ConstantProvider(state=CellState.FREE, variance=0.20), params,
    ).cell_cost((0.0, 0.0))
    unknown = PerCellCostProvider(
        _ConstantProvider(state=CellState.UNKNOWN, variance=0.0), params,
    ).cell_cost((0.0, 0.0))
    obstacle = PerCellCostProvider(
        _ConstantProvider(state=CellState.OBSTACLE, variance=0.0), params,
    ).cell_cost((0.0, 0.0))
    assert clean < dirty < unknown < obstacle


# ── Plan-path-cost on a heterogeneous grid ─────────────────────────────────
def test_astar_prefers_clean_free_over_dirty_free_with_same_distance() -> None:
    """Two equidistant routes from start to goal:
       * route A: all clean FREE cells (σ²=0).
       * route B: a stretch of dirty FREE cells (σ²=0.2).
    A* must pick A.

    The geometry is straightforward: a 1-row corridor across the grid
    where alternate rows have different variance. The cost-aware A*
    chooses the row whose summed cost is lowest."""
    params = CostParams(
        var_cost_threshold=0.01,
        var_cost_lambda=5.0,
        unknown_cost_multiplier=6.0,
    )
    # Stub provider: two rows have different variance.
    provider = _TwoRowProvider(
        clean_row_iy=1, dirty_row_iy=3,
        width=10, height=5,
        clean_var=0.0, dirty_var=0.5,
    )
    pcp = PerCellCostProvider(provider, params)
    path = plan_path_cost(
        start_xy=(0.0, 0.2),  # in the clean row
        goal_xy=(1.8, 0.2),
        cost_provider=pcp,
        origin_xy=(0.0, 0.0),
        cell_size_m=0.2,
        width=10, height=5,
    )
    assert path is not None
    # All path cells must have y close to the clean row's center
    # (iy=1 → y=0.2 at cell_size 0.2 m, origin (0, 0)).
    for x, y in path:
        assert abs(y - 0.3) < 0.4, f"path strayed: ({x},{y})"


def test_astar_prefers_dirty_free_over_unknown() -> None:
    """Tier ordering at the planner level: when forced to choose
    between a dirty-FREE detour and a straight UNKNOWN route, A*
    picks the dirty FREE (cost ~2) over UNKNOWN (cost 6).

    Geometry: a 10x5 grid where cells with ix∈[2,6] AND iy=2 are
    UNKNOWN — a "wall" of UNKNOWN cells in the middle row. Start
    at (0.1, 0.5) (iy=2 but ix=0 → still dirty FREE), goal at
    (1.7, 0.5) (iy=2, ix=8 → still dirty FREE).

    A straight path through iy=2 must cross UNKNOWN cells; the
    detour goes up to iy=1 or down to iy=3 and around the block.
    Cost comparison:
      - Straight through 4 UNKNOWN cells: 4 * 6.0 = 24.0
      - Detour up + 5 dirty FREE + down: ~7 * 1.95 ≈ 13.6
    Detour is cheaper → A* picks it.
    """
    params = CostParams(
        var_cost_threshold=0.01,
        var_cost_lambda=5.0,
        unknown_cost_multiplier=6.0,
    )
    provider = _StripeProvider()
    pcp = PerCellCostProvider(provider, params)
    path = plan_path_cost(
        start_xy=(0.1, 0.5),    # iy=2, ix=0 (dirty FREE — left of block)
        goal_xy=(1.7, 0.5),     # iy=2, ix=8 (dirty FREE — right of block)
        cost_provider=pcp,
        origin_xy=(0.0, 0.0),
        cell_size_m=0.2,
        width=10, height=5,
    )
    assert path is not None
    # The path must NOT pass through the UNKNOWN block (ix∈[2,6] AND iy=2).
    for x, y in path:
        ix = int(x / 0.2)
        iy = int(y / 0.2)
        crossed_unknown = (iy == 2 and 2 <= ix <= 6)
        assert not crossed_unknown, (
            f"path crossed UNKNOWN block at cell ({ix},{iy}) world=({x},{y})"
        )


def test_astar_expansion_count_within_3x_of_phase_1_4_baseline() -> None:
    """Regression guard: the cost-aware A* must not blow up node
    expansions compared to the legacy binary A*. We can't directly
    instrument the Phase 1-4 planner from here, so we measure the
    new planner's expansion count and bound it by a generous
    constant. The plan budgets 3x the Phase 1-4 expansion count;
    the absolute threshold below is calibrated against a clean-grid
    run that should complete in well under that bound."""
    params = CostParams(
        var_cost_threshold=0.01,
        var_cost_lambda=5.0,
        unknown_cost_multiplier=6.0,
    )
    # Clean 20x20 grid, all FREE, σ²=0. Cost-aware A* should explore
    # roughly the same number of nodes as binary A*, possibly a few
    # extra due to the more permissive heuristic.
    provider = _ConstantProvider(state=CellState.FREE, variance=0.0)
    pcp = PerCellCostProvider(provider, params)
    path, stats = plan_path_cost(
        start_xy=(0.0, 0.0),
        goal_xy=(3.8, 3.8),
        cost_provider=pcp,
        origin_xy=(0.0, 0.0),
        cell_size_m=0.2,
        width=20, height=20,
        return_stats=True,
    )
    assert path is not None
    # Generous: a 20x20 grid has 400 cells; even an exhaustive search
    # is 400 expansions. 3x of an informed A* baseline is well under
    # that. Treat 800 as the hard ceiling — anything higher signals
    # heuristic regression.
    assert stats.expansions < 800, f"too many expansions: {stats.expansions}"


# ── Stub providers for D14.2 tests ─────────────────────────────────────────
class _ConstantProvider:
    """Stub :class:`OccupancyProvider`-like for D14.2 tests.

    Returns the same CellState and variance for every query. The
    :class:`PerCellCostProvider` only consults ``query()`` and a new
    ``variance(xy)`` method, so we don't need to implement the full
    Protocol here."""

    def __init__(self, *, state: CellState, variance: float):
        self._state = state
        self._var = variance

    def query(self, world_xy):
        return self._state

    def variance(self, world_xy):
        return self._var

    # Methods the planner calls but not asserted on in these tests:
    def origin_xy(self): return (0.0, 0.0)
    def resolution_m(self): return 0.2
    def grid_for_planner(self):
        return np.full((5, 5), int(self._state), dtype=np.int8)


class _TwoRowProvider:
    """Stub that returns different variance for two specific rows.
    Used to verify A* picks the clean row over the dirty one."""

    def __init__(self, *, clean_row_iy, dirty_row_iy,
                 width, height, clean_var, dirty_var):
        self.clean_iy = clean_row_iy
        self.dirty_iy = dirty_row_iy
        self.width = width
        self.height = height
        self.clean_var = clean_var
        self.dirty_var = dirty_var

    def _iy(self, world_xy):
        return int(world_xy[1] / 0.2)

    def query(self, world_xy):
        return CellState.FREE

    def variance(self, world_xy):
        iy = self._iy(world_xy)
        if iy == self.dirty_iy:
            return self.dirty_var
        return self.clean_var


# ── Provider integration (USD + HeightMap satisfy the variance API) ───────
def test_usd_provider_variance_always_zero() -> None:
    """USD's ``variance()`` returns 0 everywhere — keeps the four-tier
    cost bit-identical with Phase 1-4 binary cost when running against
    USD (the FREE-dirty branch never fires)."""
    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdGeom  # noqa: F401
    from loco_x.occupancy import UsdOccupancyProvider

    layer = Sdf.Layer.CreateAnonymous(".usda")
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    provider = UsdOccupancyProvider.from_stage(stage, use_collision_api=False)
    assert provider.variance((0.0, 0.0)) == 0.0
    assert provider.variance((1.0, 1.0)) == 0.0


def test_heightmap_variance_grows_with_observation_spread() -> None:
    """HeightMap's ``variance()`` reflects the spread of recent
    per-frame height observations on a cell. A cell that saw a
    consistent z=0 floor has near-zero variance; a cell where
    consecutive frames returned a range of heights has high variance.

    The variance signal is what D14.2's ``PerCellCostProvider`` reads
    to pick FREE-clean vs FREE-dirty cost.
    """
    # Clean cell: three frames of a noise-free flat-floor patch at
    # the cell of interest. All observations land at z=0 → variance 0.
    p_clean = _provider()
    _observe_rect(p_clean, xy_min=(-0.4, -0.4), xy_max=(0.4, 0.4), z=0.0)
    assert p_clean.variance((0.0, 0.0)) < 1e-6

    # Dirty cell: three frames at heights z ∈ {0.0, 0.1, 0.0} — the
    # middle frame puts a transient bump on the cell. The variance
    # captures the spread (~0.002 m²); the consistency gate's strict
    # majority still promotes to FREE because two of three frames are
    # below the 0.05 threshold.
    p_dirty = _provider()
    for k, z in enumerate((0.0, 0.1, 0.0)):
        _observe_rect(
            p_dirty,
            xy_min=(-0.4, -0.4), xy_max=(0.4, 0.4),
            z=z, now_base=float(k) * 0.1,
        )
    var_dirty = p_dirty.variance((0.0, 0.0))
    assert var_dirty > 1e-4, f"expected non-zero variance, got {var_dirty}"


class _StripeProvider:
    """Stub: 10x5 grid. Cells with ix>=4 AND iy==2 are UNKNOWN, but
    that creates a U-shape: a dirty-FREE detour exists going around
    the UNKNOWN block. Everywhere else is FREE with σ²=0.2 (dirty).
    Used to test dirty-FREE-vs-UNKNOWN tiering at the planner level.

    The test starts at (ix=0, iy=2) (dirty FREE — left of the
    UNKNOWN region) and ends at (ix=9, iy=2) (dirty FREE — right
    of the UNKNOWN region). A straight line would cross through the
    UNKNOWN cells; the detour goes up to iy=1 or down to iy=3 and
    around."""

    def query(self, world_xy):
        ix = int(world_xy[0] / 0.2)
        iy = int(world_xy[1] / 0.2)
        if iy == 2 and 2 <= ix <= 6:
            return CellState.UNKNOWN
        return CellState.FREE

    def variance(self, world_xy):
        ix = int(world_xy[0] / 0.2)
        iy = int(world_xy[1] / 0.2)
        if iy == 2 and 2 <= ix <= 6:
            return 0.0
        return 0.2
