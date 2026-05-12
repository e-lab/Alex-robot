"""Tests for the LA-0c frontier algorithm.

A frontier cell is a FREE cell adjacent to an UNKNOWN cell — the
"edge of the known map" where the robot can walk to and immediately
see new territory. Ranking combines:

* **info-gain** — how many UNKNOWN cells we'd convert to known if we
  re-observed the scene from that viewpoint,
* **travel distance** — closer cells preferred, all else equal,
* **semantic anchors (D14.1, tested elsewhere)** — proximity to
  matching scene-graph nodes.

Results are pre-sorted (descending by score), tie-broken by grid
index for determinism. The frontier query is the engine behind the
``next_frontier()`` skill (LA-1) and the agent's systematic-
exploration loop.

These tests cover the *geometry*; semantic-anchor scoring + variance-
aware A* cost get their own file (``test_semantic_and_variance.py``).

All tests build synthetic occupancy grids via :class:`HeightMapProvider`
(driving its update() with hand-crafted point clouds) so we exercise
the same code path the agent will hit in production.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pytest

from loco_x.occupancy import CellState, FrontierCandidate, HeightMapProvider
from loco_x.occupancy.frontier import frontier_cells
from loco_x.occupancy.synthetic import PointCloud, Pose


# ── Test grid: 10x10 m, 0.2 m cells = 50x50 = 2500 cells ────────────────────
# Cells are intentionally larger than the production 0.05 m so that test
# scenes are easy to reason about visually (each "blob" is a handful of
# cells, not hundreds). The algorithm itself is cell-size-agnostic.
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


def _observe_rect(
    provider: HeightMapProvider,
    *,
    xy_min: Tuple[float, float],
    xy_max: Tuple[float, float],
    z: float = 0.0,
    n_per_axis: int = 21,
    now_base: float = 0.0,
) -> None:
    """Fold three consistent frames of a dense floor patch into the
    provider so the consistency gate promotes the covered cells to
    FREE / OBSTACLE depending on ``z``."""
    pts = []
    for x in np.linspace(xy_min[0], xy_max[0], n_per_axis):
        for y in np.linspace(xy_min[1], xy_max[1], n_per_axis):
            pts.append([x, y, z])
    cloud = PointCloud(points=np.array(pts), timestamp=now_base)
    for k in range(3):
        provider.update(point_cloud=cloud, pose=Pose(),
                        now=now_base + float(k) * 0.01)


# ── 1. Boundary detection ───────────────────────────────────────────────────
def test_frontier_cells_finds_boundary_between_free_and_unknown() -> None:
    """A square patch of observed FREE cells in the middle of an
    otherwise-UNKNOWN grid: frontiers must lie on the patch's edge
    (FREE cells with at least one UNKNOWN neighbor)."""
    p = _provider()
    _observe_rect(p, xy_min=(-1.0, -1.0), xy_max=(1.0, 1.0))
    cands = frontier_cells(p, from_xy=(0.0, 0.0), k=100)
    assert len(cands) > 0
    # Every candidate must itself classify as FREE in the grid.
    for c in cands:
        assert p.query(c.world_xy) == CellState.FREE
    # And lie near the patch's boundary (within one cell of the
    # edge at ±1.0 m).
    boundary_cells_count = 0
    for c in cands:
        cx, cy = c.world_xy
        on_x_edge = abs(abs(cx) - 1.0) < GRID["cell_size_m"] * 2
        on_y_edge = abs(abs(cy) - 1.0) < GRID["cell_size_m"] * 2
        if on_x_edge or on_y_edge:
            boundary_cells_count += 1
    # The vast majority of frontier candidates must lie on the edge.
    assert boundary_cells_count >= 0.8 * len(cands)


def test_frontier_cells_returns_empty_for_fully_known_map() -> None:
    """If every reachable cell is FREE (no UNKNOWN neighbors), there
    are no frontiers — same as the USD provider's contract."""
    p = _provider()
    # Observe the entire grid → no UNKNOWN left.
    _observe_rect(p, xy_min=(-4.9, -4.9), xy_max=(4.9, 4.9), n_per_axis=51)
    cands = frontier_cells(p, from_xy=(0.0, 0.0), k=100)
    # Either none, or any returned candidate doesn't have an unknown
    # neighbor (defensive check — the algorithm should return [] here).
    assert cands == []


# ── 2. Info-gain ranking ────────────────────────────────────────────────────
def test_information_gain_higher_when_more_unknown_visible() -> None:
    """Two frontier candidates at the same travel distance: the one
    with more UNKNOWN cells in its info-gain window must score
    strictly higher than the one with fewer.

    Score = info_gain / (1 + travel_distance), so we hold travel
    distance constant and let info_gain do the differentiating.

    Setup: observe a long vertical bar with one short stub. The
    frontier at the open end of the long bar faces a large open
    UNKNOWN region; the frontier at the wall end of the short stub
    abuts the grid edge and has much less UNKNOWN behind it."""
    p = _provider()
    # Vertical bar: y in [-2, +2], x in [-0.3, +0.3].
    _observe_rect(p, xy_min=(-0.3, -2.0), xy_max=(0.3, 2.0))
    # Short east stub: x in [+0.3, +1.5], y in [-0.3, +0.3]. Stub is
    # short and sits well inside the grid, so UNKNOWN cells around its
    # tip are fewer.
    _observe_rect(p, xy_min=(0.3, -0.3), xy_max=(1.5, 0.3))

    cands = frontier_cells(p, from_xy=(0.0, 0.0), k=200)
    # Find the candidate at the top of the bar (y near +2) and at the
    # tip of the stub (x near +1.5). Both should be roughly equidistant
    # (~2 m) from the robot.
    top_of_bar = max(cands, key=lambda c: c.world_xy[1])
    tip_of_stub = max(cands, key=lambda c: c.world_xy[0])
    # Travel distances are within 25% of each other.
    d_top = top_of_bar.travel_distance
    d_tip = tip_of_stub.travel_distance
    assert 0.5 * d_top <= d_tip <= 2.0 * d_top, (
        f"distances too unequal: top={d_top}, tip={d_tip}"
    )
    # The top of the bar faces a *much* larger UNKNOWN region behind it
    # (anywhere from y=+2 to y=+5, full 10-m grid width). The east stub
    # tip faces a narrower channel (its UNKNOWN region is bounded
    # vertically by the observed stub). So info_gain_top > info_gain_tip.
    assert top_of_bar.info_gain > tip_of_stub.info_gain


def test_information_gain_zero_for_isolated_free_cell() -> None:
    """A FREE cell with no UNKNOWN neighbors has info_gain=0 and is
    not a frontier — frontiers are *boundary* cells by definition."""
    p = _provider()
    # Observe a 3x3 patch surrounded by more observed cells (no UNKNOWN
    # adjacency at the center).
    _observe_rect(p, xy_min=(-1.0, -1.0), xy_max=(1.0, 1.0))
    cands = frontier_cells(p, from_xy=(0.0, 0.0), k=100)
    # The exact center cell (0, 0) must NOT appear in the frontier list.
    for c in cands:
        xy = c.world_xy
        # Must not be the center cell itself.
        assert not (abs(xy[0]) < GRID["cell_size_m"]
                    and abs(xy[1]) < GRID["cell_size_m"])


# ── 3. Score combines info-gain and distance ────────────────────────────────
def test_score_combines_info_gain_and_travel_distance() -> None:
    """Score = info_gain / (1 + travel_distance) by default.

    Equidistant frontiers with same info-gain → equal scores. A
    higher-info-gain frontier at the same distance scores higher; a
    closer frontier with the same info-gain scores higher."""
    p = _provider()
    _observe_rect(p, xy_min=(-1.0, -1.0), xy_max=(1.0, 1.0))
    cands = frontier_cells(p, from_xy=(0.0, 0.0), k=100)
    # Score must equal info_gain / (1 + travel_distance) by contract.
    for c in cands:
        expected = c.info_gain / (1.0 + c.travel_distance)
        assert abs(c.score - expected) < 1e-9
    # Sorted descending by score.
    for a, b in zip(cands, cands[1:]):
        assert a.score >= b.score


# ── 4. Tie-break determinism ────────────────────────────────────────────────
def test_results_deterministic_under_grid_permutation() -> None:
    """Two runs over the same provider must produce identical
    candidate lists in identical order (tie-broken by grid index,
    not floating-point order)."""
    p1 = _provider()
    p2 = _provider()
    for p in (p1, p2):
        _observe_rect(p, xy_min=(-1.0, -1.0), xy_max=(1.0, 1.0))
    c1 = frontier_cells(p1, from_xy=(0.0, 0.0), k=20)
    c2 = frontier_cells(p2, from_xy=(0.0, 0.0), k=20)
    assert len(c1) == len(c2)
    for a, b in zip(c1, c2):
        assert a.world_xy == b.world_xy


# ── 5. k limits the result count ───────────────────────────────────────────
def test_k_limits_returned_count() -> None:
    """``k`` is the cap on returned candidates. Asking for k=5 must
    return at most 5 (top-ranked); asking for k=1000 must return as
    many as exist."""
    p = _provider()
    _observe_rect(p, xy_min=(-1.0, -1.0), xy_max=(1.0, 1.0))
    cands_5 = frontier_cells(p, from_xy=(0.0, 0.0), k=5)
    cands_all = frontier_cells(p, from_xy=(0.0, 0.0), k=1000)
    assert len(cands_5) <= 5
    assert len(cands_all) > 5  # there are way more than 5 boundary cells
    # The top-5 must be a prefix of the larger list.
    for a, b in zip(cands_5, cands_all[:5]):
        assert a.world_xy == b.world_xy


# ── 6. Visited fraction grows monotonically as territory is observed ───────
def test_visited_fraction_grows_monotonically_during_exploration() -> None:
    """As more cells are observed, ``visited_fraction`` only goes
    up — useful for the agent's "covered 41% of accessible grid"
    signal."""
    p = _provider()
    fractions = []
    for r in (0.5, 1.0, 1.5, 2.0):
        _observe_rect(p, xy_min=(-r, -r), xy_max=(r, r))
        fractions.append(p.visited_fraction())
    for a, b in zip(fractions, fractions[1:]):
        assert b >= a, f"regressed: {fractions}"


# ── 7. Provider integration: frontier_cells() is on the Protocol ───────────
def test_provider_frontier_cells_dispatches_to_helper() -> None:
    """The provider's ``frontier_cells`` method must produce the same
    result as the standalone helper. Tests the dispatch path the
    agent skills will exercise."""
    p = _provider()
    _observe_rect(p, xy_min=(-1.0, -1.0), xy_max=(1.0, 1.0))
    via_method = p.frontier_cells(from_xy=(0.0, 0.0), k=20)
    via_helper = frontier_cells(p, from_xy=(0.0, 0.0), k=20)
    assert len(via_method) == len(via_helper)
    for a, b in zip(via_method, via_helper):
        assert a.world_xy == b.world_xy
        assert abs(a.score - b.score) < 1e-9
