"""Frontier cells + info-gain ranking (LA-0c, D5).

A frontier cell is a FREE cell adjacent (4-connected) to an UNKNOWN
cell — the edge of the known map where the robot can walk to and
immediately see new territory by re-observing the scene. The agent
calls :func:`frontier_cells` indirectly via the ``next_frontier()``
skill, and the result is also exposed in the agent observation as
``next frontier suggestion: world=(...)``.

Scoring (D5 + D14.1):

    score = info_gain * W_semantic / (1 + travel_distance)

* **info_gain** ~ count of UNKNOWN cells in a forward cone from the
  candidate, capped by an info_gain_max_range. Cheap proxy for "how
  much would I learn from standing here."
* **travel_distance** — Euclidean from ``from_xy`` to the candidate.
* **W_semantic** (D14.1) — bounded multiplier (1.0 by default) that
  D14 implements; for the bare-geometry path it stays 1.0.

Results are sorted descending by score and tie-broken by **grid index**
(`(iy, ix)` lexicographic) so two runs over the same provider
deterministically return the same order.

This implementation works against any :class:`OccupancyProvider` that
exposes a CellState grid + origin + resolution. USD providers return
``[]`` because their grid never contains UNKNOWN by construction.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import CellState, FrontierCandidate, OccupancyProvider, WorldXY


# D14.1 — bounded semantic-boost multiplier.
#
#   W_semantic(cell, anchors) = 1 + alpha * max_a exp(-d(cell, a) / sigma)
#
# Range: [1.0, 1 + alpha]. Alpha defaults to 0.5 → max boost 1.5x.
_DEFAULT_SEMANTIC_BOOST_ALPHA = 0.5
_DEFAULT_SEMANTIC_BOOST_SIGMA_M = 2.0


# Default info-gain cone: cardinal directions, scanned to a fixed
# Euclidean radius. Symmetric so the result doesn't depend on the
# robot's heading — frontier ranking is a global geometric property,
# not a "from here looking forward" one. The agent uses next_frontier()
# as a strategic hint, then the planner picks the route.
_DEFAULT_INFO_GAIN_RANGE_CELLS = 12   # 12 cells = 0.6 m at 5 cm, 2.4 m at 20 cm
_DEFAULT_K = 10


def _world_to_grid(
    world_xy: WorldXY, origin_xy: WorldXY, res: float
) -> Tuple[int, int]:
    ix = int(np.floor((world_xy[0] - origin_xy[0]) / res))
    iy = int(np.floor((world_xy[1] - origin_xy[1]) / res))
    return ix, iy


def _grid_to_world(
    ix: int, iy: int, origin_xy: WorldXY, res: float
) -> WorldXY:
    return (origin_xy[0] + (ix + 0.5) * res, origin_xy[1] + (iy + 0.5) * res)


def _find_boundary_cells(grid: np.ndarray) -> List[Tuple[int, int]]:
    """4-connected scan: return ``(ix, iy)`` of every FREE cell that
    has at least one UNKNOWN neighbor.

    Vectorised via ``np.roll`` so a 200x200 grid takes microseconds
    instead of milliseconds.
    """
    free_mask = grid == int(CellState.FREE)
    unknown_mask = grid == int(CellState.UNKNOWN)
    # Shifted unknown masks → True wherever a neighbor in that direction
    # is UNKNOWN. The borders wrap around with np.roll; we explicitly
    # clear the wrapping edge so a FREE cell at column 0 isn't paired
    # with the rightmost column.
    up = np.roll(unknown_mask, -1, axis=0)
    up[-1, :] = False
    down = np.roll(unknown_mask, 1, axis=0)
    down[0, :] = False
    right = np.roll(unknown_mask, -1, axis=1)
    right[:, -1] = False
    left = np.roll(unknown_mask, 1, axis=1)
    left[:, 0] = False

    has_unknown_neighbor = up | down | right | left
    boundary = free_mask & has_unknown_neighbor
    # ``np.argwhere`` returns (row, col) = (iy, ix); flip to (ix, iy)
    # so callers see the same convention as ``world_to_grid``.
    iy_ix = np.argwhere(boundary)
    return [(int(c), int(r)) for r, c in iy_ix]


def _info_gain(
    grid: np.ndarray,
    cell: Tuple[int, int],
    *,
    radius_cells: int,
) -> int:
    """Count UNKNOWN cells within a square of side ``2*radius+1``
    centred on the candidate. Cheap O(radius²) proxy for ray-cast
    info gain; the agent uses the score for a *ranking*, not for
    absolute path planning, so the proxy is adequate."""
    ix, iy = cell
    H, W = grid.shape
    x0 = max(0, ix - radius_cells)
    x1 = min(W, ix + radius_cells + 1)
    y0 = max(0, iy - radius_cells)
    y1 = min(H, iy + radius_cells + 1)
    window = grid[y0:y1, x0:x1]
    return int((window == int(CellState.UNKNOWN)).sum())


def frontier_cells(
    provider: OccupancyProvider,
    *,
    from_xy: Optional[WorldXY] = None,
    k: int = _DEFAULT_K,
    info_gain_radius_cells: int = _DEFAULT_INFO_GAIN_RANGE_CELLS,
    prefer_near: Optional[List[str]] = None,
    scene_anchors: Optional[Dict[str, List[WorldXY]]] = None,
    semantic_boost_alpha: float = _DEFAULT_SEMANTIC_BOOST_ALPHA,
    semantic_boost_sigma_m: float = _DEFAULT_SEMANTIC_BOOST_SIGMA_M,
) -> List[FrontierCandidate]:
    """Rank candidate frontier cells.

    Parameters
    ----------
    provider
        Any :class:`OccupancyProvider`. USD providers have no UNKNOWN
        cells → returns ``[]``.
    from_xy
        Robot's current world XY, used to compute ``travel_distance``.
        Defaults to ``(0, 0)`` — meaningful only for tests; production
        callers always pass the robot pose.
    k
        Maximum number of candidates to return. The function sorts the
        full candidate set and returns the top-k.
    info_gain_radius_cells
        Half-side of the square window used as the info-gain proxy
        (default 12 cells ≈ 0.6 m at 5 cm resolution, or 2.4 m at 20 cm).
    prefer_near
        D14.1 — list of scene-graph node *labels* that the LLM has
        named as semantic anchors ("microwaves live near countertops"
        → ``prefer_near=["countertop", "stove"]``). The provider does
        not store scene-graph nodes itself; callers must pass
        ``scene_anchors`` mapping label → list of world XYs.
    scene_anchors
        Optional ``{label: [world_xys]}`` map supplying anchor
        positions. Missing labels are silently ignored — graceful
        degradation: the boost falls back to 1.0 when no matches are
        found.
    semantic_boost_alpha
        Max multiplier; ``W_semantic ∈ [1, 1+alpha]``. Default 0.5.
    semantic_boost_sigma_m
        Falloff distance. Distance from candidate to nearest anchor
        passes through ``exp(-d/sigma)`` so close anchors boost
        strongly and far anchors barely.

    Returns
    -------
    List of :class:`FrontierCandidate`, sorted descending by ``score``,
    tie-broken by grid index ``(iy, ix)``.
    """
    grid = provider.grid_for_planner()
    if grid.size == 0:
        return []
    # No UNKNOWN at all → no frontiers (USD case).
    if not (grid == int(CellState.UNKNOWN)).any():
        return []

    origin = provider.origin_xy()
    res = provider.resolution_m()
    rx, ry = from_xy if from_xy is not None else (0.0, 0.0)

    boundary = _find_boundary_cells(grid)
    if not boundary:
        return []

    # D14.1: collect matching anchor positions. Missing labels are
    # silently dropped (graceful degradation: boost falls back to 1.0).
    anchor_xys: List[WorldXY] = []
    if prefer_near and scene_anchors:
        for label in prefer_near:
            anchor_xys.extend(scene_anchors.get(label, []))
    apply_semantic = bool(anchor_xys)

    candidates: List[Tuple[float, int, int, FrontierCandidate]] = []
    for (ix, iy) in boundary:
        wx, wy = _grid_to_world(ix, iy, origin, res)
        info_gain = _info_gain(grid, (ix, iy), radius_cells=info_gain_radius_cells)
        if info_gain <= 0:
            # Boundary detection caught it, but no UNKNOWN within range
            # → not worth visiting.
            continue
        dist = float(np.hypot(wx - rx, wy - ry))

        # D14.1 — W_semantic ∈ [1.0, 1+alpha]. Distance to the *nearest*
        # matching anchor; ``exp(-d/sigma)`` falloff so close anchors
        # boost strongly and far anchors barely.
        w_semantic = 1.0
        if apply_semantic:
            best_d = min(
                math.hypot(wx - ax, wy - ay) for ax, ay in anchor_xys
            )
            w_semantic = 1.0 + semantic_boost_alpha * math.exp(
                -best_d / semantic_boost_sigma_m
            )

        score = float(info_gain) * w_semantic / (1.0 + dist)
        cand = FrontierCandidate(
            world_xy=(wx, wy),
            info_gain=float(info_gain),
            travel_distance=dist,
            score=score,
        )
        # Sort key: descending by score (so negate), then ascending by
        # (iy, ix) for tie-break determinism.
        candidates.append((-score, iy, ix, cand))

    candidates.sort(key=lambda t: (t[0], t[1], t[2]))
    return [t[3] for t in candidates[:k]]


__all__ = ["frontier_cells"]
