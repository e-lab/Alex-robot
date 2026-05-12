"""Variance-aware A* on the occupancy grid (D14.2).

The Phase 1-4 planner ``autonomy.planner.plan_path`` consumes a boolean
``occ`` grid (True = blocked) and runs 8-connected A* with constant
step cost. That's right for USD (no UNKNOWN, no variance) and we keep
it bit-identical for the USD demo.

This module adds a *cost-aware* A* that consumes a callable
``cell_cost(world_xy) → float`` and treats:

    FREE-clean   →  1.0                          (lowest)
    FREE-dirty   →  1.0 + lambda * (σ² - σ²_clean)   (rises with variance,
                                                     capped below UNKNOWN)
    UNKNOWN      →  unknown_cost_multiplier      (~6.0)
    OBSTACLE     →  inf                          (blocked)

The tier ordering is invariant: ``clean < dirty < unknown < obstacle``.
The variance-induced penalty is clamped so dirty FREE never crosses
into UNKNOWN territory; otherwise the planner would sometimes prefer
driving into unknown space over a slightly-noisy seen patch, which is
the opposite of what we want.

This planner is invoked by the agent's ``goto`` / ``goto_xy`` skills
once LA-1 lands. The Phase 1-4 demo continues to call the legacy
binary planner unchanged.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple, Union

import numpy as np

from .base import CellState, WorldXY


@dataclass
class CostParams:
    """Tunable knobs for the four-tier cost (D14.2)."""

    var_cost_threshold: float = 0.01    # σ² below this is "clean free"
    var_cost_lambda: float = 5.0        # variance penalty scale
    unknown_cost_multiplier: float = 6.0   # cost of a single UNKNOWN step


class _ProviderLike(Protocol):
    """The slice of OccupancyProvider that ``PerCellCostProvider`` needs.

    Two methods: classification + variance. Real
    :class:`HeightMapProvider` instances satisfy both; USD wrappers
    satisfy the classification half and return σ²=0 from a default
    ``variance`` method.
    """

    def query(self, world_xy: WorldXY) -> CellState: ...
    def variance(self, world_xy: WorldXY) -> float: ...


class PerCellCostProvider:
    """Adapts a provider into the cost callable A* consumes.

    A single instance gets reused across plan calls; the callable
    ``cell_cost(world_xy)`` is cheap and free of state, so it composes
    with whatever heuristic the planner picks.
    """

    def __init__(self, provider: _ProviderLike, params: CostParams) -> None:
        self._p = provider
        self._params = params

    def cell_cost(self, world_xy: WorldXY) -> float:
        state = self._p.query(world_xy)
        if state == CellState.OBSTACLE:
            return math.inf
        if state == CellState.UNKNOWN:
            return self._params.unknown_cost_multiplier
        # FREE — apply the variance ramp, then clamp below UNKNOWN.
        var = self._p.variance(world_xy)
        if var <= self._params.var_cost_threshold:
            return 1.0
        penalty = self._params.var_cost_lambda * (
            var - self._params.var_cost_threshold
        )
        cost = 1.0 + penalty
        # Clamp: dirty FREE must remain strictly below UNKNOWN so the
        # tier ordering holds and the planner never prefers driving
        # into UNKNOWN over a seen-but-noisy patch.
        cap = self._params.unknown_cost_multiplier - 1e-6
        return min(cost, cap)


@dataclass
class PlanStats:
    """Diagnostic counters returned by ``plan_path_cost(... return_stats=True)``.

    Used by the regression test that bounds expansion count vs the
    Phase 1-4 baseline.
    """

    expansions: int = 0


# 8-connected neighbors with their per-step distance multiplier.
_NEIGHBORS: Tuple[Tuple[int, int, float], ...] = (
    ( 1,  0, 1.0),
    (-1,  0, 1.0),
    ( 0,  1, 1.0),
    ( 0, -1, 1.0),
    ( 1,  1, math.sqrt(2.0)),
    ( 1, -1, math.sqrt(2.0)),
    (-1,  1, math.sqrt(2.0)),
    (-1, -1, math.sqrt(2.0)),
)


def _world_to_grid(xy: WorldXY, origin: WorldXY, res: float) -> Tuple[int, int]:
    return (
        int(math.floor((xy[0] - origin[0]) / res)),
        int(math.floor((xy[1] - origin[1]) / res)),
    )


def _grid_to_world(ix: int, iy: int, origin: WorldXY, res: float) -> WorldXY:
    return (origin[0] + (ix + 0.5) * res, origin[1] + (iy + 0.5) * res)


def plan_path_cost(
    *,
    start_xy: WorldXY,
    goal_xy: WorldXY,
    cost_provider: PerCellCostProvider,
    origin_xy: WorldXY,
    cell_size_m: float,
    width: int,
    height: int,
    return_stats: bool = False,
) -> Union[Optional[List[WorldXY]], Tuple[Optional[List[WorldXY]], PlanStats]]:
    """A* with per-cell cost from :class:`PerCellCostProvider`.

    Returns a list of world-XY waypoints (cell centres), or ``None`` if
    no path exists. When ``return_stats=True`` returns ``(path, stats)``
    instead.

    The heuristic is octile distance (exact for 8-connected unit grid),
    scaled by a *conservative* minimum cost of 1.0 per step so the
    heuristic remains admissible even when actual step costs grow with
    variance.
    """
    stats = PlanStats()

    sx, sy = _world_to_grid(start_xy, origin_xy, cell_size_m)
    gx, gy = _world_to_grid(goal_xy, origin_xy, cell_size_m)
    if not (0 <= sx < width and 0 <= sy < height):
        result = None
        return (result, stats) if return_stats else result
    if not (0 <= gx < width and 0 <= gy < height):
        result = None
        return (result, stats) if return_stats else result

    # Pre-fetch start cost; if start cell is OBSTACLE → no path.
    start_cost = cost_provider.cell_cost(_grid_to_world(sx, sy, origin_xy, cell_size_m))
    if math.isinf(start_cost):
        result = None
        return (result, stats) if return_stats else result

    def h(ix: int, iy: int) -> float:
        # Octile heuristic with minimum step cost = 1.0 (admissible).
        dx, dy = abs(ix - gx), abs(iy - gy)
        return (math.sqrt(2.0) - 1.0) * min(dx, dy) + max(dx, dy)

    open_heap: List[Tuple[float, int, int, int]] = []
    came_from: dict = {}
    g_score: dict = {(sx, sy): 0.0}
    counter = 0
    heapq.heappush(open_heap, (h(sx, sy), counter, sx, sy))

    while open_heap:
        _, _, ix, iy = heapq.heappop(open_heap)
        stats.expansions += 1
        if (ix, iy) == (gx, gy):
            cells = [(ix, iy)]
            while (ix, iy) in came_from:
                ix, iy = came_from[(ix, iy)]
                cells.append((ix, iy))
            cells.reverse()
            path = [_grid_to_world(x, y, origin_xy, cell_size_m) for x, y in cells]
            return (path, stats) if return_stats else path

        for dx, dy, step_dist in _NEIGHBORS:
            jx, jy = ix + dx, iy + dy
            if not (0 <= jx < width and 0 <= jy < height):
                continue
            wx, wy = _grid_to_world(jx, jy, origin_xy, cell_size_m)
            cell_cost = cost_provider.cell_cost((wx, wy))
            if math.isinf(cell_cost):
                continue
            tentative = g_score[(ix, iy)] + step_dist * cell_cost
            if tentative < g_score.get((jx, jy), math.inf):
                g_score[(jx, jy)] = tentative
                came_from[(jx, jy)] = (ix, iy)
                counter += 1
                heapq.heappush(open_heap, (tentative + h(jx, jy), counter, jx, jy))

    result = None
    return (result, stats) if return_stats else result


__all__ = [
    "CostParams",
    "PerCellCostProvider",
    "PlanStats",
    "plan_path_cost",
]
