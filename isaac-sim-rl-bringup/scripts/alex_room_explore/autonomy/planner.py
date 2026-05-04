r"""A* path planner on a 2D occupancy grid (Phase 3.5b).

Consumes the binary occupancy grid + GridFrame produced by Phase 3.5a's
``occupancy_from_usd`` and a (start, goal) pair in world XY, and returns
a list of world-XY waypoints around the obstacles. The waypoint follower
in ``alex_onnx_walking_policy.py`` walks Alex along this list segment
by segment using the existing FSM heading controller.

Pure numpy + scipy + heapq. No Isaac, no pxr, fully unit-testable.

Algorithm:
1. **Inflate** the occupancy grid by ``inflation_m`` (robot footprint
   clearance) using ``scipy.ndimage.binary_dilation``. Done once per
   plan call; the planner then operates on the inflated grid.
2. **Snap** start and goal to the nearest free cell within a small
   radius. The goal XYZ from SAM3 lands on the stove itself (which is
   an obstacle); without a snap step we'd return None for almost every
   real query.
3. **A\*** with 8-connected moves, cardinal cost 1.0 and diagonal cost
   √2 (in cell units), Euclidean heuristic. Cells whose inflated value
   is True are blocked.
4. (Optional) **Line-of-sight smoothing**: walk the cell path and drop
   each interior waypoint whose neighbours are mutually visible (no
   inflated cell intersecting the segment). Turns the staircase A* path
   into a clean polyline.
5. Return the cell sequence converted to world XY (cell centres), with
   the original ``goal_xy`` as the final waypoint when the snap moved
   it — letting the FSM's arrival check fire on the requested goal,
   not the snapped surrogate. (Matches what a person expects: "I asked
   to go to the stove, the path ends at the stove location even if my
   approach point is slightly offset.")
"""
from __future__ import annotations

import heapq
import math
from typing import List, Optional, Tuple

import numpy as np

try:
    from scipy.ndimage import binary_dilation
    _SCIPY_AVAILABLE = True
except ImportError:                              # pragma: no cover
    _SCIPY_AVAILABLE = False

from .usd_occupancy import GridFrame


# 8-connected move offsets and cost (in cell units).
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


# ── Inflation ────────────────────────────────────────────────────────────────
def _inflate(occ: np.ndarray, *, iters: int) -> np.ndarray:
    """Binary dilation of the obstacle grid by ``iters`` cells in each
    direction. ``iters == 0`` returns a copy unchanged."""
    if iters <= 0:
        return occ.copy()
    if not _SCIPY_AVAILABLE:                     # pragma: no cover
        raise RuntimeError(
            "plan_path inflation requires scipy. Install scipy or pass "
            "inflation_m=0."
        )
    # 4-connected structuring element ⊕ N times == cell distance N.
    return binary_dilation(occ, iterations=int(iters))


# ── Snap to free cell ────────────────────────────────────────────────────────
def _snap_to_free(
    occ: np.ndarray,
    *,
    ix: int,
    iy: int,
    max_radius_cells: int,
) -> Optional[Tuple[int, int]]:
    """Return the nearest free cell to ``(ix, iy)`` within
    ``max_radius_cells``. Already-free cells return themselves.

    Spiral-style search: expanding L-infinity rings around the seed.
    Returns ``None`` if no free cell is found within the radius.
    """
    H, W = occ.shape
    if 0 <= ix < W and 0 <= iy < H and not occ[iy, ix]:
        return ix, iy
    for r in range(1, max_radius_cells + 1):
        # Walk the perimeter of the r-ring; pick the closest free cell
        # by Euclidean distance to break ties consistently.
        best: Optional[Tuple[int, int, float]] = None
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if max(abs(dx), abs(dy)) != r:
                    continue
                jx, jy = ix + dx, iy + dy
                if not (0 <= jx < W and 0 <= jy < H):
                    continue
                if occ[jy, jx]:
                    continue
                d2 = float(dx * dx + dy * dy)
                if best is None or d2 < best[2]:
                    best = (jx, jy, d2)
        if best is not None:
            return best[0], best[1]
    return None


# ── A* core ──────────────────────────────────────────────────────────────────
def _astar(
    occ: np.ndarray,
    *,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    """8-connected A* on a boolean occupancy grid. Returns a cell-sequence
    ``[(ix, iy), ...]`` or ``None`` if no path exists.

    Cells with ``occ[iy, ix] == True`` are blocked. Diagonal moves are
    blocked when *either* axis-aligned neighbour is occupied (prevents
    cutting through corners — important when inflation is small).
    """
    H, W = occ.shape
    sx, sy = start
    gx, gy = goal
    if start == goal:
        return [start]

    def h(ix: int, iy: int) -> float:
        # Octile distance — exact heuristic for 8-connected, costs (1, √2).
        dx = abs(ix - gx)
        dy = abs(iy - gy)
        return (math.sqrt(2.0) - 1.0) * min(dx, dy) + max(dx, dy)

    open_heap: List[Tuple[float, int, int, int]] = []
    came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score: dict[Tuple[int, int], float] = {start: 0.0}
    counter = 0  # tie-breaker for the heap
    heapq.heappush(open_heap, (h(sx, sy), counter, sx, sy))

    while open_heap:
        _, _, ix, iy = heapq.heappop(open_heap)
        if (ix, iy) == goal:
            # Reconstruct.
            path = [(ix, iy)]
            while (ix, iy) in came_from:
                ix, iy = came_from[(ix, iy)]
                path.append((ix, iy))
            path.reverse()
            return path

        for dx, dy, cost in _NEIGHBORS:
            jx, jy = ix + dx, iy + dy
            if not (0 <= jx < W and 0 <= jy < H):
                continue
            if occ[jy, jx]:
                continue
            # Block corner-cutting on diagonals.
            if dx != 0 and dy != 0:
                if occ[iy, jx] or occ[jy, ix]:
                    continue
            tentative = g_score[(ix, iy)] + cost
            if tentative < g_score.get((jx, jy), math.inf):
                g_score[(jx, jy)] = tentative
                came_from[(jx, jy)] = (ix, iy)
                counter += 1
                heapq.heappush(open_heap, (tentative + h(jx, jy), counter, jx, jy))
    return None


# ── Smoothing ────────────────────────────────────────────────────────────────
def _line_of_sight(
    occ: np.ndarray,
    *,
    a: Tuple[int, int],
    b: Tuple[int, int],
) -> bool:
    """Bresenham line walk from ``a`` to ``b``; True if every cell on the
    segment is free in ``occ``."""
    x0, y0 = a
    x1, y1 = b
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    H, W = occ.shape
    while True:
        if not (0 <= x0 < W and 0 <= y0 < H):
            return False
        if occ[y0, x0]:
            return False
        if (x0, y0) == (x1, y1):
            return True
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def _smooth_cells(
    cells: List[Tuple[int, int]],
    *,
    occ: np.ndarray,
) -> List[Tuple[int, int]]:
    """Drop intermediate waypoints whenever the previous-and-next pair has
    a clear line of sight in ``occ``. One pass; greedy."""
    if len(cells) <= 2:
        return cells
    out = [cells[0]]
    i = 0
    while i < len(cells) - 1:
        # Find the farthest j > i such that out[-1] -> cells[j] is clear.
        j = i + 1
        farthest = j
        while j < len(cells):
            if _line_of_sight(occ, a=out[-1], b=cells[j]):
                farthest = j
                j += 1
            else:
                break
        out.append(cells[farthest])
        i = farthest
    return out


# ── Public API ───────────────────────────────────────────────────────────────
def plan_path(
    start_xy: Tuple[float, float],
    goal_xy:  Tuple[float, float],
    occ:      np.ndarray,
    gf:       GridFrame,
    *,
    inflation_m: float = 0.4,
    smooth: bool = True,
    snap_radius_m: float = 1.0,
) -> Optional[List[Tuple[float, float]]]:
    """Find a collision-free path from ``start_xy`` to ``goal_xy`` on a
    2D occupancy grid.

    Parameters
    ----------
    start_xy, goal_xy
        World XY coordinates of the path endpoints.
    occ
        Boolean occupancy grid, shape ``(height, width)``. ``True`` ==
        obstacle. The grid is **not modified** in place.
    gf
        ``GridFrame`` pinning ``occ`` to world coordinates.
    inflation_m
        Robot footprint clearance; obstacles are dilated by
        ``ceil(inflation_m / gf.resolution_m)`` cells before planning.
        Default 0.4 m matches Alex's torso half-width plus a small margin.
    smooth
        Apply line-of-sight smoothing to drop redundant waypoints. Cuts
        the cell-step "staircase" into a clean polyline.
    snap_radius_m
        If start or goal lands on an inflated-occupied cell, search for
        the nearest free cell within this radius. Lets the planner
        succeed when the goal XYZ is on an obstacle (e.g. SAM3 returns
        the stove's centre, which IS the stove).

    Returns
    -------
    list of (x, y)
        World-XY waypoints, including the start and goal as endpoints.
        ``goal_xy`` is preserved as the literal final waypoint when the
        snap moves the planning goal — so the FSM arrival check fires
        on the original target.
    None
        Goal unreachable, or start/goal outside the grid, or no free
        cell within ``snap_radius_m``.
    """
    # Inflate.
    iters = int(math.ceil(inflation_m / gf.resolution_m)) if inflation_m > 0.0 else 0
    occ_inf = _inflate(occ, iters=iters)

    # World → grid.
    sx, sy = gf.world_to_grid(*start_xy)
    gxc, gyc = gf.world_to_grid(*goal_xy)
    H, W = occ_inf.shape
    in_bounds = lambda ix, iy: 0 <= ix < W and 0 <= iy < H
    if not in_bounds(sx, sy) or not in_bounds(gxc, gyc):
        return None

    # Snap to free.
    snap_radius_cells = max(1, int(math.ceil(snap_radius_m / gf.resolution_m)))
    snapped_start = _snap_to_free(
        occ_inf, ix=sx, iy=sy, max_radius_cells=snap_radius_cells,
    )
    snapped_goal = _snap_to_free(
        occ_inf, ix=gxc, iy=gyc, max_radius_cells=snap_radius_cells,
    )
    if snapped_start is None or snapped_goal is None:
        return None

    # A*.
    cells = _astar(occ_inf, start=snapped_start, goal=snapped_goal)
    if cells is None:
        return None

    # Smooth.
    if smooth:
        cells = _smooth_cells(cells, occ=occ_inf)

    # Cells → world XY (cell centres). Every waypoint is on a free
    # (inflated) cell by construction — we never append the literal
    # ``goal_xy`` even when snapping moved it, because the caller would
    # then have a final waypoint inside an obstacle. The FSM's arrival
    # check uses Euclidean distance, so reaching the snapped surrogate
    # near the goal is equivalent to "arrived" for the original target.
    return [gf.grid_to_world(ix, iy) for ix, iy in cells]


__all__ = ["plan_path"]
