"""Occupancy provider protocol (D10).

The interface every provider implements. Two concrete impls today:

* :class:`UsdOccupancyProvider` (LA-0a) — wraps the Phase 1-4 USD
  rasteriser. Used when a ground-truth scene USD is available
  (FloorPlan1 demo). ``update()`` is a no-op; ``frontier_cells()``
  returns ``[]``; ``query()`` never returns ``UNKNOWN`` except for
  points outside the grid.
* :class:`HeightMapProvider` (LA-0b) — built online from RGBD +
  head-cam pose. Used for the real-robot / no-prior path. Produces
  FREE / OBSTACLE / UNKNOWN classifications with per-cell timestamps
  and variances.

Both share this Protocol, so the planner, agent skills, and the
path-invalidation watchdog don't care which is in use.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

WorldXY = Tuple[float, float]


class CellState(IntEnum):
    """Integer label per occupancy cell.

    Values are pinned: downstream code (planner cost, observation
    rendering) treats these as ``int8`` array values, not Python enum
    members. A re-order would silently flip semantics — locked in
    ``test_cellstate_enum_values_are_stable``.
    """

    UNKNOWN = 0   # never observed — frontier candidate
    FREE = 1      # observed and traversable
    OBSTACLE = 2  # observed and not traversable


@dataclass(frozen=True)
class FrontierCandidate:
    """One ranked frontier candidate returned by ``frontier_cells``.

    ``score`` already combines info-gain, travel-distance, and (D14.1)
    optional semantic-anchor boost. Callers pick the highest-scoring
    candidate via ``max(candidates, key=lambda c: c.score)`` or take the
    first entry (results are pre-sorted).
    """

    world_xy: WorldXY
    info_gain: float
    travel_distance: float
    score: float


@runtime_checkable
class OccupancyProvider(Protocol):
    """The interface every occupancy backend implements.

    The agent never holds a reference to a concrete provider class — it
    only sees this Protocol via the bundle. That keeps the agent loop
    identical whether we're running the USD-prior demo (sim shortcut)
    or the no-prior height-map build (real-world / no-USD path).
    """

    # ── Planner-facing ──────────────────────────────────────────────
    def grid_for_planner(self) -> np.ndarray:
        """Return the current 2D grid as an ``int8`` array of
        :class:`CellState` values. Shape ``(height, width)``.

        Callers must treat the returned array as read-only — the
        provider may reuse the underlying buffer on the next update.
        """
        ...

    def origin_xy(self) -> WorldXY:
        """World-frame XY of grid cell ``(0, 0)``'s lower-left corner."""
        ...

    def resolution_m(self) -> float:
        """Cell size in metres (uniform in X and Y)."""
        ...

    # ── Online update (no-op for USD; real work for HeightMap) ──────
    def update(self, rgbd, head_cam_pose) -> None:  # noqa: ANN001
        """Fold one RGBD frame + head-cam pose into the map.

        The frame and pose types are deliberately unconstrained at the
        Protocol level so the implementation can pick what's natural
        (dataclasses, torch tensors, raw numpy). The USD provider
        ignores both arguments.
        """
        ...

    # ── Per-cell queries ────────────────────────────────────────────
    def query(self, world_xy: WorldXY) -> CellState:
        """Return the :class:`CellState` of the cell containing
        ``world_xy``. Out-of-bounds returns ``UNKNOWN`` (graceful
        degradation; this method never raises)."""
        ...

    def frontier_cells(
        self,
        from_xy: Optional[WorldXY] = None,
        *,
        k: int = 10,
        prefer_near: Optional[List[str]] = None,
    ) -> List[FrontierCandidate]:
        """Return up to ``k`` ranked frontier candidates.

        A frontier cell is a FREE cell adjacent to an UNKNOWN cell that
        the robot can plausibly reach. Ranked by an info-gain score
        (see :mod:`loco_x.occupancy.frontier` — LA-0c). The optional
        ``prefer_near`` list of scene-graph label strings (D14.1) biases
        scoring toward cells near matching anchor nodes; ``[]`` or
        ``None`` is identity (pure geometric ranking).

        The USD provider returns ``[]`` — its grid has no UNKNOWN cells.
        """
        ...

    def visited_fraction(self) -> float:
        """``count(FREE) / count(all cells)`` ∈ ``[0, 1]``.

        Used as an exploration progress signal (D11.A, D11.D) and as a
        success / give-up criterion in the agent observation.
        """
        ...


__all__ = [
    "CellState",
    "FrontierCandidate",
    "OccupancyProvider",
    "WorldXY",
]
