"""USD-backed :class:`OccupancyProvider` (LA-0a).

Thin wrapper around the Phase 1-4 rasteriser
(``autonomy.usd_occupancy.occupancy_from_usd``). The provider builds
the static grid once at construction; ``update()`` is a no-op because
the USD is ground truth. ``frontier_cells()`` returns ``[]`` because
nothing is ever UNKNOWN — every cell is FREE or OBSTACLE.

This module exists so the Phase 1-4 demo keeps working unchanged while
LA-0b's :class:`HeightMapProvider` builds beside it on the same
interface. When the Hydra config selects ``occupancy=usd`` the agent,
planner, and watchdog see USD-prior behaviour; selecting
``occupancy=heightmap`` swaps the backend with zero changes upstream.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

# autonomy.usd_occupancy lives under scripts/alex_room_explore/. The
# import is deferred to call time so plain `import loco_x.occupancy`
# succeeds even when the bringup script hasn't put that path on
# sys.path yet (e.g. IDE static analysis, standalone tooling). Tests
# add the path explicitly in tests/loco_x/conftest.py.
try:
    from autonomy.usd_occupancy import GridFrame  # type: ignore[import-not-found]
    _AUTONOMY_ON_PATH = True
except ImportError:                                # pragma: no cover
    GridFrame = None  # type: ignore[assignment,misc]
    _AUTONOMY_ON_PATH = False

from .base import CellState, FrontierCandidate, OccupancyProvider, WorldXY


class UsdOccupancyProvider:
    """USD-rasterised :class:`OccupancyProvider`.

    Construct via :meth:`from_stage`; the constructor accepts a
    pre-built grid + frame so unit tests can inject synthetic data
    without an open USD stage.
    """

    def __init__(self, *, grid: np.ndarray, frame: GridFrame) -> None:
        if grid.shape != (frame.height, frame.width):
            raise ValueError(
                f"grid shape {grid.shape} does not match frame "
                f"({frame.height}, {frame.width})"
            )
        # Convert legacy boolean grid → CellState int8 grid. The
        # provider's contract is integer cell labels; the planner can
        # still recover a boolean view via ``grid == OBSTACLE``.
        if grid.dtype == bool:
            cell_grid = np.where(
                grid, int(CellState.OBSTACLE), int(CellState.FREE)
            ).astype(np.int8)
        else:
            cell_grid = grid.astype(np.int8, copy=True)
        self._grid: np.ndarray = cell_grid
        self._frame: GridFrame = frame

    # ── Construction helper ─────────────────────────────────────────
    @classmethod
    def from_stage(
        cls,
        stage,  # noqa: ANN001 — pxr.Usd.Stage; unannotated to avoid hard pxr dep
        *,
        z_band: Tuple[float, float] = (0.10, 1.50),
        resolution_m: float = 0.05,
        bounds_xy: Optional[Tuple[float, float, float, float]] = None,
        skip_prim_paths=(),
        auto_pad_m: float = 0.5,
        use_collision_api: bool = True,
    ) -> "UsdOccupancyProvider":
        """Rasterise the given ``Usd.Stage`` into a provider.

        Forwards every keyword to :func:`occupancy_from_usd`, so this
        is a *pure rename* of the Phase 1-4 entry point — same
        defaults, same behaviour.
        """
        # Import here (not at module load) so plain `import loco_x.occupancy`
        # works in environments where autonomy/ isn't on sys.path.
        from autonomy.usd_occupancy import occupancy_from_usd  # noqa: WPS433

        legacy_occ, frame = occupancy_from_usd(
            stage,
            z_band=z_band,
            resolution_m=resolution_m,
            bounds_xy=bounds_xy,
            skip_prim_paths=skip_prim_paths,
            auto_pad_m=auto_pad_m,
            use_collision_api=use_collision_api,
        )
        return cls(grid=legacy_occ, frame=frame)

    # ── Planner-facing ──────────────────────────────────────────────
    def grid_for_planner(self) -> np.ndarray:
        """Return the integer :class:`CellState` grid."""
        return self._grid

    def origin_xy(self) -> WorldXY:
        return (self._frame.origin_x, self._frame.origin_y)

    def resolution_m(self) -> float:
        return self._frame.resolution_m

    # Convenience accessor used by Loco-X glue code that wants the full
    # frame (planner needs it too). Not part of the Protocol.
    def grid_frame(self) -> GridFrame:
        return self._frame

    # ── Online update — no-op ──────────────────────────────────────
    def update(self, rgbd, head_cam_pose) -> None:  # noqa: ANN001
        """USD is static ground truth; nothing to do."""
        return None

    # ── Per-cell queries ────────────────────────────────────────────
    def query(self, world_xy: WorldXY) -> CellState:
        ix, iy = self._frame.world_to_grid(*world_xy)
        if not self._frame.in_bounds(ix, iy):
            return CellState.UNKNOWN
        return CellState(int(self._grid[iy, ix]))

    def frontier_cells(
        self,
        from_xy: Optional[WorldXY] = None,
        *,
        k: int = 10,
        prefer_near: Optional[List[str]] = None,
    ) -> List[FrontierCandidate]:
        """No UNKNOWN cells → no frontiers. Always returns ``[]``."""
        return []

    def visited_fraction(self) -> float:
        """USD has no UNKNOWN cells; the formula reduces to
        ``count(FREE) / count(all cells)``."""
        total = float(self._grid.size)
        if total == 0.0:
            return 0.0
        free = float((self._grid == int(CellState.FREE)).sum())
        return free / total


# Static check: the class structurally satisfies the Protocol. We can
# only build the sentinel when autonomy.usd_occupancy is importable —
# unit tests put it on sys.path via conftest, the live runtime via the
# Phase 1-4 bringup script. Without the path it's still a soft contract
# (the tests assert ``isinstance(p, OccupancyProvider)`` at runtime).
if _AUTONOMY_ON_PATH:
    _PROTOCOL_CHECK: OccupancyProvider = UsdOccupancyProvider(  # pragma: no cover
        grid=np.zeros((1, 1), dtype=bool),
        frame=GridFrame(0.0, 0.0, 1.0, 1, 1),  # type: ignore[misc]
    )


__all__ = ["UsdOccupancyProvider"]
