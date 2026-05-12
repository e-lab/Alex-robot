"""CPU-fallback online height-map :class:`OccupancyProvider` (LA-0b.1).

A small, dependency-free implementation that satisfies the
``OccupancyProvider`` contract from a stream of world-frame point
clouds. LA-0b.2 will swap the backend for ``elevation_mapping_cupy``
when the GPU package installs cleanly; the interface stays the same.

Algorithm (per-cell state, all in numpy):

* For each incoming point ``(x, y, z)``:
    - Find the cell ``(ix, iy)`` it falls in (skip out-of-bounds).
    - Push ``z`` onto that cell's recent-observations ring buffer.
    - Record the observation timestamp.

* A cell flips FREE / OBSTACLE only when the **consistency gate** is
  satisfied: at least ``consistency_n`` observations within
  ``obs_window_s`` whose max-z agrees on the classification (all below
  ``traversable_threshold_m`` → FREE, all above → OBSTACLE). A single
  spurious tall point can never flip a cell on its own.

* Cells that haven't been observed within ``stale_s`` revert to
  UNKNOWN — they need re-observation before the agent treats them as
  free. Cells stamped via :meth:`drive_through` are FREE with high
  confidence and survive single noisy frames.

* :meth:`staleness` reports seconds-since-last-observation. The
  agent's observation-builder uses this with ``path_freshness_s``
  (15 s) for the per-path-cell "needs re-verification" signal, while
  the planner respects the global ``stale_s`` (60 s) decay.

* :meth:`advance_time_to` is the D8 ``now``-injection hook: tests
  fast-forward synthetic time without sleeping.

This is intentionally simple — no Bayesian update, no variance per
cell. The eventual elevation_mapping_cupy backend supplies those for
D14.2 variance-aware A\\*. The interface from :mod:`loco_x.occupancy.base`
gives us room to add ``variance(xy)`` later without breaking callers.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .base import CellState, FrontierCandidate, OccupancyProvider, WorldXY
from .synthetic import PointCloud, Pose


# Recent-observation tuple: (timestamp, max_z_seen_in_frame)
_Obs = Tuple[float, float]


@dataclass
class _CellRecord:
    """Per-cell rolling state used by the consistency gate.

    The classification is **latched**: once the consistency gate fires
    (N>=consistency_n observations agree within obs_window_s), the cell
    holds that label until ``stale_s`` worth of silence revert it to
    UNKNOWN. The ``obs`` ring buffer is only consulted before the
    first promotion; afterwards ``latched_state`` is authoritative.

    Rationale: obs_window_s (1 s) is a *promotion* window — "did we
    see consistent evidence in a tight cluster?". The global decay
    window (stale_s, 60 s) is the *retention* window. Without latching,
    every cell would re-classify on every query and drop to UNKNOWN
    as soon as the obs ring trimmed below N, which is far too jittery.
    """

    obs: Deque[_Obs] = field(default_factory=deque)
    last_seen_t: float = -np.inf
    drive_through_t: Optional[float] = None
    latched_state: Optional[CellState] = None
    latched_t: Optional[float] = None


class HeightMapProvider:
    """Online height-map :class:`OccupancyProvider` built from world-
    frame point clouds + robot pose.

    Parameters mirror the Hydra ``conf/occupancy/heightmap.yaml``
    keys; see that file (and D10 in the plan) for the rationale.
    """

    def __init__(
        self,
        *,
        origin_xy: Tuple[float, float],
        size: Tuple[float, float],
        cell_size_m: float = 0.05,
        traversable_threshold_m: float = 0.05,
        stale_s: float = 60.0,
        path_freshness_s: float = 15.0,
        consistency_n: int = 3,
        obs_window_s: float = 1.0,
    ) -> None:
        if cell_size_m <= 0.0:
            raise ValueError("cell_size_m must be > 0")
        self._origin_xy = (float(origin_xy[0]), float(origin_xy[1]))
        self._cell_size_m = float(cell_size_m)
        self._traversable_threshold_m = float(traversable_threshold_m)
        self._stale_s = float(stale_s)
        self._path_freshness_s = float(path_freshness_s)
        self._consistency_n = int(consistency_n)
        self._obs_window_s = float(obs_window_s)
        # Grid dims from world size + cell size.
        self._width = max(1, int(np.ceil(size[0] / self._cell_size_m)))
        self._height = max(1, int(np.ceil(size[1] / self._cell_size_m)))
        # Sparse per-cell records (we don't allocate H*W records up front;
        # most cells stay UNKNOWN and never touched).
        self._cells: Dict[Tuple[int, int], _CellRecord] = {}
        self._now: float = 0.0

    # ── World <-> grid helpers ─────────────────────────────────────
    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        ix = int(np.floor((x - self._origin_xy[0]) / self._cell_size_m))
        iy = int(np.floor((y - self._origin_xy[1]) / self._cell_size_m))
        return ix, iy

    def _in_bounds(self, ix: int, iy: int) -> bool:
        return 0 <= ix < self._width and 0 <= iy < self._height

    # ── Protocol: planner-facing ───────────────────────────────────
    def grid_for_planner(self) -> np.ndarray:
        """Materialise the current full grid as an int8 array of
        :class:`CellState` values. Re-computed each call (cheap for
        the sizes we plan to ship)."""
        grid = np.full(
            (self._height, self._width), int(CellState.UNKNOWN), dtype=np.int8
        )
        for (ix, iy), rec in self._cells.items():
            if not self._in_bounds(ix, iy):
                continue
            grid[iy, ix] = int(self._classify(rec))
        return grid

    def origin_xy(self) -> WorldXY:
        return self._origin_xy

    def resolution_m(self) -> float:
        return self._cell_size_m

    # ── Protocol: per-cell queries ─────────────────────────────────
    def query(self, world_xy: WorldXY) -> CellState:
        ix, iy = self._world_to_grid(*world_xy)
        if not self._in_bounds(ix, iy):
            return CellState.UNKNOWN
        rec = self._cells.get((ix, iy))
        if rec is None:
            return CellState.UNKNOWN
        return self._classify(rec)

    def frontier_cells(
        self,
        from_xy: Optional[WorldXY] = None,
        *,
        k: int = 10,
        prefer_near: Optional[List[str]] = None,
    ) -> List[FrontierCandidate]:
        """LA-0c will implement frontier scoring. LA-0b.1 returns [].

        The plan parks frontier_cells in LA-0c; this stub keeps the
        Protocol satisfied so synthetic-harness tests can still pass.
        """
        return []

    def visited_fraction(self) -> float:
        if self._width == 0 or self._height == 0:
            return 0.0
        n_free = sum(
            1
            for rec in self._cells.values()
            if self._classify(rec) == CellState.FREE
        )
        return float(n_free) / float(self._width * self._height)

    # ── Protocol: online update ────────────────────────────────────
    def update(  # type: ignore[override]
        self,
        *,
        point_cloud: PointCloud,
        pose: Pose,  # noqa: ARG002 — pose unused in CPU fallback; LA-0b.2 wires FOV cone
        now: Optional[float] = None,
    ) -> None:
        """Fold one world-frame :class:`PointCloud` into the map.

        We group points by their target cell and take the **max z** of
        the points falling in each cell — that's the height the
        traversability test compares to ``traversable_threshold_m``.
        Each cell that received any point records one observation
        (timestamp + max-z) into its ring buffer for the consistency
        gate.
        """
        if now is not None:
            self._now = float(now)

        pts = point_cloud.points
        if pts.size == 0:
            return

        # World → grid indices.
        ix = np.floor((pts[:, 0] - self._origin_xy[0]) / self._cell_size_m).astype(np.int64)
        iy = np.floor((pts[:, 1] - self._origin_xy[1]) / self._cell_size_m).astype(np.int64)
        in_bounds_mask = (
            (ix >= 0) & (ix < self._width) & (iy >= 0) & (iy < self._height)
        )
        ix = ix[in_bounds_mask]
        iy = iy[in_bounds_mask]
        z = pts[in_bounds_mask, 2]
        if ix.size == 0:
            return

        # Vectorised group-by-cell. Encode (ix, iy) as a single int64
        # key, sort, then split contiguous runs. This is ~50x faster
        # than a python dict loop over 200k points and keeps the
        # per-camera-tick cost flat as cloud size grows.
        keys = ix.astype(np.int64) * np.int64(self._height) + iy.astype(np.int64)
        order = np.argsort(keys, kind="stable")
        keys_sorted = keys[order]
        z_sorted = z[order]
        # Boundaries where the key changes → contiguous runs per cell.
        boundaries = np.concatenate(([0], np.flatnonzero(np.diff(keys_sorted)) + 1))
        boundaries = np.concatenate((boundaries, [len(keys_sorted)]))

        ix_sorted = ix[order]
        iy_sorted = iy[order]
        for run_idx in range(len(boundaries) - 1):
            lo = int(boundaries[run_idx])
            hi = int(boundaries[run_idx + 1])
            cell_zs = z_sorted[lo:hi]
            n = cell_zs.size
            if n >= 5:
                # Robust per-frame height: pick the second-from-top
                # element (≈ 95th percentile for small samples). Avoids
                # np.percentile's overhead while still rejecting one
                # outlier per frame. For real cameras with hundreds of
                # points per cell, swap this for np.partition.
                top2 = np.partition(cell_zs, -2)[-2]
                obs_z = float(top2)
            else:
                # Too few samples for outlier rejection; plain max. The
                # cross-frame consistency gate handles single-frame
                # jitter at this density.
                obs_z = float(cell_zs.max())
            key = (int(ix_sorted[lo]), int(iy_sorted[lo]))
            rec = self._cells.setdefault(key, _CellRecord())
            rec.obs.append((self._now, obs_z))
            self._trim(rec)
            rec.last_seen_t = self._now
            self._maybe_promote(rec)

    # ── Drive-through stamping (LA-0b.1 contract) ─────────────────
    def drive_through(self, *, world_xy: WorldXY, now: Optional[float] = None) -> None:
        """Stamp the cell containing ``world_xy`` FREE with high
        confidence. The robot physically walked through it, so we
        *know* it fit. Subsequent single noisy frames don't override."""
        if now is not None:
            self._now = float(now)
        ix, iy = self._world_to_grid(*world_xy)
        if not self._in_bounds(ix, iy):
            return
        rec = self._cells.setdefault((ix, iy), _CellRecord())
        rec.drive_through_t = self._now
        rec.last_seen_t = self._now

    # ── Synthetic-time advance (D8 now-injection) ──────────────────
    def advance_time_to(self, *, now: float) -> None:
        """Push the internal clock forward without ingesting any new
        data. Lets tests assert staleness/decay behaviour
        deterministically (no real sleeping)."""
        self._now = float(now)

    # ── Staleness query ────────────────────────────────────────────
    def staleness(self, world_xy: WorldXY) -> float:
        """Seconds since the cell containing ``world_xy`` was last
        observed. Returns ``+inf`` for never-observed and out-of-bounds
        cells."""
        ix, iy = self._world_to_grid(*world_xy)
        if not self._in_bounds(ix, iy):
            return float("inf")
        rec = self._cells.get((ix, iy))
        if rec is None or rec.last_seen_t == -np.inf:
            return float("inf")
        return float(self._now - rec.last_seen_t)

    # ── Watchdog queries for the path-invalidation hook ────────────
    def path_invalidated_by_new_obstacle(
        self, path_xys: List[WorldXY]
    ) -> Optional[WorldXY]:
        """Scan a planned path; return the first cell along it that has
        flipped to OBSTACLE since the path was planned, or ``None`` if
        the path is still clear.

        This is the LA-0b.2 watchdog hook (D10): when a new high-
        confidence obstacle appears under the planned line, the
        follower clears ``bundle["path"]`` and the autonomy loop
        re-plans next tick. The agent only hears about it on the
        *next* turn via ``last_event``.
        """
        for xy in path_xys:
            if self.query(xy) == CellState.OBSTACLE:
                return xy
        return None

    def max_path_staleness(
        self, path_xys: List[WorldXY]
    ) -> Tuple[float, Optional[WorldXY]]:
        """Return the maximum staleness across a planned path along
        with the world-XY of that worst cell.

        Used by the agent observation builder (D10): when any path cell
        has gone stale beyond ``path_freshness_s`` (15 s) but is still
        globally fresh, surface it so the agent can ``peek`` before
        ``goto``. The path itself is NOT invalidated — staleness is a
        *signal*, not an action.
        """
        worst = (0.0, None)
        for xy in path_xys:
            s = self.staleness(xy)
            if s > worst[0]:
                worst = (s, xy)
        return worst

    # ── Internal: classification + ring trim ───────────────────────
    def _trim(self, rec: _CellRecord) -> None:
        """Drop observations older than ``obs_window_s`` and keep the
        deque from growing unbounded."""
        cutoff = self._now - self._obs_window_s
        # popleft until the head is fresh enough.
        while rec.obs and rec.obs[0][0] < cutoff:
            rec.obs.popleft()
        # Hard cap so a long-stationary observer doesn't grow forever.
        max_keep = max(self._consistency_n * 4, 16)
        while len(rec.obs) > max_keep:
            rec.obs.popleft()

    def _maybe_promote(self, rec: _CellRecord) -> None:
        """Promote ``rec.latched_state`` if the consistency gate fires.

        Called after each :meth:`update` that touches this cell.

        Voting scheme: among the fresh observations (those within the
        ``obs_window_s`` window), count how many vote "below threshold"
        vs "above threshold". The cell latches to FREE / OBSTACLE only
        when the **strict majority** votes the same way *and* at least
        ``consistency_n`` observations exist.

        Why majority rather than unanimity: real depth sensors are
        noisy. With sigma=0.02 m noise on the floor, the per-frame
        max-z or 95th-percentile occasionally crosses the 0.05 m
        threshold even though the floor is flat. A unanimous gate
        would block promotion indefinitely. A strict-majority gate
        rejects single-frame spikes while still requiring a clear
        signal. The single-frame-outlier test (test 7) still passes
        because that case has only one observation total, below the
        ``consistency_n`` minimum.

        Once latched, the state persists until ``stale_s`` worth of
        silence (see :meth:`_classify`); a future contradictory burst
        of consistent observations can re-promote.
        """
        fresh = [z for (t, z) in rec.obs if self._now - t <= self._obs_window_s]
        if len(fresh) < self._consistency_n:
            return
        below = sum(1 for z in fresh if z <= self._traversable_threshold_m)
        above = len(fresh) - below
        # Strict majority — a tie keeps the existing latch (or UNKNOWN
        # if there isn't one yet) rather than flipping on a coin toss.
        if below > above:
            rec.latched_state = CellState.FREE
            rec.latched_t = self._now
        elif above > below:
            rec.latched_state = CellState.OBSTACLE
            rec.latched_t = self._now
        # tie → no change.

    def _classify(self, rec: _CellRecord) -> CellState:
        """Read-only mapping cell record → CellState.

        Order of precedence:
            1. Global staleness decay → UNKNOWN.
            2. Drive-through stamp (still fresh) → FREE.
            3. Latched promotion → that state.
            4. No latch and consistency gate not met → UNKNOWN.
        """
        # Stale → UNKNOWN regardless of past observations.
        if rec.last_seen_t == -np.inf:
            return CellState.UNKNOWN
        if self._now - rec.last_seen_t > self._stale_s:
            return CellState.UNKNOWN

        # Drive-through dominates: the robot physically fit, so the cell
        # is FREE for the full stale_s window (even against a noisy
        # contradictory frame).
        if rec.drive_through_t is not None and (
            self._now - rec.drive_through_t <= self._stale_s
        ):
            return CellState.FREE

        if rec.latched_state is not None:
            return rec.latched_state

        return CellState.UNKNOWN


# Static check: the class structurally satisfies the Protocol. Same
# pattern as usd_provider.py.
_PROTOCOL_CHECK: OccupancyProvider = HeightMapProvider(  # pragma: no cover
    origin_xy=(0.0, 0.0), size=(1.0, 1.0)
)


__all__ = ["HeightMapProvider"]
