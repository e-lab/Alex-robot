"""Loco-X occupancy backend glue (LA-0b.2).

Thin facade over the two :class:`OccupancyProvider` impls + the depth
back-projection helper. The autonomy script calls into this facade at
exactly one place (``step_perception``) and one watchdog query
(``maybe_invalidate_path``); the backend handles provider selection
and the "no head-cam wired yet" case gracefully.

Two reasons for the facade:

* It keeps the autonomy script's call site small. Wiring the live
  head-cam to ``HeightMapProvider`` would otherwise leak depth-image
  preprocessing, frame skip logic, drive-through stamping, and pose
  transforms into the script.
* It keeps the test surface pure-Python. The facade is exercised by
  ``tests/loco_x/occupancy/test_backend.py`` with synthetic depth
  images and a stub pose; no Isaac required.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .backproject import CameraIntrinsics, CameraPose, depth_to_world_points
from .base import CellState, OccupancyProvider, WorldXY
from .heightmap_provider import HeightMapProvider
from .synthetic import PointCloud, Pose
from .usd_provider import UsdOccupancyProvider


@dataclass
class WatchdogReport:
    """Per-tick output of :meth:`OccupancyBackend.maybe_invalidate_path`.

    Carries everything the autonomy loop and the agent observation
    builder need to know about the planned path's status:

    * ``blocker_xy`` — first cell along the path that has flipped to
      OBSTACLE since the path was planned (D10 watchdog). ``None`` if
      the path is still clear. When non-None, the follower clears
      ``bundle["path"]`` and the autonomy loop re-plans next tick.
    * ``max_staleness_s`` — oldest staleness across the path. The agent
      observation surfaces this so the LLM can ``peek`` before ``goto``
      when fresh cells are needed.
    * ``stalest_xy`` — world XY of the cell at ``max_staleness_s``.
    """

    blocker_xy: Optional[WorldXY]
    max_staleness_s: float
    stalest_xy: Optional[WorldXY]


class OccupancyBackend:
    """Facade over :class:`OccupancyProvider` impls.

    Construct via :meth:`from_cfg` to pick USD vs. height-map from the
    Hydra config.
    """

    def __init__(
        self,
        provider: OccupancyProvider,
        *,
        intrinsics: Optional[CameraIntrinsics] = None,
        depth_stride: int = 4,
        min_range_m: float = 0.05,
        max_range_m: float = 6.0,
    ) -> None:
        self.provider = provider
        self.intrinsics = intrinsics
        self.depth_stride = int(depth_stride)
        self.min_range_m = float(min_range_m)
        self.max_range_m = float(max_range_m)

    # ── Construction ───────────────────────────────────────────────
    @classmethod
    def from_cfg(cls, cfg, *, stage=None) -> "OccupancyBackend":
        """Pick the provider based on ``cfg.occupancy.provider``.

        ``cfg.occupancy`` is the Hydra-composed occupancy group
        (``loco_x/conf/occupancy/{usd,heightmap}.yaml``). ``stage`` is
        the open ``Usd.Stage`` for the ``"usd"`` provider; it's
        ignored when ``provider == "heightmap"``.
        """
        provider_name = str(cfg.occupancy.provider).lower()
        if provider_name == "usd":
            if stage is None:
                raise ValueError("usd provider requires an open Usd.Stage")
            prov: OccupancyProvider = UsdOccupancyProvider.from_stage(
                stage,
                z_band=tuple(cfg.occupancy.get("z_band", (0.10, 1.50))),
                resolution_m=float(cfg.occupancy.cell_size_m),
                use_collision_api=bool(cfg.occupancy.get("use_collision_api", True)),
            )
            return cls(prov)
        elif provider_name == "heightmap":
            # The grid bounds need to be supplied by the scene config
            # rather than the occupancy config (different rooms have
            # different sizes). Default to a 20x20 m grid centred on
            # the spawn — enough for FloorPlan1.
            bounds = cfg.scene.get("hm_bounds_xy", None)
            if bounds is None:
                size = (20.0, 20.0)
                origin_xy = (-10.0, -10.0)
            else:
                xmin, ymin, xmax, ymax = bounds
                size = (xmax - xmin, ymax - ymin)
                origin_xy = (xmin, ymin)
            prov = HeightMapProvider(
                origin_xy=origin_xy,
                size=size,
                cell_size_m=float(cfg.occupancy.cell_size_m),
                traversable_threshold_m=float(cfg.occupancy.traversable_threshold_m),
                stale_s=float(cfg.occupancy.stale_s),
                path_freshness_s=float(cfg.occupancy.path_freshness_s),
            )
            return cls(prov)
        else:
            raise ValueError(f"unknown occupancy provider: {provider_name}")

    # ── Per-tick perception fold-in ────────────────────────────────
    def step_perception(
        self,
        *,
        depth: Optional[np.ndarray],
        camera_pose: Optional[CameraPose],
        robot_xy: Optional[Tuple[float, float]],
        now: float,
    ) -> None:
        """Fold one camera tick into the provider.

        ``depth``/``camera_pose``/``intrinsics`` are required for the
        height-map provider; USD ignores them. ``robot_xy`` drives the
        drive-through stamp regardless of provider — USD's
        ``drive_through`` is a no-op via the Protocol default, height
        map records "the robot fit here".
        """
        # Drive-through stamp: works for both providers (USD's is no-op).
        if robot_xy is not None and isinstance(self.provider, HeightMapProvider):
            self.provider.drive_through(world_xy=robot_xy, now=now)

        # Depth → world point cloud → height-map update.
        # USD has no update() side-effects, but we call it anyway so the
        # Protocol contract holds end-to-end.
        if isinstance(self.provider, HeightMapProvider):
            if depth is None or camera_pose is None or self.intrinsics is None:
                # Live perception not yet wired; keep ticking the clock
                # so staleness queries advance.
                self.provider.advance_time_to(now=now)
                return
            cloud = depth_to_world_points(
                depth,
                self.intrinsics,
                camera_pose,
                min_range_m=self.min_range_m,
                max_range_m=self.max_range_m,
                stride=self.depth_stride,
                timestamp=now,
            )
            self.provider.update(point_cloud=cloud, pose=Pose(), now=now)
        else:
            # USD provider: nothing to fold in. Still safe to call
            # update() per the Protocol — it's a no-op.
            self.provider.update(None, None)

    # ── Watchdog query ─────────────────────────────────────────────
    def maybe_invalidate_path(
        self, path_xys: Optional[List[WorldXY]]
    ) -> WatchdogReport:
        """Scan the planned path; report blocker + staleness.

        Returns a :class:`WatchdogReport` with both fields populated.
        The caller decides what to do — typically the autonomy script
        clears ``bundle["path"]`` on a non-None blocker, and the agent
        observation builder surfaces the staleness signal.
        """
        if not path_xys:
            return WatchdogReport(
                blocker_xy=None, max_staleness_s=0.0, stalest_xy=None
            )
        # USD provider's query() is FREE/OBSTACLE only — never UNKNOWN
        # for in-bounds cells. Its staleness is always 0 (no observation
        # decay model). HeightMap honours both.
        blocker = None
        for xy in path_xys:
            if self.provider.query(xy) == CellState.OBSTACLE:
                blocker = xy
                break
        if isinstance(self.provider, HeightMapProvider):
            max_age, worst_xy = self.provider.max_path_staleness(path_xys)
        else:
            max_age, worst_xy = 0.0, None
        return WatchdogReport(
            blocker_xy=blocker,
            max_staleness_s=max_age,
            stalest_xy=worst_xy,
        )


__all__ = ["OccupancyBackend", "WatchdogReport"]
