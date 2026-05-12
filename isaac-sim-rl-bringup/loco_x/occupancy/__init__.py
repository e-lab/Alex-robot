"""Loco-X occupancy providers (D10).

See :mod:`loco_x.occupancy.base` for the Protocol; concrete impls live
beside it. The agent and planner depend on the Protocol, never on a
concrete class.
"""
from .backend import OccupancyBackend, WatchdogReport
from .backproject import CameraIntrinsics, CameraPose, depth_to_world_points
from .base import CellState, FrontierCandidate, OccupancyProvider, WorldXY
from .frontier import frontier_cells
from .heightmap_provider import HeightMapProvider
from .planner_cost import CostParams, PerCellCostProvider, PlanStats, plan_path_cost
from .usd_provider import UsdOccupancyProvider

__all__ = [
    "CameraIntrinsics",
    "CameraPose",
    "CellState",
    "CostParams",
    "FrontierCandidate",
    "HeightMapProvider",
    "OccupancyBackend",
    "OccupancyProvider",
    "PerCellCostProvider",
    "PlanStats",
    "UsdOccupancyProvider",
    "WatchdogReport",
    "WorldXY",
    "depth_to_world_points",
    "frontier_cells",
    "plan_path_cost",
]
