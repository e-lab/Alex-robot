"""isaac-sim-rl-bringup — scene_graph package.

MIT-Hydra-style hierarchical scene graph (Objects → Places → Rooms →
Building), built from RGB + depth + ground-truth pose streamed out of
Isaac Sim. See Plan/scene_graph_plan.md for architecture and phases.

Public re-exports for convenience:
"""

from .graph.node_types import (
    MeshLayer, ObjectNode, WallNode, PlaceNode, RoomNode, BuildingNode,
)
from .graph.scene_graph import SceneGraph
from .graph import serialize

__all__ = [
    "MeshLayer", "ObjectNode", "WallNode",
    "PlaceNode", "RoomNode", "BuildingNode",
    "SceneGraph", "serialize",
]
