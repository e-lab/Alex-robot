"""The SceneGraph container.

Holds one dict of nodes per MIT-Hydra layer. All mutation happens through
free functions under `association/`, `layers/`, `pipeline/` — this class is
just storage + tiny convenience helpers.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .node_types import (
    MeshLayer, ObjectNode, WallNode, PlaceNode, RoomNode, BuildingNode,
)


@dataclass
class SceneGraph:
    scene: str = ""                       # scene identifier (e.g. "hallway")
    scan_complete: bool = False
    pose_frames: int = 0
    vocabulary: List[str] = field(default_factory=list)

    # MIT-Hydra layers (bottom → top). mesh stays None until Phase 6 ships.
    mesh:     Optional[MeshLayer]     = None
    objects:  Dict[str, ObjectNode]   = field(default_factory=dict)
    walls:    Dict[str, WallNode]     = field(default_factory=dict)  # Phase 3.5
    places:   Dict[str, PlaceNode]    = field(default_factory=dict)
    rooms:    Dict[str, RoomNode]     = field(default_factory=dict)
    building: Optional[BuildingNode]  = None

    # keep a robot-path trace (pose list) for diagnostics / replay
    robot_path: List[List[float]] = field(default_factory=list)

    # ---- Phase 0 (P0.2) pending-detections bucket ---------------------------
    # Candidates that haven't been observed enough times to promote into
    # `objects`. Keyed by an auto-assigned provisional id ("pending_1").
    # Once `n_observations >= MIN_SIGHTINGS`, promoted to `objects` with
    # a proper label_N id. Mutation lives in association/merge.py.
    pending: Dict[str, ObjectNode] = field(default_factory=dict)

    # ---- id generation ------------------------------------------------------
    _label_counts:   Dict[str, int] = field(default_factory=dict, repr=False)
    _pending_count:  int            = field(default=0, repr=False)

    def new_object_id(self, label: str) -> str:
        n = self._label_counts.get(label, 0) + 1
        self._label_counts[label] = n
        return f"{label}_{n}"

    def new_pending_id(self) -> str:
        """Provisional id for a candidate that hasn't cleared MIN_SIGHTINGS yet."""
        self._pending_count += 1
        return f"pending_{self._pending_count}"

    # ---- summary ------------------------------------------------------------
    def summary(self) -> str:
        by_label: Dict[str, int] = {}
        for obj in self.objects.values():
            by_label[obj.label] = by_label.get(obj.label, 0) + 1
        parts = [f"{v}×{k}" for k, v in sorted(by_label.items())]
        return (
            f"objects={len(self.objects)} ({', '.join(parts) if parts else '—'})  "
            f"walls={len(self.walls)}  places={len(self.places)}  rooms={len(self.rooms)}"
        )


__all__ = ["SceneGraph"]
