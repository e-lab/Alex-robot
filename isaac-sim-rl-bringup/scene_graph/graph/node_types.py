"""MIT-Hydra layer node dataclasses.

State-only dataclasses — all mutation lives in free functions under
`association/`, `layers/`, `pipeline/`. Matches ConceptGraphs' convention of
`slam_classes.py` + separate `slam/utils.py`.

See `docs/scene_graph_spec.md` for the full written schema contract.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ── Layer 1: Mesh (placeholder, Phase 6) ─────────────────────────────────────
@dataclass
class MeshLayer:
    """Metric-semantic geometry (Hydra Layer 1). Phase 0 leaves this unset;
    Phase 6 will populate `storage` with a path to a sidecar file."""
    voxel_size_m: float = 0.05
    storage: Optional[str] = None          # relative path to .npz / .ply
    n_voxels: int = 0


# ── Layer 2: Objects ─────────────────────────────────────────────────────────
@dataclass
class ObjectNode:
    """One furniture/item instance in the scene.

    Fields mirror the slide-deck schema. `concept_embedding` is the
    mask-pooled SAM 3 vision feature — optional on construction, filled by
    `detection/embeddings.py`.

    `attrs` is the extensibility slot: future phases (VLM captions, size,
    material, colour, etc.) can add fields without touching the schema.
    """
    id: str
    label: str
    position_xyz: List[float]
    bbox_min_xyz: List[float]
    bbox_max_xyz: List[float]
    confidence: float
    n_observations: int = 1
    first_seen_tick: int = 0
    last_seen_tick: int = 0
    concept_embedding: Optional[np.ndarray] = None          # (D,) float32
    source_tracks: List[str] = field(default_factory=list)  # raw obs IDs for debug
    parent_room: Optional[str] = None                       # layer-4 back-ref
    # Best-view RGB crop (Phase 5.5). `best_view_bgr` is the in-memory
    # uint8 (H,W,3) image; not serialised as JSON. `attrs["best_view_*"]`
    # carries the sidecar path + the score at which it was captured so
    # the VLM can fetch the highest-confidence visual confirmation for
    # an object.
    best_view_bgr: Optional[np.ndarray] = None
    # Per-object segmented point cloud (Phase 5.6 — Option B viz). Accumulated
    # across frames from the masked-pixel unprojection in `object_layer`,
    # voxel-downsampled at merge time to stay bounded. Not serialised as
    # JSON — persists as a sidecar `.npz` (see serialize._save_object_point_sidecars).
    points_xyz: Optional[np.ndarray] = None                 # (N, 3) float32 world-frame
    points_rgb: Optional[np.ndarray] = None                 # (N, 3) uint8
    attrs: Dict[str, Any] = field(default_factory=dict)     # open extension slot

    # -- convenience --
    def centroid(self) -> np.ndarray:
        return np.asarray(self.position_xyz, dtype=np.float32)

    def bbox_volume(self) -> float:
        mn = np.asarray(self.bbox_min_xyz)
        mx = np.asarray(self.bbox_max_xyz)
        return float(np.prod(np.maximum(mx - mn, 0.0)))

    def horizontal_extent(self) -> float:
        mn = np.asarray(self.bbox_min_xyz)
        mx = np.asarray(self.bbox_max_xyz)
        ext = mx - mn
        return float(max(ext[0], ext[1]))

    def height(self) -> float:
        return float(self.bbox_max_xyz[2] - self.bbox_min_xyz[2])


# ── Layer 2b: Walls (Phase 3.5, SpatialLM-derived) ───────────────────────────
@dataclass
class WallNode:
    """One planar wall segment, axis-aligned-rectangle by default.

    `ax, ay, bx, by` are the XY endpoints of the wall's base (floor-level).
    `az` is the base elevation (usually 0). The wall extends upward by
    `height_m`, with half-thickness `thickness_m` on either side of the
    ab segment. Phase 3.5 treats walls as first-class occupancy, and
    doors store a back-ref via `attrs["parent_wall"]`.
    """
    id: str
    ax: float
    ay: float
    bx: float
    by: float
    az: float = 0.0
    height_m: float = 2.5
    thickness_m: float = 0.1
    door_ids: List[str] = field(default_factory=list)   # forward refs
    attrs: Dict[str, Any] = field(default_factory=dict)

    def length_m(self) -> float:
        return float(np.hypot(self.bx - self.ax, self.by - self.ay))

    def tangent_xy(self) -> np.ndarray:
        dx = self.bx - self.ax
        dy = self.by - self.ay
        n = float(np.hypot(dx, dy)) + 1e-12
        return np.asarray([dx / n, dy / n], dtype=np.float32)


# ── Layer 3: Places ──────────────────────────────────────────────────────────
@dataclass
class PlaceNode:
    """A free-space waypoint in the traversability graph (Hydra Layer 3)."""
    id: str
    position_xyz: List[float]
    clearance_m: float                                 # distance to nearest obstacle
    neighbors: List[str] = field(default_factory=list)
    parent_room: Optional[str] = None
    attrs: Dict[str, Any] = field(default_factory=dict)


# ── Layer 4: Rooms ───────────────────────────────────────────────────────────
@dataclass
class RoomNode:
    """A semantic region containing objects + places (Hydra Layer 4)."""
    id: str
    label: Optional[str]                               # VLM-assigned in Phase 5
    centroid_xyz: List[float]
    bbox_min_xy: List[float]
    bbox_max_xy: List[float]
    object_ids: List[str] = field(default_factory=list)
    place_ids: List[str] = field(default_factory=list)
    neighbors: List[str] = field(default_factory=list)  # other room IDs
    attrs: Dict[str, Any] = field(default_factory=dict)


# ── Layer 5: Building ────────────────────────────────────────────────────────
@dataclass
class BuildingNode:
    """Root of the hierarchy (Hydra Layer 5)."""
    id: str = "building_1"
    room_ids: List[str] = field(default_factory=list)
    floor: int = 0
    attrs: Dict[str, Any] = field(default_factory=dict)


__all__ = ["MeshLayer", "ObjectNode", "WallNode", "PlaceNode", "RoomNode", "BuildingNode"]
