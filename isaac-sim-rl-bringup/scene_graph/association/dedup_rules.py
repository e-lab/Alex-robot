"""Per-label dedup thresholds. Plain data — no behaviour here.

Phase 0 will consume these in `association/merge.py`. Hydra config can
override `DEFAULT_MERGE_RADIUS_M` + `PER_LABEL_MERGE_RADIUS_M` later (see
Plan/scene_graph_plan.md §10 `configs/association/defaults.yaml`).
"""

from __future__ import annotations

# Default merge radius when a label isn't in the per-label table below.
DEFAULT_MERGE_RADIUS_M: float = 0.5

# Large things need a bigger radius (centroid drifts further when you see
# only part of the object), small things need a smaller radius.
PER_LABEL_MERGE_RADIUS_M = {
    "door":         1.5,
    "wall":         2.0,
    "window":       1.0,
    "sofa":         1.0,
    "couch":        1.0,
    "bed":          1.5,
    "table":        1.0,
    "dining table": 1.2,
    "coffee table": 0.8,
    "desk":         1.0,
    "fridge":       0.8,
    "oven":         0.8,
    "cabinet":      1.0,
    "armchair":     1.0,
    "chair":        1.0,
    "stool":        0.6,
    "plant":        0.4,
    "bottle":       0.2,
    "cup":          0.2,
    "book":         0.2,
}

# Cross-label merge distance — if two candidates of ANY label sit within
# this radius, collapse to the highest-confidence label.
CROSS_LABEL_MERGE_RADIUS_M: float = 0.3

# Visual similarity threshold. When embedding is available, this must also
# be exceeded before we merge two candidates that aren't geometrically
# overlapping but are close.
EMBEDDING_COSINE_THRESHOLD: float = 0.85

# Geometric merge threshold — 3D bbox IoU. Merge when exceeded.
BBOX_IOU_THRESHOLD: float = 0.30

# Minimum observations before a tracked candidate is promoted to a graph
# node (drops one-frame flashes of noise).
MIN_SIGHTINGS: int = 3

# Max per-axis extent of any single object (metres). Absorbing a detection
# that would push the bbox past this cap is rejected — the merge is almost
# certainly wrong. Real doors are not 5 m wide; real chairs are not 3 m
# across. Caps match the widest conceivable real-world instance for each
# class.
DEFAULT_MAX_EXTENT_M: float = 2.0

PER_LABEL_MAX_EXTENT_M = {
    "door":         1.5,
    "window":       2.5,
    "wall":         10.0,      # walls can be long
    "ceiling":      10.0,
    "floor":        10.0,
    "sofa":         3.0,
    "couch":        3.0,
    "bed":          3.0,
    "table":        3.0,
    "dining table": 3.0,
    "coffee table": 2.2,
    "desk":         2.5,
    "fridge":       1.5,
    "oven":         1.5,
    "cabinet":      2.5,
    "armchair":     1.8,
    "chair":        1.5,
    "stool":        0.8,
    "plant":        1.2,
    "bottle":       0.3,
    "cup":          0.3,
    "book":         0.5,
}


def merge_radius(label: str) -> float:
    return PER_LABEL_MERGE_RADIUS_M.get(label, DEFAULT_MERGE_RADIUS_M)


def max_extent(label: str) -> float:
    return PER_LABEL_MAX_EXTENT_M.get(label, DEFAULT_MAX_EXTENT_M)


__all__ = [
    "DEFAULT_MERGE_RADIUS_M",
    "PER_LABEL_MERGE_RADIUS_M",
    "CROSS_LABEL_MERGE_RADIUS_M",
    "EMBEDDING_COSINE_THRESHOLD",
    "BBOX_IOU_THRESHOLD",
    "MIN_SIGHTINGS",
    "DEFAULT_MAX_EXTENT_M",
    "PER_LABEL_MAX_EXTENT_M",
    "merge_radius",
    "max_extent",
]
