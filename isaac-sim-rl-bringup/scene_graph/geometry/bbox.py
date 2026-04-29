"""3D axis-aligned bounding box helpers (numpy only)."""

from __future__ import annotations
from typing import Tuple

import numpy as np


def points_to_aabb(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (min_xyz, max_xyz) of an (N, 3) point cloud.

    Empty input returns two arrays of +/- inf so downstream IoU is defined.
    """
    if points.size == 0:
        inf = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
        return inf.copy(), -inf.copy()
    return points.min(axis=0).astype(np.float32), points.max(axis=0).astype(np.float32)


def aabb_volume(mn: np.ndarray, mx: np.ndarray) -> float:
    return float(np.prod(np.maximum(mx - mn, 0.0)))


def aabb_iou(a_min: np.ndarray, a_max: np.ndarray,
             b_min: np.ndarray, b_max: np.ndarray) -> float:
    """Intersection-over-Union of two 3D AABBs. 0 when disjoint."""
    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    inter = aabb_volume(inter_min, inter_max)
    if inter <= 0.0:
        return 0.0
    union = aabb_volume(a_min, a_max) + aabb_volume(b_min, b_max) - inter
    return inter / max(union, 1e-9)


def aabb_merge(a_min: np.ndarray, a_max: np.ndarray,
               b_min: np.ndarray, b_max: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Union AABB of two boxes."""
    return np.minimum(a_min, b_min).astype(np.float32), np.maximum(a_max, b_max).astype(np.float32)


def aabb_center(mn: np.ndarray, mx: np.ndarray) -> np.ndarray:
    return ((mn + mx) * 0.5).astype(np.float32)


__all__ = ["points_to_aabb", "aabb_volume", "aabb_iou", "aabb_merge", "aabb_center"]
