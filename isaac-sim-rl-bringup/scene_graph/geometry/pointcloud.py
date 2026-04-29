"""Tiny point-cloud utilities (numpy only). Kept minimal — no Open3D."""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np


def voxel_downsample(points: np.ndarray, voxel_size: float = 0.05) -> np.ndarray:
    """Drop-points-in-same-voxel downsample. Returns (M, 3)."""
    if points.size == 0 or voxel_size <= 0:
        return points
    idx = np.floor(points / voxel_size).astype(np.int64)
    _, keep = np.unique(idx, axis=0, return_index=True)
    return points[np.sort(keep)]


def voxel_downsample_with_rgb(
    points: np.ndarray,
    rgb: np.ndarray,
    voxel_size: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Voxel-downsample (points, rgb) keeping the first hit in each voxel.

    Returns (points_ds, rgb_ds). `rgb` must line up row-for-row with `points`.
    """
    if points.size == 0 or voxel_size <= 0:
        return points, rgb
    idx = np.floor(points / voxel_size).astype(np.int64)
    _, keep = np.unique(idx, axis=0, return_index=True)
    order = np.sort(keep)
    return points[order], rgb[order]


def merge_and_cap(
    old_pts: Optional[np.ndarray],
    old_rgb: Optional[np.ndarray],
    new_pts: np.ndarray,
    new_rgb: np.ndarray,
    voxel_size: float = 0.02,
    max_points: int = 5000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate (old + new) points/RGB, voxel-downsample, and cap at
    `max_points`. Returns (pts, rgb) arrays with matching row count.

    When the cap is tripped, points are uniformly subsampled — preserves
    the spatial distribution rather than biasing toward one end of the
    array.
    """
    if old_pts is None or old_pts.size == 0:
        pts = np.asarray(new_pts, dtype=np.float32)
        rgb = np.asarray(new_rgb, dtype=np.uint8)
    elif new_pts is None or new_pts.size == 0:
        pts = np.asarray(old_pts, dtype=np.float32)
        rgb = np.asarray(old_rgb, dtype=np.uint8)
    else:
        pts = np.concatenate(
            [np.asarray(old_pts, dtype=np.float32),
             np.asarray(new_pts, dtype=np.float32)], axis=0)
        rgb = np.concatenate(
            [np.asarray(old_rgb, dtype=np.uint8),
             np.asarray(new_rgb, dtype=np.uint8)], axis=0)

    pts, rgb = voxel_downsample_with_rgb(pts, rgb, voxel_size=voxel_size)

    n = pts.shape[0]
    if n > max_points:
        # Uniform spaced indices keep spatial coverage better than head-slice.
        keep = np.linspace(0, n - 1, max_points).astype(np.int64)
        pts = pts[keep]
        rgb = rgb[keep]
    return pts, rgb


__all__ = ["voxel_downsample", "voxel_downsample_with_rgb", "merge_and_cap"]
