"""Pinhole unprojection + camera-to-world transform.

Pure numpy — no Isaac / pxr imports here. The driver (pipeline/frame_loop)
collects (K, cam_pos, cam_quat_wxyz, depth[, mask]) and passes them in.

Convention: camera is OpenGL (+X right, +Y up, −Z forward), matching
`CameraCfg(convention="opengl")` in the Isaac bringup scripts. World frame
is right-handed, Z-up.
"""

from __future__ import annotations
from typing import Tuple

import numpy as np


def quat_wxyz_to_rot(q: np.ndarray) -> np.ndarray:
    """Convert a wxyz unit quaternion to a 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def pixel_grid_to_world(
    depth: np.ndarray,
    K: np.ndarray,
    cam_pos: np.ndarray,
    cam_quat_wxyz: np.ndarray,
    stride: int = 1,
    max_depth: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unproject an entire depth image to world-frame points.

    Returns ``(pts_world, us, vs)`` where ``pts_world`` has shape (N, 3) and
    ``us/vs`` are the pixel indices for each point (useful to index into the
    matching RGB image). Points with invalid depth or depth ≥ max_depth are
    dropped.
    """
    if depth.ndim == 3:
        depth = depth[..., 0]
    H, W = depth.shape

    vs = np.arange(0, H, stride)
    us = np.arange(0, W, stride)
    vv, uu = np.meshgrid(vs, us, indexing="ij")
    z = depth[vv, uu]

    mask = np.isfinite(z) & (z > 0.05) & (z < max_depth)
    if not np.any(mask):
        return np.empty((0, 3), dtype=np.float32), np.empty(0, int), np.empty(0, int)

    uu = uu[mask].astype(np.float32)
    vv = vv[mask].astype(np.float32)
    z = z[mask].astype(np.float32)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    # OpenGL camera frame: +X right, +Y up, -Z forward. `z` is range along -Z.
    x_cam = (uu - cx) * z / fx
    y_cam = -(vv - cy) * z / fy
    z_cam = -z
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

    R = quat_wxyz_to_rot(cam_quat_wxyz)
    pts_world = pts_cam @ R.T + cam_pos.astype(np.float32)
    return pts_world, uu.astype(int), vv.astype(int)


def pixel_to_world(
    u: int,
    v: int,
    depth: np.ndarray,
    K: np.ndarray,
    cam_pos: np.ndarray,
    cam_quat_wxyz: np.ndarray,
) -> "np.ndarray | None":
    """Single-pixel unprojection. Returns None on invalid depth."""
    if depth.ndim == 3:
        depth = depth[..., 0]
    H, W = depth.shape
    if not (0 <= u < W and 0 <= v < H):
        return None
    z = float(depth[v, u])
    if not np.isfinite(z) or z <= 0.05 or z > 20.0:
        return None
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (u - cx) * z / fx
    y_cam = -(v - cy) * z / fy
    z_cam = -z
    R = quat_wxyz_to_rot(cam_quat_wxyz)
    return R @ np.array([x_cam, y_cam, z_cam], dtype=np.float32) + cam_pos.astype(np.float32)


__all__ = ["quat_wxyz_to_rot", "pixel_grid_to_world", "pixel_to_world"]
