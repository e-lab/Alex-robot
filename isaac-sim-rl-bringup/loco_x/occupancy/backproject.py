"""Depth → world-frame point cloud back-projection (LA-0b.2).

Turns one ``(H, W)`` depth image + pinhole intrinsics + camera pose
into a :class:`PointCloud` in world coordinates suitable for
:meth:`HeightMapProvider.update`. This is the only LA-0b.2 module that
touches real geometry; everything else delegates to the LA-0b.1
fold-in math.

Conventions (matched to Alex's existing head-cam handling in
``alex_onnx_walking_policy.py``):

* World frame: +X forward, +Y left, +Z up. Right-handed.
* Camera optical axis points in **camera-frame +X** when yaw=pitch=0.
  Camera frame is also right-handed (+X forward, +Y left, +Z up).
* ``depth[v, u]`` is the distance along the camera optical axis at
  pixel ``(u, v)`` (the usual depth-camera convention). Invalid
  pixels: NaN, +/-inf, <= 0 — all dropped.
* Yaw rotates camera frame around world +Z. Pitch rotates camera
  frame around the camera's own +Y (after yaw is applied). Roll is
  not modelled — Alex's head only yaws and pitches.

The transform from camera-frame point ``p_c`` to world is::

    p_w = R_yaw @ R_pitch @ p_c + camera_position

with the rotations expanded inline (no scipy dependency).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .synthetic import PointCloud


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics. All values in pixels except width/height."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class CameraPose:
    """Camera pose in world frame.

    ``xy`` is the camera position in the horizontal plane; ``z`` is the
    camera height. ``yaw_rad`` rotates the camera around world +Z;
    ``pitch_rad`` then rotates around the (yawed) camera +Y. Roll is
    not modelled.
    """

    xy: Tuple[float, float] = (0.0, 0.0)
    z: float = 0.0
    yaw_rad: float = 0.0
    pitch_rad: float = 0.0


def depth_to_world_points(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    pose: CameraPose,
    *,
    min_range_m: float = 0.05,
    max_range_m: float = 6.0,
    stride: int = 1,
    timestamp: float = 0.0,
) -> PointCloud:
    """Back-project a depth image into a world-frame :class:`PointCloud`.

    Parameters
    ----------
    depth
        ``(H, W)`` float array. Each pixel is the distance along the
        camera optical axis in metres. NaN / inf / non-positive
        values are dropped.
    intrinsics
        Pinhole intrinsics (fx, fy, cx, cy in pixels).
    pose
        Camera pose in world frame.
    min_range_m, max_range_m
        Valid depth range in metres. Pixels outside are dropped.
        Defaults (0.05, 6.0) reject the robot's own body in close
        range and noisy far-field readings the height-map shouldn't
        ingest.
    stride
        Take every ``stride``-th pixel in each axis. Default 1 (all
        pixels). Higher values trade resolution for speed — useful at
        camera-tick rate.
    timestamp
        Carried onto the resulting :class:`PointCloud` for downstream
        bookkeeping; the height-map provider already takes its own
        ``now`` argument so this is informational.
    """
    if depth.ndim != 2 or depth.shape != (intrinsics.height, intrinsics.width):
        raise ValueError(
            f"depth shape {depth.shape} does not match intrinsics "
            f"({intrinsics.height}, {intrinsics.width})"
        )

    # Pixel grid (optionally strided).
    vs = np.arange(0, intrinsics.height, stride, dtype=np.int64)
    us = np.arange(0, intrinsics.width, stride, dtype=np.int64)
    uu, vv = np.meshgrid(us, vs)            # both shape (H', W')
    d = depth[vv, uu].astype(np.float64)     # shape (H', W')

    # Reject invalid pixels.
    valid = np.isfinite(d) & (d >= min_range_m) & (d <= max_range_m)
    if not np.any(valid):
        return PointCloud(points=np.zeros((0, 3), dtype=np.float64), timestamp=timestamp)

    uu = uu[valid].ravel()
    vv = vv[valid].ravel()
    d = d[valid].ravel()

    # Camera-frame coordinates. Optical axis = +X_c (forward).
    # +Y_c = left in image (so a pixel at u > cx is to the right, i.e.
    # negative Y_c). +Z_c = up.
    x_c = d
    y_c = -(uu - intrinsics.cx) * d / intrinsics.fx
    z_c = -(vv - intrinsics.cy) * d / intrinsics.fy

    # Apply pitch (rotation around camera +Y_c — "left" axis). Convention:
    # pitch > 0 looks up, pitch < 0 looks down. So pitch = -pi/2 must
    # map camera-frame +X (forward) onto world-frame -Z (down).
    #
    #   [X']   [ cosP  -sinP] [X_c]
    #   [Z'] = [ sinP   cosP] [Z_c]
    #
    # Sanity check at pitch = -pi/2: cosP = 0, sinP = -1, so
    #   X' = 0 + Z_c = Z_c        (forward → up component)
    #   Z' = -X_c + 0 = -X_c      (forward → down) ✔
    cp = np.cos(pose.pitch_rad)
    sp = np.sin(pose.pitch_rad)
    x_p = cp * x_c - sp * z_c
    z_p = sp * x_c + cp * z_c
    y_p = y_c

    # Apply yaw (rotation around world +Z) and translation.
    cy_ = np.cos(pose.yaw_rad)
    sy_ = np.sin(pose.yaw_rad)
    x_w = cy_ * x_p - sy_ * y_p + pose.xy[0]
    y_w = sy_ * x_p + cy_ * y_p + pose.xy[1]
    z_w = z_p + pose.z

    pts = np.stack([x_w, y_w, z_w], axis=1)
    return PointCloud(points=pts, timestamp=timestamp)


__all__ = [
    "CameraIntrinsics",
    "CameraPose",
    "depth_to_world_points",
]
