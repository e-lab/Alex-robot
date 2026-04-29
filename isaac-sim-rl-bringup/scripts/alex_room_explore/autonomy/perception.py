"""Isaac-coupled head-camera adapters.

The only file in the ``autonomy`` package that touches Isaac / pxr APIs.
Everything else stays pure numpy so it can be unit-tested without the sim.

Shape contract — what the vendored ``scene_graph.pipeline.frame_loop``
expects:

    rgb            : (H, W, 3) uint8       — RGB image
    depth          : (H, W)    float32     — metres along -Z (OpenGL convention)
    K              : (3, 3)    float32     — pinhole intrinsics
    cam_pos        : (3,)      float32     — world XYZ
    cam_quat_wxyz  : (4,)      float32     — scalar-first unit quaternion

The Alex head_cam was created with ``CameraCfg(convention="opengl")`` so the
vendored ``unprojection.pixel_to_world`` (which assumes OpenGL) "just works".
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two scalar-first quaternions: q1 ⊗ q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32)


def _quat_rotate_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate 3-vector ``v`` by scalar-first unit quaternion ``q``."""
    w, x, y, z = q
    qv = np.array([x, y, z], dtype=np.float32)
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def get_head_cam_pose_K(
    head_cam,
    *,
    robot=None,
    body_name: str = "HEAD_LINK",
    cam_offset_pos: "tuple[float, float, float] | None" = None,
    cam_offset_quat_wxyz: "tuple[float, float, float, float] | None" = None,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Return ``(cam_pos, cam_quat_wxyz, K)`` for the Alex head camera.

    Two implementations, picked by what the caller passes:

    1. **Live articulated pose (preferred)** — caller passes ``robot`` plus the
       ``body_name`` of the parent link and the camera's local offset
       (``cam_offset_pos`` + ``cam_offset_quat_wxyz``). We read the current
       world transform of that link off the articulation and compose with the
       offset to get the camera's live world pose. **This is the only path
       that works on humanoid bodies whose camera prim sits under an
       articulated link** — Isaac Lab's ``head_cam.data.pos_w`` returns the
       authored USD prim transform on those configurations and never
       refreshes (verified by logging ``cam_pos`` per camera tick — value
       stayed frozen for 1100+ ticks while the robot rotated 200°). The
       rendering pipeline uses the live pose internally for ray traversal,
       but exposes the static pose on the data buffer.

    2. **Sensor data buffer (fallback)** — when ``robot`` is None, fall back
       to ``head_cam.data.pos_w`` / ``data.quat_w_opengl``. Correct for
       cameras attached to static prims; broken for our humanoid head_cam.

    The unprojection module expects OpenGL convention (camera looks down -Z,
    +Y up) — matches the ``CameraCfg(convention="opengl")`` the head_cam was
    created with.
    """
    K = head_cam.data.intrinsic_matrices[0].cpu().numpy().astype(np.float32)

    if robot is not None and cam_offset_pos is not None and cam_offset_quat_wxyz is not None:
        body_idx_list, _ = robot.find_bodies(body_name)
        if not body_idx_list:
            raise RuntimeError(
                f"get_head_cam_pose_K: body '{body_name}' not found on robot — "
                f"can't compute live camera pose"
            )
        body_idx = body_idx_list[0]
        link_pos = robot.data.body_link_pos_w[0, body_idx].cpu().numpy().astype(np.float32)
        link_quat = robot.data.body_link_quat_w[0, body_idx].cpu().numpy().astype(np.float32)

        offset_pos = np.asarray(cam_offset_pos, dtype=np.float32)
        offset_quat = np.asarray(cam_offset_quat_wxyz, dtype=np.float32)

        cam_pos = link_pos + _quat_rotate_wxyz(link_quat, offset_pos)
        cam_quat_wxyz = _quat_mul_wxyz(link_quat, offset_quat)
        return cam_pos, cam_quat_wxyz, K

    cam_pos = head_cam.data.pos_w[0].cpu().numpy().astype(np.float32)
    cam_quat_wxyz = head_cam.data.quat_w_opengl[0].cpu().numpy().astype(np.float32)
    return cam_pos, cam_quat_wxyz, K


def read_rgb_depth(head_cam) -> "tuple[np.ndarray, np.ndarray] | None":
    """Pull the latest RGB + depth tensors off the head camera.

    Returns ``(rgb_uint8_HxWx3, depth_float32_HxW)`` or ``None`` if the
    camera hasn't produced output yet (first few ticks before sensors are
    warm). All Isaac-tensor → numpy conversion lives here so callers stay
    framework-agnostic.
    """
    output = head_cam.data.output
    if output is None or "rgb" not in output or "distance_to_image_plane" not in output:
        return None

    rgb = output["rgb"][0].cpu().numpy()
    if rgb.shape[-1] == 4:                      # drop alpha if present
        rgb = rgb[..., :3]
    rgb = rgb.astype(np.uint8, copy=False)

    depth = output["distance_to_image_plane"][0].cpu().numpy().astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    # Isaac returns inf for pixels beyond clip-far; clamp so numpy-only
    # consumers don't crash on isfinite checks downstream.
    depth = np.where(np.isfinite(depth), depth, 20.0).astype(np.float32)
    return rgb, depth


__all__ = ["get_head_cam_pose_K", "read_rgb_depth"]
