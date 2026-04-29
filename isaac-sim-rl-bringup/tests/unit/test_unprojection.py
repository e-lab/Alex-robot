"""Unit tests for scene_graph.geometry.unprojection.

Round-trip invariant: project a known 3D point through a pinhole camera to
(u, v) + depth, then run `pixel_to_world` to get back to 3D world.  With no
pose noise we should recover the original point.
"""

from __future__ import annotations

import numpy as np
import pytest

from scene_graph.geometry import unprojection as up


# Synthetic pinhole camera (640×480, 60° HFOV)
FX = FY = 554.2562
CX, CY = 320.0, 240.0
K = np.array([[FX, 0.0, CX],
              [0.0, FY, CY],
              [0.0, 0.0, 1.0]], dtype=np.float64)


def _project_opengl(p_world: np.ndarray,
                    cam_pos: np.ndarray,
                    cam_quat_wxyz: np.ndarray):
    """Forward project a world-frame point through our OpenGL-style camera.

    Inverse of `pixel_to_world`. Returns (u, v, depth).
    """
    R = up.quat_wxyz_to_rot(cam_quat_wxyz)
    p_cam = R.T @ (p_world - cam_pos)          # world → cam (R is orthogonal)
    x_cam, y_cam, z_cam = p_cam
    # OpenGL: forward = −Z. Depth (scalar in the camera's distance_to_image_plane
    # sense) is −z_cam. Image u/v follow the pinhole projection used in the
    # script.
    z = -z_cam
    u = x_cam * FX / z + CX
    v = -y_cam * FY / z + CY
    return float(u), float(v), float(z)


class TestQuatToRot:
    def test_identity(self):
        R = up.quat_wxyz_to_rot(np.array([1.0, 0.0, 0.0, 0.0]))
        assert np.allclose(R, np.eye(3), atol=1e-6)

    def test_rot_z_90(self):
        # 90° about world Z: x → y, y → -x
        R = up.quat_wxyz_to_rot(np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]))
        v = R @ np.array([1.0, 0.0, 0.0])
        assert np.allclose(v, [0.0, 1.0, 0.0], atol=1e-6)


class TestPixelToWorld:
    def test_identity_camera_roundtrip(self):
        # Camera at origin, looking along world +X. Project and unproject.
        cam_pos = np.array([0.0, 0.0, 0.0])
        cam_q = np.array([1.0, 0.0, 0.0, 0.0])   # identity
        # A point 2 m in front of an identity-camera is at world (0, 0, -2) in
        # OpenGL convention (camera looks along -Z). Pick a point safely
        # inside the frustum.
        p_world = np.array([0.3, -0.2, -2.5])
        u, v, z = _project_opengl(p_world, cam_pos, cam_q)

        # Build a full-depth image with 0 everywhere except at (v, u)
        H, W = 480, 640
        depth = np.full((H, W), np.inf, dtype=np.float32)
        ui, vi = int(round(u)), int(round(v))
        depth[vi, ui] = z

        recovered = up.pixel_to_world(ui, vi, depth, K, cam_pos, cam_q)
        assert recovered is not None
        assert np.allclose(recovered, p_world, atol=0.02)

    def test_returns_none_on_invalid_depth(self):
        depth = np.full((100, 100), np.nan, dtype=np.float32)
        out = up.pixel_to_world(50, 50, depth, K,
                                np.zeros(3), np.array([1, 0, 0, 0]))
        assert out is None


class TestPixelGridToWorld:
    def test_empty_depth_yields_nothing(self):
        depth = np.full((10, 10), -1.0, dtype=np.float32)  # invalid everywhere
        pts, us, vs = up.pixel_grid_to_world(
            depth, K, np.zeros(3), np.array([1, 0, 0, 0]))
        assert pts.shape == (0, 3)
        assert us.size == 0 and vs.size == 0

    def test_single_valid_pixel(self):
        depth = np.full((5, 5), -1.0, dtype=np.float32)
        depth[2, 2] = 1.5
        pts, us, vs = up.pixel_grid_to_world(
            depth, K, np.zeros(3), np.array([1, 0, 0, 0]))
        assert pts.shape == (1, 3)
        assert us.tolist() == [2] and vs.tolist() == [2]
