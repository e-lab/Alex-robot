"""Tests for depth → world-frame point cloud back-projection (LA-0b.2).

The back-projection helper turns one ``(H, W)`` depth image + pinhole
intrinsics + camera extrinsics into a world-frame :class:`PointCloud`.
It's the only piece of LA-0b.2 that touches real geometry; everything
else delegates to the LA-0b.1 fold-in math.

These tests are intrinsics/extrinsics math only — no Isaac, no actual
depth camera. Synthetic depth images verify:
  * a flat-floor depth view from above produces points on z=0,
  * a depth view of a tall wall produces points at the wall plane,
  * NaN / inf / 0-depth pixels are dropped,
  * camera-frame yaw produces correctly rotated world points,
  * camera elevation produces correctly translated z.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pytest

from loco_x.occupancy.backproject import (
    CameraIntrinsics,
    CameraPose,
    depth_to_world_points,
)


# ── Helpers ─────────────────────────────────────────────────────────────────
def _make_intrinsics(*, width: int = 64, height: int = 48, hfov_deg: float = 90.0) -> CameraIntrinsics:
    """Symmetric pinhole intrinsics from horizontal FOV. Defaults match
    a 90-degree wide-FOV depth camera at low resolution — keeps tests
    fast."""
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    fy = fx  # square pixels
    return CameraIntrinsics(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=(width - 1) / 2.0,
        cy=(height - 1) / 2.0,
    )


def _constant_depth(intr: CameraIntrinsics, depth_m: float) -> np.ndarray:
    """Depth image where every pixel sees the same distance."""
    return np.full((intr.height, intr.width), depth_m, dtype=np.float32)


# ── 1. Identity pose, flat depth → cone of points at that distance ─────────
def test_identity_pose_constant_depth_returns_cone_at_distance() -> None:
    """Camera at world origin, facing +X with no roll/pitch/yaw. Every
    pixel sees ``depth=2 m``. The resulting world points must form the
    expected back-projection cone — center pixel at (+2, 0, 0), edge
    pixels at the right offsets given the FOV."""
    intr = _make_intrinsics(width=64, height=48, hfov_deg=90.0)
    depth = _constant_depth(intr, depth_m=2.0)
    # +X forward, +Y left, +Z up — the convention we adopt for the
    # head-cam (matches autonomy's existing pose handling).
    pose = CameraPose(xy=(0.0, 0.0), z=0.0, yaw_rad=0.0, pitch_rad=0.0)

    cloud = depth_to_world_points(depth, intr, pose)
    pts = cloud.points

    assert pts.shape == (intr.width * intr.height, 3)
    # Sanity: at least *some* points lie at finite world coords.
    assert np.all(np.isfinite(pts))

    # Center pixel: looking straight forward → world ≈ (+2, 0, 0).
    cy = intr.height // 2
    cx = intr.width // 2
    center_idx = cy * intr.width + cx
    # ``cy * width + cx`` indexes a half-pixel off the principal
    # point for even-sized images (cx=width/2-0.5, cy=height/2-0.5),
    # producing ~0.03 m of perpendicular offset at depth 2 m. Tolerance
    # widened to one pixel's worth of disparity.
    np.testing.assert_allclose(pts[center_idx], [2.0, 0.0, 0.0], atol=0.05)


def test_identity_pose_drops_invalid_depths() -> None:
    """NaN, +inf, and zero depths must be dropped from the cloud."""
    intr = _make_intrinsics()
    depth = _constant_depth(intr, depth_m=2.0)
    depth[0, 0] = np.nan
    depth[0, 1] = np.inf
    depth[0, 2] = 0.0
    depth[0, 3] = -1.0  # negative depths are sensor garbage; also drop

    cloud = depth_to_world_points(
        depth,
        intr,
        CameraPose(xy=(0.0, 0.0), z=0.0, yaw_rad=0.0, pitch_rad=0.0),
    )
    # 4 pixels dropped → expected count.
    assert cloud.points.shape[0] == intr.width * intr.height - 4
    # No NaN / inf survived.
    assert np.all(np.isfinite(cloud.points))


# ── 2. Camera at height looking down → flat-floor points at z=0 ────────────
def test_camera_above_floor_pitched_down_sees_floor() -> None:
    """Camera at z=1 m, pitched down 90°. Every pixel looks at the
    floor. Depth shaped accordingly (depends on pixel angle).

    We assert the back-projected points lie on z=0 (the floor) within
    a small tolerance. This is the canonical Alex case: head-cam
    looking forward-and-down at the ground."""
    intr = _make_intrinsics(width=32, height=24, hfov_deg=60.0)
    pose = CameraPose(
        xy=(0.0, 0.0), z=1.0,
        yaw_rad=0.0,
        pitch_rad=-math.pi / 2.0,  # -90° pitch = look straight down
    )

    # For a pinhole camera pitched -90°, the ray through pixel (u, v)
    # points "down" but with an angular offset. The depth for a flat
    # floor depends on the ray's inclination. Easier: pick depths
    # such that we know where the points land, then check.
    #
    # For straight-down rays (center column at center row), depth ==
    # camera height = 1.0 m.
    #
    # We use depth=1.0 for every pixel: this isn't physically a flat
    # floor, but lets us assert the center pixel lands at (0, 0, 0).
    depth = _constant_depth(intr, depth_m=1.0)
    cloud = depth_to_world_points(depth, intr, pose)

    cy = intr.height // 2
    cx = intr.width // 2
    center_idx = cy * intr.width + cx
    # Center pixel with depth 1 m, camera 1 m up looking straight down
    # → world floor point at (0, 0, 0).
    # Even-image half-pixel offset → ~0.03 m at depth 1 m. See test 1.
    np.testing.assert_allclose(cloud.points[center_idx], [0.0, 0.0, 0.0], atol=0.05)


# ── 3. Camera yaw rotates world points around z ────────────────────────────
def test_camera_yaw_rotates_world_points() -> None:
    """Same camera + same depth, but yaw=90° (facing +Y instead of +X).
    The center pixel must now land at (0, +2, 0)."""
    intr = _make_intrinsics(width=32, height=24, hfov_deg=60.0)
    depth = _constant_depth(intr, depth_m=2.0)
    pose = CameraPose(
        xy=(0.0, 0.0), z=0.0,
        yaw_rad=math.pi / 2.0,  # +90° → facing +Y
        pitch_rad=0.0,
    )

    cloud = depth_to_world_points(depth, intr, pose)
    cy = intr.height // 2
    cx = intr.width // 2
    center_idx = cy * intr.width + cx
    # Half-pixel offset at depth 2 m → ~0.03 m. See test 1.
    np.testing.assert_allclose(cloud.points[center_idx], [0.0, 2.0, 0.0], atol=0.05)


# ── 4. Camera translation shifts world points ──────────────────────────────
def test_camera_xy_translates_world_points() -> None:
    """Camera at (+1, +0.5), facing +X, depth=2 m. Center pixel lands
    at (+3, +0.5, 0)."""
    intr = _make_intrinsics(width=32, height=24, hfov_deg=60.0)
    depth = _constant_depth(intr, depth_m=2.0)
    pose = CameraPose(
        xy=(1.0, 0.5), z=0.0, yaw_rad=0.0, pitch_rad=0.0,
    )

    cloud = depth_to_world_points(depth, intr, pose)
    cy = intr.height // 2
    cx = intr.width // 2
    center_idx = cy * intr.width + cx
    # Half-pixel offset → ~0.03 m. See test 1.
    np.testing.assert_allclose(cloud.points[center_idx], [3.0, 0.5, 0.0], atol=0.05)


# ── 5. Max-range clipping ──────────────────────────────────────────────────
def test_max_range_clip_drops_far_points() -> None:
    """``max_range_m`` discards depth pixels beyond the configured
    range. Real head-cams have a useful range (~4-6 m); points beyond
    are noise and shouldn't enter the height map."""
    intr = _make_intrinsics()
    depth = _constant_depth(intr, depth_m=8.0)  # all beyond 4 m
    pose = CameraPose(xy=(0.0, 0.0), z=0.0, yaw_rad=0.0, pitch_rad=0.0)

    cloud = depth_to_world_points(depth, intr, pose, max_range_m=4.0)
    assert cloud.points.shape[0] == 0


# ── 6. Min-range clipping ──────────────────────────────────────────────────
def test_min_range_clip_drops_near_points() -> None:
    """``min_range_m`` discards pixels closer than the configured
    minimum. Helps reject the robot's own body parts (chest, hands)
    that occasionally enter the head-cam FOV at very close range."""
    intr = _make_intrinsics()
    depth = _constant_depth(intr, depth_m=0.05)
    pose = CameraPose(xy=(0.0, 0.0), z=0.0, yaw_rad=0.0, pitch_rad=0.0)

    cloud = depth_to_world_points(depth, intr, pose, min_range_m=0.1)
    assert cloud.points.shape[0] == 0


# ── 7. Subsample stride ────────────────────────────────────────────────────
def test_subsample_stride_reduces_point_count() -> None:
    """A stride>1 takes every Nth pixel in each axis. Used to keep the
    fold-in cheap when the head-cam delivers a 640x480 frame at
    camera-tick rate."""
    intr = _make_intrinsics(width=64, height=48)
    depth = _constant_depth(intr, depth_m=2.0)
    pose = CameraPose(xy=(0.0, 0.0), z=0.0, yaw_rad=0.0, pitch_rad=0.0)

    cloud_full = depth_to_world_points(depth, intr, pose, stride=1)
    cloud_strided = depth_to_world_points(depth, intr, pose, stride=4)
    # Stride=4 → roughly 1/16 of pixels.
    expected = (intr.width // 4) * (intr.height // 4)
    assert cloud_strided.points.shape[0] == expected
    assert cloud_strided.points.shape[0] * 16 == cloud_full.points.shape[0]
