"""Unit tests for scene_graph.geometry.pointcloud."""

from __future__ import annotations

import numpy as np

from scene_graph.geometry import pointcloud


class TestVoxelDownsample:
    def test_identical_points_collapse_to_one(self):
        pts = np.tile(np.array([[0.1, 0.1, 0.1]], dtype=np.float32), (50, 1))
        out = pointcloud.voxel_downsample(pts, voxel_size=0.5)
        assert out.shape == (1, 3)

    def test_widely_spaced_points_all_kept(self):
        pts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        out = pointcloud.voxel_downsample(pts, voxel_size=0.1)
        assert out.shape == (4, 3)

    def test_empty_input(self):
        out = pointcloud.voxel_downsample(np.empty((0, 3), dtype=np.float32))
        assert out.shape == (0, 3)

    def test_zero_voxel_passes_through(self):
        pts = np.random.RandomState(0).randn(10, 3).astype(np.float32)
        out = pointcloud.voxel_downsample(pts, voxel_size=0.0)
        assert np.array_equal(out, pts)
