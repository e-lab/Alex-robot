"""Unit tests for scene_graph.geometry.bbox."""

from __future__ import annotations

import numpy as np
import pytest

from scene_graph.geometry import bbox


class TestPointsToAABB:
    def test_simple(self):
        pts = np.array([[0, 0, 0], [1, 2, 3], [-1, 1, 2]], dtype=np.float32)
        mn, mx = bbox.points_to_aabb(pts)
        assert mn.tolist() == [-1.0, 0.0, 0.0]
        assert mx.tolist() == [1.0, 2.0, 3.0]

    def test_empty_returns_inf(self):
        mn, mx = bbox.points_to_aabb(np.empty((0, 3), dtype=np.float32))
        assert np.all(np.isinf(mn))
        assert np.all(np.isinf(mx))


class TestAABBVolume:
    def test_unit_cube(self):
        mn = np.zeros(3)
        mx = np.ones(3)
        assert bbox.aabb_volume(mn, mx) == pytest.approx(1.0)

    def test_zero_when_inverted(self):
        mn = np.ones(3)
        mx = np.zeros(3)
        assert bbox.aabb_volume(mn, mx) == 0.0


class TestAABBIoU:
    def test_identical_boxes(self):
        mn = np.zeros(3); mx = np.ones(3)
        assert bbox.aabb_iou(mn, mx, mn, mx) == pytest.approx(1.0)

    def test_disjoint_boxes(self):
        a_mn = np.array([0.0, 0, 0]); a_mx = np.array([1.0, 1, 1])
        b_mn = np.array([2.0, 2, 2]); b_mx = np.array([3.0, 3, 3])
        assert bbox.aabb_iou(a_mn, a_mx, b_mn, b_mx) == 0.0

    def test_half_overlap(self):
        # Two unit cubes, second shifted by 0.5 along X: overlap = 0.5×1×1 = 0.5
        # union = 1 + 1 - 0.5 = 1.5 → IoU = 1/3
        a_mn = np.array([0.0, 0, 0]); a_mx = np.array([1.0, 1, 1])
        b_mn = np.array([0.5, 0, 0]); b_mx = np.array([1.5, 1, 1])
        assert bbox.aabb_iou(a_mn, a_mx, b_mn, b_mx) == pytest.approx(1.0 / 3.0)

    def test_contained(self):
        # small box inside big box → IoU = small/big
        a_mn = np.zeros(3); a_mx = 2 * np.ones(3)           # volume 8
        b_mn = np.full(3, 0.5); b_mx = np.full(3, 1.5)      # volume 1
        assert bbox.aabb_iou(a_mn, a_mx, b_mn, b_mx) == pytest.approx(1.0 / 8.0)


class TestAABBMerge:
    def test_union(self):
        a_mn = np.array([-1, 0, 0]); a_mx = np.array([1, 2, 2])
        b_mn = np.array([0, -1, 1]); b_mx = np.array([2, 1, 3])
        mn, mx = bbox.aabb_merge(a_mn, a_mx, b_mn, b_mx)
        assert mn.tolist() == [-1.0, -1.0, 0.0]
        assert mx.tolist() == [2.0, 2.0, 3.0]


class TestAABBCenter:
    def test_unit(self):
        mn = np.zeros(3); mx = np.ones(3)
        assert bbox.aabb_center(mn, mx).tolist() == [0.5, 0.5, 0.5]
