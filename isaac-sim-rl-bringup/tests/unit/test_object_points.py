"""Unit tests for per-object segmented point-cloud accumulation (Phase 5.6).

Covers three pieces:
  - `geometry.pointcloud.merge_and_cap` — bounded merge with voxel + cap
  - `association.merge.insert_or_merge` — points thread through creation
    and re-observation, with cap enforced
  - `graph.serialize.save` + `get_object_points` — sidecar round-trip
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from scene_graph.graph.scene_graph import SceneGraph
from scene_graph.graph import serialize
from scene_graph.geometry.pointcloud import merge_and_cap
from scene_graph.association.merge import insert_or_merge


class TestMergeAndCap:
    def test_first_call_copies_new(self):
        new_pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        new_rgb = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
        pts, rgb = merge_and_cap(None, None, new_pts, new_rgb,
                                 voxel_size=0.5, max_points=10)
        assert pts.shape == (2, 3)
        assert rgb.shape == (2, 3)
        np.testing.assert_allclose(sorted(pts.ravel()), sorted(new_pts.ravel()))

    def test_voxel_downsample_deduplicates(self):
        # Three points, two of them in the same 0.5 m voxel.
        new_pts = np.array([
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1],   # same voxel as row 0 at voxel=0.5
            [1.0, 1.0, 1.0],
        ], dtype=np.float32)
        new_rgb = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
        pts, rgb = merge_and_cap(None, None, new_pts, new_rgb,
                                 voxel_size=0.5, max_points=100)
        assert pts.shape[0] == 2  # one voxel collapsed
        assert rgb.shape[0] == 2

    def test_cap_enforced(self):
        rng = np.random.default_rng(42)
        n_in = 1000
        new_pts = rng.uniform(-5, 5, size=(n_in, 3)).astype(np.float32)
        new_rgb = rng.integers(0, 256, size=(n_in, 3), dtype=np.uint8)
        pts, rgb = merge_and_cap(None, None, new_pts, new_rgb,
                                 voxel_size=0.001, max_points=50)
        # voxel=0.001 keeps everything unique; cap enforces ≤ 50.
        assert pts.shape[0] == 50
        assert rgb.shape[0] == 50

    def test_second_call_concatenates(self):
        a = np.array([[0, 0, 0]], dtype=np.float32)
        b = np.array([[10, 10, 10]], dtype=np.float32)
        a_rgb = np.array([[1, 1, 1]], dtype=np.uint8)
        b_rgb = np.array([[2, 2, 2]], dtype=np.uint8)
        pts, rgb = merge_and_cap(a, a_rgb, b, b_rgb,
                                 voxel_size=0.5, max_points=100)
        assert pts.shape == (2, 3)
        assert rgb.shape == (2, 3)


class TestInsertOrMergeWithPoints:
    def _fresh_sg(self) -> SceneGraph:
        return SceneGraph(scene="unit_test")

    def test_brand_new_object_stores_points(self):
        sg = self._fresh_sg()
        pts = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float32)
        rgb = np.array([[200, 0, 0], [0, 200, 0]], dtype=np.uint8)
        obj_id = insert_or_merge(
            sg,
            det_label="chair",
            det_xyz=np.array([0.05, 0, 0], dtype=np.float32),
            det_bbox_min=np.array([-0.1, -0.1, -0.1], dtype=np.float32),
            det_bbox_max=np.array([0.2, 0.1, 0.1], dtype=np.float32),
            det_score=0.9,
            det_embedding=None,
            det_track_id=None,
            tick=0,
            det_points_xyz=pts,
            det_points_rgb=rgb,
        )
        bucket = sg.objects if obj_id in sg.objects else sg.pending
        node = bucket[obj_id]
        assert node.points_xyz is not None
        assert node.points_rgb is not None
        assert node.points_xyz.shape[1] == 3
        assert node.points_rgb.shape[1] == 3
        assert node.points_xyz.shape[0] == node.points_rgb.shape[0]

    def test_reobservation_accumulates(self):
        sg = self._fresh_sg()
        common_kwargs = dict(
            det_label="chair",
            det_bbox_min=np.array([-0.1, -0.1, -0.1], dtype=np.float32),
            det_bbox_max=np.array([0.2, 0.1, 0.1], dtype=np.float32),
            det_score=0.9,
            det_embedding=None,
            det_track_id=None,
        )
        # Two detections with points 10m apart ensure voxel downsample keeps
        # both rather than collapsing them.
        id1 = insert_or_merge(
            sg, det_xyz=np.array([0.0, 0, 0], dtype=np.float32), tick=0,
            det_points_xyz=np.array([[0, 0, 0]], dtype=np.float32),
            det_points_rgb=np.array([[1, 1, 1]], dtype=np.uint8),
            **common_kwargs,
        )
        id2 = insert_or_merge(
            sg, det_xyz=np.array([0.05, 0, 0], dtype=np.float32), tick=1,
            det_points_xyz=np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
            det_points_rgb=np.array([[2, 2, 2]], dtype=np.uint8),
            **common_kwargs,
        )
        assert id1 == id2
        node = sg.objects.get(id1) or sg.pending[id1]
        assert node.points_xyz.shape[0] >= 2  # both survived

    def test_cap_survives_huge_detection(self):
        sg = self._fresh_sg()
        rng = np.random.default_rng(0)
        huge_pts = rng.uniform(0, 5, (20000, 3)).astype(np.float32)
        huge_rgb = rng.integers(0, 256, (20000, 3), dtype=np.uint8)
        obj_id = insert_or_merge(
            sg,
            det_label="chair",
            det_xyz=np.array([2.5, 2.5, 2.5], dtype=np.float32),
            det_bbox_min=np.array([0, 0, 0], dtype=np.float32),
            det_bbox_max=np.array([5, 5, 5], dtype=np.float32),
            det_score=0.9,
            det_embedding=None,
            det_track_id=None,
            tick=0,
            det_points_xyz=huge_pts,
            det_points_rgb=huge_rgb,
        )
        node = sg.objects.get(obj_id) or sg.pending[obj_id]
        assert node.points_xyz.shape[0] <= 5000  # default cap

    def test_none_points_is_noop(self):
        sg = self._fresh_sg()
        obj_id = insert_or_merge(
            sg,
            det_label="chair",
            det_xyz=np.array([0, 0, 0], dtype=np.float32),
            det_bbox_min=np.array([-0.1, -0.1, -0.1], dtype=np.float32),
            det_bbox_max=np.array([0.1, 0.1, 0.1], dtype=np.float32),
            det_score=0.9,
            det_embedding=None,
            det_track_id=None,
            tick=0,
            det_points_xyz=None,
            det_points_rgb=None,
        )
        node = sg.objects.get(obj_id) or sg.pending[obj_id]
        assert node.points_xyz is None
        assert node.points_rgb is None


class TestSidecarRoundTrip:
    def _scene_with_points(self):
        from scene_graph.graph.node_types import ObjectNode
        sg = SceneGraph(scene="t")
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        rgb = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8)
        sg.objects["chair_1"] = ObjectNode(
            id="chair_1", label="chair",
            position_xyz=[0.33, 0.33, 0.0],
            bbox_min_xyz=[0.0, 0.0, 0.0],
            bbox_max_xyz=[1.0, 1.0, 0.0],
            confidence=0.9,
            points_xyz=pts, points_rgb=rgb,
        )
        return sg

    def test_sidecar_written_and_json_points_to_it(self, tmp_path: pathlib.Path):
        sg = self._scene_with_points()
        out = tmp_path / "g.json"
        serialize.save(sg, str(out))
        sidecar = tmp_path / "g_objects.npz"
        assert sidecar.exists()
        data = json.loads(out.read_text())
        attrs = data["layers"]["objects"]["chair_1"]["attrs"]
        assert attrs["points_path"] == "g_objects.npz"
        assert len(attrs["points_npz_keys"]) == 2
        # Raw point arrays must NOT be in the JSON.
        assert "points_xyz" not in data["layers"]["objects"]["chair_1"]
        assert "points_rgb" not in data["layers"]["objects"]["chair_1"]

    def test_get_object_points_round_trip(self, tmp_path: pathlib.Path):
        sg = self._scene_with_points()
        out = tmp_path / "g.json"
        serialize.save(sg, str(out))
        # Reload the SG to drop the in-memory points_xyz.
        sg2 = serialize.load(str(out))
        sg2.objects["chair_1"].points_xyz = None
        sg2.objects["chair_1"].points_rgb = None
        result = serialize.get_object_points(sg2, "chair_1", str(out))
        assert result is not None
        pts, rgb = result
        assert pts.shape == (3, 3)
        assert rgb.shape == (3, 3)
        np.testing.assert_array_equal(rgb[0], [10, 20, 30])

    def test_no_points_means_no_sidecar_no_attrs(self,
                                                 tmp_path: pathlib.Path):
        from scene_graph.graph.node_types import ObjectNode
        sg = SceneGraph(scene="t")
        sg.objects["chair_1"] = ObjectNode(
            id="chair_1", label="chair",
            position_xyz=[0, 0, 0],
            bbox_min_xyz=[0, 0, 0], bbox_max_xyz=[1, 1, 1],
            confidence=0.9,
        )
        out = tmp_path / "g.json"
        serialize.save(sg, str(out))
        assert not (tmp_path / "g_objects.npz").exists()
        data = json.loads(out.read_text())
        attrs = data["layers"]["objects"]["chair_1"]["attrs"]
        assert "points_path" not in attrs
