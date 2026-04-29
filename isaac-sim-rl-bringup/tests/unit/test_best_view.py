"""Unit tests for Phase 5.5 — best-view RGB per object."""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from scene_graph import SceneGraph, ObjectNode, serialize
from scene_graph.detection.sam3_detector import RawDetection
from scene_graph.layers import object_layer


def _mk_det(label: str, bbox_xyxy, score: float,
            mask_shape=(480, 640)):
    """Build a RawDetection with a minimal mask that matches bbox_xyxy."""
    mask = np.zeros(mask_shape, dtype=bool)
    x0, y0, x1, y1 = map(int, bbox_xyxy)
    mask[y0:y1, x0:x1] = True
    return RawDetection(
        label=label,
        score=score,
        mask=mask,
        bbox_xyxy=np.asarray(bbox_xyxy, dtype=np.float32),
    )


class TestBestViewSnapshot:
    def test_first_detection_stores_crop(self):
        sg = SceneGraph()
        sg.objects["chair_1"] = ObjectNode(
            id="chair_1", label="chair",
            position_xyz=[0, 0, 0.5],
            bbox_min_xyz=[-0.3, -0.3, 0], bbox_max_xyz=[0.3, 0.3, 1],
            confidence=0.7, n_observations=4,
        )
        rgb = (np.random.default_rng(0).integers(0, 255, (480, 640, 3))
               .astype(np.uint8))
        d = _mk_det("chair", (100, 100, 180, 200), score=0.7)
        object_layer._update_best_view(sg, "chair_1", rgb, d, tick=10)
        obj = sg.objects["chair_1"]
        assert obj.best_view_bgr is not None
        # Expect a (H, W, 3) crop ≥ the raw bbox, with 8 px margin.
        assert obj.best_view_bgr.shape[2] == 3
        assert obj.attrs["best_view_score"] == pytest.approx(0.7)
        assert obj.attrs["best_view_tick"] == 10

    def test_higher_score_replaces(self):
        sg = SceneGraph()
        sg.objects["chair_1"] = ObjectNode(
            id="chair_1", label="chair",
            position_xyz=[0, 0, 0.5],
            bbox_min_xyz=[-0.3, -0.3, 0], bbox_max_xyz=[0.3, 0.3, 1],
            confidence=0.9, n_observations=4,
        )
        rgb = np.full((480, 640, 3), 10, dtype=np.uint8)
        object_layer._update_best_view(sg, "chair_1", rgb,
                                       _mk_det("chair", (10, 10, 50, 60), 0.6),
                                       tick=0)
        first = sg.objects["chair_1"].best_view_bgr.copy()
        rgb[:] = 200   # newer frame, visibly different
        object_layer._update_best_view(sg, "chair_1", rgb,
                                       _mk_det("chair", (0, 0, 40, 40), 0.9),
                                       tick=20)
        new = sg.objects["chair_1"].best_view_bgr
        assert not np.array_equal(first, new), "higher score should overwrite"
        assert sg.objects["chair_1"].attrs["best_view_score"] == pytest.approx(0.9)
        assert sg.objects["chair_1"].attrs["best_view_tick"] == 20

    def test_lower_score_does_not_replace(self):
        sg = SceneGraph()
        sg.objects["chair_1"] = ObjectNode(
            id="chair_1", label="chair",
            position_xyz=[0, 0, 0.5],
            bbox_min_xyz=[-0.3, -0.3, 0], bbox_max_xyz=[0.3, 0.3, 1],
            confidence=0.9, n_observations=4,
        )
        rgb = np.full((480, 640, 3), 50, dtype=np.uint8)
        object_layer._update_best_view(sg, "chair_1", rgb,
                                       _mk_det("chair", (10, 10, 50, 60), 0.9),
                                       tick=5)
        stored = sg.objects["chair_1"].best_view_bgr.copy()
        rgb[:] = 200
        object_layer._update_best_view(sg, "chair_1", rgb,
                                       _mk_det("chair", (0, 0, 40, 40), 0.3),
                                       tick=50)
        assert np.array_equal(sg.objects["chair_1"].best_view_bgr, stored)
        assert sg.objects["chair_1"].attrs["best_view_score"] == pytest.approx(0.9)

    def test_pending_candidate_gets_snapshot(self):
        sg = SceneGraph()
        # Put a candidate in pending (not promoted).
        sg.pending["pending_1"] = ObjectNode(
            id="pending_1", label="chair",
            position_xyz=[0, 0, 0.5],
            bbox_min_xyz=[-0.3, -0.3, 0], bbox_max_xyz=[0.3, 0.3, 1],
            confidence=0.7, n_observations=1,
        )
        rgb = np.full((480, 640, 3), 40, dtype=np.uint8)
        object_layer._update_best_view(sg, "pending_1", rgb,
                                       _mk_det("chair", (10, 10, 40, 40), 0.8),
                                       tick=2)
        assert sg.pending["pending_1"].best_view_bgr is not None
        assert sg.pending["pending_1"].attrs["best_view_score"] == pytest.approx(0.8)

    def test_degenerate_bbox_skipped(self):
        sg = SceneGraph()
        sg.objects["chair_1"] = ObjectNode(
            id="chair_1", label="chair",
            position_xyz=[0, 0, 0.5],
            bbox_min_xyz=[-0.3, -0.3, 0], bbox_max_xyz=[0.3, 0.3, 1],
            confidence=0.8, n_observations=3,
        )
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        # bbox outside image → no crop
        object_layer._update_best_view(sg, "chair_1", rgb,
                                       _mk_det("chair", (700, 700, 750, 750), 0.9),
                                       tick=0)
        assert sg.objects["chair_1"].best_view_bgr is None


class TestSidecarSaveLoad:
    def _scene_with_crop(self, crop):
        sg = SceneGraph(scene="t")
        obj = ObjectNode(
            id="chair_1", label="chair",
            position_xyz=[0, 0, 0.5],
            bbox_min_xyz=[-0.3, -0.3, 0], bbox_max_xyz=[0.3, 0.3, 1],
            confidence=0.9, n_observations=4,
            attrs={"best_view_score": 0.9, "best_view_tick": 5},
        )
        obj.best_view_bgr = crop
        sg.objects["chair_1"] = obj
        return sg

    def test_save_creates_sidecar_png(self, tmp_path: pathlib.Path):
        crop = (np.random.default_rng(3).integers(0, 255, (40, 60, 3))
                .astype(np.uint8))
        sg = self._scene_with_crop(crop)
        out_json = tmp_path / "graph.json"
        serialize.save(sg, str(out_json))
        # Sidecar directory <stem>_objects/<oid>.png exists and is a PNG.
        sidecar = tmp_path / "graph_objects" / "chair_1.png"
        assert sidecar.exists()
        from PIL import Image
        loaded = np.asarray(Image.open(str(sidecar)).convert("RGB"))
        assert loaded.shape == crop.shape

    def test_best_view_path_recorded_in_json(self, tmp_path: pathlib.Path):
        crop = np.zeros((30, 30, 3), dtype=np.uint8)
        sg = self._scene_with_crop(crop)
        out_json = tmp_path / "graph.json"
        serialize.save(sg, str(out_json))
        raw = json.loads(out_json.read_text())
        attrs = raw["layers"]["objects"]["chair_1"]["attrs"]
        assert attrs["best_view_path"] == "graph_objects/chair_1.png"
        # schema version tracks serialize.SCHEMA_VERSION (bumped when new
        # sidecar types land — currently 4 for object-points sidecar).
        assert raw["meta"]["schema_version"] == serialize.SCHEMA_VERSION

    def test_raw_pixels_never_leak_into_json(self, tmp_path: pathlib.Path):
        crop = np.ones((25, 25, 3), dtype=np.uint8)
        sg = self._scene_with_crop(crop)
        out_json = tmp_path / "graph.json"
        serialize.save(sg, str(out_json))
        raw = json.loads(out_json.read_text())
        obj_dict = raw["layers"]["objects"]["chair_1"]
        assert "best_view_bgr" not in obj_dict, \
            "raw image data leaked into the JSON"

    def test_load_preserves_best_view_path(self, tmp_path: pathlib.Path):
        crop = (np.random.default_rng(7).integers(0, 255, (20, 20, 3))
                .astype(np.uint8))
        sg = self._scene_with_crop(crop)
        out_json = tmp_path / "graph.json"
        serialize.save(sg, str(out_json))

        loaded = serialize.load(str(out_json))
        obj = loaded.objects["chair_1"]
        # best_view_path sits on attrs and survives the round-trip.
        assert obj.attrs["best_view_path"] == "graph_objects/chair_1.png"
        # In-memory field defaults to None on a cold load.
        assert obj.best_view_bgr is None

    def test_no_crop_no_sidecar_dir(self, tmp_path: pathlib.Path):
        sg = SceneGraph(scene="empty")
        sg.objects["chair_1"] = ObjectNode(
            id="chair_1", label="chair",
            position_xyz=[0, 0, 0.5],
            bbox_min_xyz=[-0.3, -0.3, 0], bbox_max_xyz=[0.3, 0.3, 1],
            confidence=0.9, n_observations=4,
        )
        out_json = tmp_path / "graph.json"
        serialize.save(sg, str(out_json))
        sidecar_dir = tmp_path / "graph_objects"
        assert not sidecar_dir.exists(), \
            "sidecar directory should only be created when there's a crop"


class TestGetBestView:
    def test_returns_in_memory_crop_when_available(self, tmp_path: pathlib.Path):
        crop = (np.random.default_rng(9).integers(0, 255, (30, 30, 3))
                .astype(np.uint8))
        sg = SceneGraph()
        obj = ObjectNode(
            id="chair_1", label="chair",
            position_xyz=[0, 0, 0.5],
            bbox_min_xyz=[-0.3, -0.3, 0], bbox_max_xyz=[0.3, 0.3, 1],
            confidence=0.9, n_observations=4,
        )
        obj.best_view_bgr = crop
        sg.objects["chair_1"] = obj
        got = serialize.get_best_view(sg, "chair_1",
                                      str(tmp_path / "does_not_exist.json"))
        assert np.array_equal(got, crop)

    def test_loads_from_sidecar_after_round_trip(self, tmp_path: pathlib.Path):
        crop = (np.random.default_rng(11).integers(0, 255, (20, 24, 3))
                .astype(np.uint8))
        sg = SceneGraph()
        obj = ObjectNode(
            id="chair_1", label="chair",
            position_xyz=[0, 0, 0.5],
            bbox_min_xyz=[-0.3, -0.3, 0], bbox_max_xyz=[0.3, 0.3, 1],
            confidence=0.9, n_observations=4,
        )
        obj.best_view_bgr = crop
        sg.objects["chair_1"] = obj
        out_json = tmp_path / "graph.json"
        serialize.save(sg, str(out_json))

        # Fresh load — no in-memory crop, must pull from PNG.
        loaded = serialize.load(str(out_json))
        got = serialize.get_best_view(loaded, "chair_1", str(out_json))
        assert got is not None
        assert got.shape == crop.shape

    def test_missing_object_returns_none(self, tmp_path: pathlib.Path):
        sg = SceneGraph()
        assert serialize.get_best_view(
            sg, "nonexistent", str(tmp_path / "x.json")) is None

    def test_missing_sidecar_returns_none(self, tmp_path: pathlib.Path):
        sg = SceneGraph()
        sg.objects["chair_1"] = ObjectNode(
            id="chair_1", label="chair",
            position_xyz=[0, 0, 0.5],
            bbox_min_xyz=[-0.3, -0.3, 0], bbox_max_xyz=[0.3, 0.3, 1],
            confidence=0.9, n_observations=4,
            attrs={"best_view_path": "graph_objects/chair_1.png"},
        )
        # Object claims a sidecar but it doesn't exist on disk.
        assert serialize.get_best_view(
            sg, "chair_1", str(tmp_path / "graph.json")) is None
