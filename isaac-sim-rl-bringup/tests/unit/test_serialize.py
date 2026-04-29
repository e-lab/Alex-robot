"""Unit tests for scene_graph.graph.serialize (JSON round-trip)."""

from __future__ import annotations

import json
import pathlib
import tempfile

import numpy as np
import pytest

from scene_graph import SceneGraph, ObjectNode, serialize
from scene_graph.graph.node_types import (
    MeshLayer, PlaceNode, RoomNode, BuildingNode,
)


def _sample_graph() -> SceneGraph:
    sg = SceneGraph(scene="hallway", vocabulary=["door", "chair"], pose_frames=42)
    sg.objects["chair_1"] = ObjectNode(
        id="chair_1", label="chair",
        position_xyz=[1.0, 2.0, 0.5],
        bbox_min_xyz=[0.8, 1.8, 0.0],
        bbox_max_xyz=[1.2, 2.2, 1.0],
        confidence=0.92, n_observations=5,
        first_seen_tick=100, last_seen_tick=200,
        concept_embedding=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        source_tracks=["t_1", "t_7"],
        parent_room="room_1",
    )
    sg.places["place_1"] = PlaceNode(
        id="place_1", position_xyz=[0.5, 0.5, 0.0],
        clearance_m=0.8, neighbors=["place_2"], parent_room="room_1",
    )
    sg.rooms["room_1"] = RoomNode(
        id="room_1", label="hallway",
        centroid_xyz=[1.0, 1.0, 0.0],
        bbox_min_xy=[-2.0, -2.0], bbox_max_xy=[2.0, 2.0],
        object_ids=["chair_1"], place_ids=["place_1"],
    )
    sg.building = BuildingNode(id="b_1", room_ids=["room_1"])
    sg.robot_path = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]
    return sg


class TestRoundTrip:
    def test_save_and_load_preserves_everything(self):
        sg = _sample_graph()
        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as f:
            path = f.name
        try:
            serialize.save(sg, path)
            sg2 = serialize.load(path)
        finally:
            pathlib.Path(path).unlink(missing_ok=True)

        assert sg2.scene == sg.scene
        assert sg2.vocabulary == sg.vocabulary
        assert sg2.pose_frames == sg.pose_frames
        assert set(sg2.objects) == set(sg.objects)
        assert set(sg2.places) == set(sg.places)
        assert set(sg2.rooms) == set(sg.rooms)
        assert sg2.building.room_ids == ["room_1"]
        # robot path preserved bit-for-bit
        assert sg2.robot_path == sg.robot_path

    def test_object_fields_roundtrip(self):
        sg = _sample_graph()
        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as f:
            path = f.name
        try:
            serialize.save(sg, path)
            sg2 = serialize.load(path)
        finally:
            pathlib.Path(path).unlink(missing_ok=True)

        obj = sg2.objects["chair_1"]
        assert obj.label == "chair"
        assert obj.n_observations == 5
        assert obj.source_tracks == ["t_1", "t_7"]
        assert obj.parent_room == "room_1"
        assert obj.concept_embedding.dtype == np.float32
        assert np.allclose(obj.concept_embedding, [0.1, 0.2, 0.3])

    def test_empty_graph(self):
        sg = SceneGraph(scene="empty")
        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as f:
            path = f.name
        try:
            serialize.save(sg, path)
            sg2 = serialize.load(path)
        finally:
            pathlib.Path(path).unlink(missing_ok=True)
        assert sg2.scene == "empty"
        assert len(sg2.objects) == 0

    def test_json_is_human_readable(self):
        sg = _sample_graph()
        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as f:
            path = f.name
        try:
            serialize.save(sg, path)
            raw = pathlib.Path(path).read_text()
        finally:
            pathlib.Path(path).unlink(missing_ok=True)
        # Pretty-printed (has newlines and the top-level keys).
        assert "\n" in raw
        parsed = json.loads(raw)
        assert parsed["meta"]["scene"] == "hallway"
        assert parsed["meta"]["schema_version"] == serialize.SCHEMA_VERSION
        assert "layers" in parsed and "objects" in parsed["layers"]
        # Every layer is present in the dict, even when empty / null.
        for k in ("mesh", "objects", "walls", "places", "rooms", "building"):
            assert k in parsed["layers"]


class TestMeshLayerRoundTrip:
    def test_mesh_placeholder_persists(self):
        sg = SceneGraph(scene="hallway")
        sg.mesh = MeshLayer(voxel_size_m=0.05, storage="voxels.npz", n_voxels=500)
        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as f:
            path = f.name
        try:
            serialize.save(sg, path)
            sg2 = serialize.load(path)
        finally:
            pathlib.Path(path).unlink(missing_ok=True)
        assert sg2.mesh is not None
        assert sg2.mesh.storage == "voxels.npz"
        assert sg2.mesh.n_voxels == 500

    def test_absent_mesh_is_null(self):
        sg = SceneGraph(scene="hallway")
        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as f:
            path = f.name
        try:
            serialize.save(sg, path)
            parsed = json.loads(pathlib.Path(path).read_text())
        finally:
            pathlib.Path(path).unlink(missing_ok=True)
        assert parsed["layers"]["mesh"] is None


class TestAttrsExtensionSlot:
    def test_arbitrary_attrs_roundtrip(self):
        sg = SceneGraph(scene="test")
        sg.objects["chair_1"] = ObjectNode(
            id="chair_1", label="chair",
            position_xyz=[0, 0, 0.5],
            bbox_min_xyz=[-0.2, -0.2, 0], bbox_max_xyz=[0.2, 0.2, 0.9],
            confidence=0.9, n_observations=3,
            attrs={"caption": "a blue armchair", "colour": "blue"},
        )
        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as f:
            path = f.name
        try:
            serialize.save(sg, path)
            sg2 = serialize.load(path)
        finally:
            pathlib.Path(path).unlink(missing_ok=True)
        assert sg2.objects["chair_1"].attrs["caption"] == "a blue armchair"
        assert sg2.objects["chair_1"].attrs["colour"] == "blue"


class TestForwardCompatIgnoresUnknownFields:
    def test_loader_ignores_unknown_object_field(self, tmp_path):
        """A future saver adds a field we don't know — we must silently drop
        it, not crash."""
        path = tmp_path / "future.json"
        path.write_text(json.dumps({
            "meta": {"scene": "hallway", "scan_complete": True,
                     "pose_frames": 0, "vocabulary": [],
                     "schema_version": 99},
            "layers": {
                "mesh": None,
                "objects": {
                    "chair_1": {
                        "id": "chair_1", "label": "chair",
                        "position_xyz": [0, 0, 0.5],
                        "bbox_min_xyz": [-0.1, -0.1, 0],
                        "bbox_max_xyz": [0.1, 0.1, 1.0],
                        "confidence": 0.9,
                        "n_observations": 3,
                        "future_field_we_dont_know": {"weird": True},
                    }
                },
                "places": {}, "rooms": {}, "building": None,
            },
            "robot_path": [],
        }))
        sg = serialize.load(str(path))
        assert "chair_1" in sg.objects


class TestLegacyFlatLayoutMigration:
    def test_migrates_old_flat_json(self, tmp_path):
        """A JSON written before Phase 0 used a flat
        {objects, robot_path, room_id} layout and per-object _n_obs /
        bbox_area_px keys. The loader must upgrade it silently."""
        path = tmp_path / "legacy.json"
        path.write_text(json.dumps({
            "room_id": "FloorPlan1",
            "scan_complete": True,
            "objects": {
                "chair_1": {
                    "label": "chair",
                    "position_xyz": [1.0, 2.0, 0.5],
                    "confidence": 0.93,
                    "first_seen_tick": 100,
                    "last_seen_tick": 200,
                    "bbox_area_px": 48000.0,
                    "_n_obs": 7,
                }
            },
            "robot_path": [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
            "n_objects": 1,
            "object_labels": ["chair"],
        }))

        sg = serialize.load(str(path))
        assert sg.scene == "FloorPlan1"      # from room_id
        assert sg.scan_complete is True
        obj = sg.objects.get("chair_1")
        assert obj is not None
        assert obj.label == "chair"
        assert obj.n_observations == 7        # _n_obs → n_observations
        # bbox fields auto-filled around the centroid
        assert obj.bbox_min_xyz[0] < obj.position_xyz[0] < obj.bbox_max_xyz[0]
        assert obj.id == "chair_1"            # key used as id when missing
        assert sg.robot_path[0] == [0.0, 0.0, 0.0]
