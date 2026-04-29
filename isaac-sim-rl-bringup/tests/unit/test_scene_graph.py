"""Unit tests for scene_graph.graph.scene_graph.SceneGraph container."""

from __future__ import annotations

import pytest

from scene_graph import SceneGraph, ObjectNode


class TestIdGeneration:
    def test_sequential_ids(self):
        sg = SceneGraph()
        assert sg.new_object_id("chair") == "chair_1"
        assert sg.new_object_id("chair") == "chair_2"
        assert sg.new_object_id("door") == "door_1"
        assert sg.new_object_id("chair") == "chair_3"

    def test_ids_are_unique_per_label(self):
        sg = SceneGraph()
        used = {sg.new_object_id("box") for _ in range(50)}
        assert len(used) == 50


class TestSummary:
    def test_empty(self):
        assert SceneGraph().summary() == "objects=0 (—)  walls=0  places=0  rooms=0"

    def test_counts_by_label(self):
        sg = SceneGraph()
        for _ in range(2):
            new_id = sg.new_object_id("chair")
            sg.objects[new_id] = ObjectNode(
                id=new_id, label="chair",
                position_xyz=[0, 0, 0], bbox_min_xyz=[0, 0, 0], bbox_max_xyz=[0, 0, 0],
                confidence=1.0,
            )
        new_id = sg.new_object_id("door")
        sg.objects[new_id] = ObjectNode(
            id=new_id, label="door",
            position_xyz=[0, 0, 0], bbox_min_xyz=[0, 0, 0], bbox_max_xyz=[0, 0, 0],
            confidence=1.0,
        )
        s = sg.summary()
        assert "2×chair" in s
        assert "1×door" in s
        assert "objects=3" in s


class TestObjectNode:
    def test_centroid_matches_position(self):
        o = ObjectNode(
            id="x", label="x",
            position_xyz=[1.0, 2.0, 3.0],
            bbox_min_xyz=[0, 0, 0], bbox_max_xyz=[2, 4, 6], confidence=1.0,
        )
        assert o.centroid().tolist() == [1.0, 2.0, 3.0]

    def test_bbox_volume(self):
        o = ObjectNode(
            id="x", label="x",
            position_xyz=[0, 0, 0],
            bbox_min_xyz=[0, 0, 0], bbox_max_xyz=[2, 3, 4], confidence=1.0,
        )
        assert o.bbox_volume() == pytest.approx(24.0)
