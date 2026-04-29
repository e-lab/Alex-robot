"""Unit tests for merge.collapse_overlapping_same_label (Phase 0 refinement).

Catches the failure mode where two same-label promoted objects have
overlapping 3D bboxes but centroid distance > cross-label radius — so
collapse_cross_label leaves both intact.
"""
from __future__ import annotations

import numpy as np

from scene_graph import SceneGraph, ObjectNode, WallNode, RoomNode
from scene_graph.association import merge


def _mk_obj(oid: str, label: str, mn, mx, conf=0.9, n_obs=5):
    mn = np.asarray(mn, dtype=np.float32)
    mx = np.asarray(mx, dtype=np.float32)
    centre = ((mn + mx) * 0.5).tolist()
    return ObjectNode(
        id=oid, label=label,
        position_xyz=centre,
        bbox_min_xyz=mn.tolist(),
        bbox_max_xyz=mx.tolist(),
        confidence=conf, n_observations=n_obs,
    )


class TestCollapseOverlappingSameLabel:
    def test_fully_nested_pair_collapses(self):
        """chair_2's bbox fully contains chair_1's bbox → collapse."""
        sg = SceneGraph()
        sg.objects["chair_1"] = _mk_obj(
            "chair_1", "chair",
            mn=[0.2, 0.2, 0.2], mx=[0.8, 0.8, 0.8])
        sg.objects["chair_2"] = _mk_obj(
            "chair_2", "chair",
            mn=[0.0, 0.0, 0.0], mx=[1.0, 1.0, 1.0])

        n = merge.collapse_overlapping_same_label(sg)
        assert n == 1
        assert len(sg.objects) == 1
        # The larger-volume chair_2 keeps its id.
        assert "chair_2" in sg.objects

    def test_partial_overlap_above_threshold_collapses(self):
        """chair_1 ~80% inside chair_2 → collapse."""
        sg = SceneGraph()
        # chair_1: 0.5 × 0.5 × 0.5 cube at origin (volume 0.125).
        sg.objects["chair_1"] = _mk_obj(
            "chair_1", "chair",
            mn=[-0.25, -0.25, 0.0], mx=[0.25, 0.25, 0.5])
        # chair_2: 1.0 × 1.0 × 1.0 cube offset so ~80% of chair_1 is inside.
        sg.objects["chair_2"] = _mk_obj(
            "chair_2", "chair",
            mn=[-0.20, -0.20, 0.0], mx=[0.80, 0.80, 1.0])

        n = merge.collapse_overlapping_same_label(sg, containment_threshold=0.5)
        assert n == 1
        assert len(sg.objects) == 1

    def test_partial_overlap_below_threshold_keeps_both(self):
        sg = SceneGraph()
        # chair_1 and chair_2 barely touch → containment ~0.
        sg.objects["chair_1"] = _mk_obj(
            "chair_1", "chair",
            mn=[0.0, 0.0, 0.0], mx=[0.5, 0.5, 0.5])
        sg.objects["chair_2"] = _mk_obj(
            "chair_2", "chair",
            mn=[0.45, 0.45, 0.0], mx=[1.0, 1.0, 0.5])
        n = merge.collapse_overlapping_same_label(sg, containment_threshold=0.5)
        assert n == 0
        assert len(sg.objects) == 2

    def test_different_labels_never_collapse(self):
        sg = SceneGraph()
        sg.objects["chair_1"] = _mk_obj(
            "chair_1", "chair",
            mn=[0, 0, 0], mx=[1, 1, 1])
        sg.objects["table_1"] = _mk_obj(
            "table_1", "coffee table",
            mn=[0, 0, 0], mx=[1, 1, 1])   # perfectly overlapping but wrong label
        n = merge.collapse_overlapping_same_label(sg)
        assert n == 0
        assert len(sg.objects) == 2

    def test_preserves_observation_count(self):
        sg = SceneGraph()
        sg.objects["chair_1"] = _mk_obj(
            "chair_1", "chair",
            mn=[0, 0, 0], mx=[0.5, 0.5, 0.5], n_obs=3)
        sg.objects["chair_2"] = _mk_obj(
            "chair_2", "chair",
            mn=[0, 0, 0], mx=[1.0, 1.0, 1.0], n_obs=7)
        merge.collapse_overlapping_same_label(sg)
        survivor = next(iter(sg.objects.values()))
        # Summed observations.
        assert survivor.n_observations == 10

    def test_union_bbox(self):
        sg = SceneGraph()
        sg.objects["chair_1"] = _mk_obj(
            "chair_1", "chair",
            mn=[0.0, 0.0, 0.0], mx=[0.5, 0.5, 0.5])
        sg.objects["chair_2"] = _mk_obj(
            "chair_2", "chair",
            mn=[-0.1, -0.1, 0.0], mx=[1.0, 1.0, 0.8])
        merge.collapse_overlapping_same_label(sg)
        survivor = next(iter(sg.objects.values()))
        np.testing.assert_allclose(survivor.bbox_min_xyz, [-0.1, -0.1, 0.0], atol=1e-6)
        np.testing.assert_allclose(survivor.bbox_max_xyz, [1.0, 1.0, 0.8], atol=1e-6)

    def test_door_collapse_rewrites_wall_backref(self):
        """If a collapsed door was in wall.door_ids, the survivor takes its place."""
        sg = SceneGraph()
        sg.walls["wall_0"] = WallNode(
            id="wall_0", ax=0.0, ay=0.0, bx=5.0, by=0.0,
            door_ids=["door_1", "door_2"],
        )
        sg.objects["door_1"] = _mk_obj(
            "door_1", "door",
            mn=[1.9, -0.05, 0.0], mx=[2.5, 0.05, 2.1])
        sg.objects["door_2"] = _mk_obj(
            "door_2", "door",
            mn=[1.8, -0.05, 0.0], mx=[2.6, 0.05, 2.1])
        n = merge.collapse_overlapping_same_label(sg)
        assert n == 1
        assert len(sg.objects) == 1
        survivor = next(iter(sg.objects.keys()))
        # The wall must point only at the survivor (once).
        assert sg.walls["wall_0"].door_ids == [survivor]

    def test_room_backref_rewritten(self):
        sg = SceneGraph()
        sg.rooms["room_1"] = RoomNode(
            id="room_1", label=None,
            centroid_xyz=[0, 0, 0],
            bbox_min_xy=[-1, -1], bbox_max_xy=[1, 1],
            object_ids=["chair_1", "chair_2"],
        )
        sg.objects["chair_1"] = _mk_obj(
            "chair_1", "chair", mn=[0, 0, 0], mx=[0.5, 0.5, 0.5])
        sg.objects["chair_2"] = _mk_obj(
            "chair_2", "chair", mn=[-0.1, -0.1, 0], mx=[1, 1, 1])
        merge.collapse_overlapping_same_label(sg)
        survivor = next(iter(sg.objects.keys()))
        assert sg.rooms["room_1"].object_ids == [survivor]

    def test_triple_overlap_resolves_to_one(self):
        sg = SceneGraph()
        sg.objects["chair_1"] = _mk_obj(
            "chair_1", "chair",
            mn=[0.1, 0.1, 0.1], mx=[0.4, 0.4, 0.4])
        sg.objects["chair_2"] = _mk_obj(
            "chair_2", "chair",
            mn=[0.2, 0.2, 0.2], mx=[0.5, 0.5, 0.5])
        sg.objects["chair_3"] = _mk_obj(
            "chair_3", "chair",
            mn=[0.0, 0.0, 0.0], mx=[1.0, 1.0, 1.0])
        n = merge.collapse_overlapping_same_label(sg)
        # chair_3 absorbs both chair_1 and chair_2 in one sweep.
        assert n == 2
        assert len(sg.objects) == 1
        assert "chair_3" in sg.objects

    def test_embedding_guard_blocks_merge_when_too_different(self):
        sg = SceneGraph()
        emb_a = np.array([1, 0, 0], dtype=np.float32)
        emb_b = np.array([0, 1, 0], dtype=np.float32)    # orthogonal
        sg.objects["chair_1"] = _mk_obj(
            "chair_1", "chair", mn=[0, 0, 0], mx=[0.5, 0.5, 0.5])
        sg.objects["chair_1"].concept_embedding = emb_a
        sg.objects["chair_2"] = _mk_obj(
            "chair_2", "chair", mn=[-0.1, -0.1, 0], mx=[1, 1, 1])
        sg.objects["chair_2"].concept_embedding = emb_b

        n = merge.collapse_overlapping_same_label(
            sg, require_embedding_cosine=0.85)
        assert n == 0
        assert len(sg.objects) == 2
