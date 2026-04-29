"""Unit tests for scene_graph.association.merge.

Phase-0 status (P0.1 + P0.2):
- Per-label merge radius is in force (dedup_rules.merge_radius).
- MIN_SIGHTINGS gate keeps new detections in `sg.pending` until they've
  been observed ≥ MIN_SIGHTINGS times; only then do they migrate to
  `sg.objects` with a proper `<label>_N` id.
- Bbox IoU / cross-label collapse / embedding cosine come in P0.3-P0.9.
"""

from __future__ import annotations

import numpy as np
import pytest

from scene_graph import SceneGraph
from scene_graph.association import merge, dedup_rules


def _tiny_bbox(center):
    mn = np.asarray(center, dtype=np.float32) - 0.1
    mx = np.asarray(center, dtype=np.float32) + 0.1
    return mn, mx


def _insert(sg, label, xyz, score=0.9, tick=0, track_id=None, emb=None):
    mn, mx = _tiny_bbox(xyz)
    return merge.insert_or_merge(
        sg, det_label=label,
        det_xyz=np.asarray(xyz, dtype=np.float32),
        det_bbox_min=mn, det_bbox_max=mx,
        det_score=score, det_embedding=emb, det_track_id=track_id, tick=tick,
    )


class TestPendingBucket:
    def test_first_detection_goes_to_pending_not_objects(self):
        sg = SceneGraph(scene="test")
        oid = _insert(sg, "chair", [1.0, 2.0, 0.5])
        assert oid.startswith("pending_"), f"expected pending_* got {oid}"
        assert len(sg.objects) == 0
        assert len(sg.pending) == 1

    def test_promotion_after_min_sightings(self):
        sg = SceneGraph(scene="test")
        # Same spot 3 times (default MIN_SIGHTINGS = 3).
        ids = [_insert(sg, "chair", [1.0, 2.0, 0.5], tick=t) for t in range(3)]
        assert ids[0].startswith("pending_")
        assert ids[1] == ids[0]                        # still pending, same id
        assert ids[2] == "chair_1"                     # promoted on 3rd sighting
        assert len(sg.objects) == 1
        assert len(sg.pending) == 0
        assert sg.objects["chair_1"].n_observations == 3

    def test_single_sighting_stays_pending(self):
        sg = SceneGraph(scene="test")
        _insert(sg, "door", [0, 0, 0])
        assert len(sg.objects) == 0
        assert len(sg.pending) == 1

    def test_merge_into_already_promoted_object(self):
        sg = SceneGraph(scene="test")
        # Promote first
        for t in range(3):
            _insert(sg, "sofa", [0, 0, 0], tick=t)
        assert "sofa_1" in sg.objects
        # Fourth sighting within radius merges into sofa_1.
        oid = _insert(sg, "sofa", [0.1, 0.0, 0.0], tick=10)
        assert oid == "sofa_1"
        assert sg.objects["sofa_1"].n_observations == 4


class TestMergeRadii:
    def test_same_label_within_radius_merges(self):
        sg = SceneGraph(scene="test")
        _insert(sg, "chair", [1.0, 2.0, 0.5], score=0.8, tick=0)
        # "chair" per-label radius = 0.6 m. 0.3 m offset → same pending.
        id2 = _insert(sg, "chair", [1.3, 2.0, 0.5], score=0.95, tick=10)
        assert id2.startswith("pending_")              # still below min
        assert len(sg.pending) == 1

    def test_same_label_outside_radius_creates_new(self):
        sg = SceneGraph(scene="test")
        _insert(sg, "chair", [0.0, 0.0, 0.5])
        # 5 m apart — new pending candidate.
        _insert(sg, "chair", [5.0, 0.0, 0.5])
        assert len(sg.pending) == 2


class TestPerLabelRadii:
    def test_door_radius_is_wider_than_chair(self):
        assert dedup_rules.merge_radius("door") > dedup_rules.merge_radius("chair")


class TestMergeSemantics:
    def test_running_average_position_after_promotion(self):
        sg = SceneGraph(scene="test")
        # sofa radius is 1.0 m — use strictly-less-than offsets.
        _insert(sg, "sofa", [0.0, 0.0, 0.5], tick=0)
        _insert(sg, "sofa", [0.8, 0.0, 0.5], tick=1)   # within 1.0 m
        pid3 = _insert(sg, "sofa", [0.4, 0.0, 0.5], tick=2)
        # Promoted on 3rd; position is running mean of [0.0, 0.8, 0.4] = 0.4.
        assert pid3 == "sofa_1"
        assert sg.objects["sofa_1"].position_xyz[0] == pytest.approx(0.4, abs=0.05)

    def test_confidence_keeps_best(self):
        sg = SceneGraph(scene="test")
        _insert(sg, "oven", [0, 0, 0], score=0.5, tick=0)
        _insert(sg, "oven", [0, 0, 0], score=0.9, tick=1)
        _insert(sg, "oven", [0, 0, 0], score=0.7, tick=2)
        assert sg.objects["oven_1"].confidence == pytest.approx(0.9)

    def test_last_seen_tick_updates(self):
        sg = SceneGraph(scene="test")
        _insert(sg, "oven", [0, 0, 0], tick=0)
        _insert(sg, "oven", [0, 0, 0], tick=10)
        _insert(sg, "oven", [0, 0, 0], tick=25)
        assert sg.objects["oven_1"].last_seen_tick == 25

    def test_track_ids_accumulate_and_dedupe(self):
        sg = SceneGraph(scene="test")
        _insert(sg, "door", [0, 0, 0], track_id="t_1", tick=0)
        _insert(sg, "door", [0, 0, 0], track_id="t_2", tick=5)
        _insert(sg, "door", [0, 0, 0], track_id="t_1", tick=10)   # duplicate
        # Promoted (n=3). Track list is deduped.
        obj = sg.objects["door_1"]
        assert obj.source_tracks == ["t_1", "t_2"]


class TestCrossLabelBehaviour:
    """P0.3 enables cross-label collapse within CROSS_LABEL_MERGE_RADIUS_M.
    Detailed tests live in tests/unit/test_cross_label.py — these pin the
    merge.insert_or_merge side of it."""

    def test_same_xyz_different_label_collapses_to_one_pending(self):
        sg = SceneGraph(scene="test")
        _insert(sg, "chair", [1, 2, 0.5], score=0.6)
        _insert(sg, "armchair", [1, 2, 0.5], score=0.8)
        # Only one pending candidate, relabelled to the higher-conf label.
        assert len(sg.pending) == 1
        obj = next(iter(sg.pending.values()))
        assert obj.label == "armchair"
        assert obj.n_observations == 2

    def test_far_apart_different_label_stays_separate(self):
        sg = SceneGraph(scene="test")
        _insert(sg, "chair", [0, 0, 0], score=0.9)
        _insert(sg, "sofa", [5, 0, 0], score=0.9)
        assert len(sg.pending) == 2
