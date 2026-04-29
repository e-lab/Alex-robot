"""Unit tests for P0.3 — cross-label merge."""

from __future__ import annotations

import numpy as np
import pytest

from scene_graph import SceneGraph
from scene_graph.association import merge, dedup_rules


def _tiny_bbox(center):
    c = np.asarray(center, dtype=np.float32)
    return c - 0.1, c + 0.1


def _insert(sg, label, xyz, score=0.9, tick=0, track_id=None):
    mn, mx = _tiny_bbox(xyz)
    return merge.insert_or_merge(
        sg, det_label=label,
        det_xyz=np.asarray(xyz, dtype=np.float32),
        det_bbox_min=mn, det_bbox_max=mx,
        det_score=score, det_embedding=None, det_track_id=track_id, tick=tick,
    )


def _promote(sg, label, xyz, base_score=0.9):
    """Insert MIN_SIGHTINGS detections to guarantee promotion."""
    for i in range(dedup_rules.MIN_SIGHTINGS):
        _insert(sg, label, xyz, score=base_score, tick=i)
    return next(oid for oid, o in sg.objects.items()
                if np.allclose(o.position_xyz, xyz, atol=0.2))


class TestCrossLabelOnlineInsert:
    """When a new detection arrives at ~same XYZ as an existing different-label
    object, insert_or_merge should absorb it rather than creating a duplicate."""

    def test_absorbs_into_existing_promoted_object(self):
        sg = SceneGraph()
        # Promote an "armchair" first.
        oid = _promote(sg, "armchair", [1.0, 2.0, 0.5], base_score=0.8)
        assert oid == "armchair_1"

        # A "chair" detection lands 0.05 m away with higher confidence
        # → same object, relabel to chair.
        out = _insert(sg, "chair", [1.05, 2.0, 0.5], score=0.95, tick=10)
        assert out == "armchair_1"            # same node
        assert len(sg.objects) == 1
        assert sg.objects["armchair_1"].label == "chair"  # relabel applied
        assert sg.objects["armchair_1"].confidence == pytest.approx(0.95)

    def test_lower_confidence_new_label_does_not_relabel(self):
        sg = SceneGraph()
        oid = _promote(sg, "sofa", [1.0, 2.0, 0.5], base_score=0.9)
        _insert(sg, "chair", [1.1, 2.0, 0.5], score=0.4, tick=10)
        assert sg.objects["sofa_1"].label == "sofa"   # not relabelled

    def test_cross_label_absorb_into_pending(self):
        sg = SceneGraph()
        # One "chair" pending (n=1).
        _insert(sg, "chair", [1.0, 2.0, 0.5], score=0.6, tick=0)
        assert "pending_1" in sg.pending

        # A nearby "armchair" with higher score → absorbed + relabelled,
        # now n=2, still pending.
        _insert(sg, "armchair", [1.05, 2.0, 0.5], score=0.9, tick=1)
        assert len(sg.pending) == 1
        pend = next(iter(sg.pending.values()))
        assert pend.label == "armchair"
        assert pend.confidence == pytest.approx(0.9)
        assert pend.n_observations == 2

    def test_cross_label_far_apart_does_not_merge(self):
        sg = SceneGraph()
        _promote(sg, "sofa", [0.0, 0.0, 0.5], base_score=0.9)
        # 2 m away is well beyond CROSS_LABEL_MERGE_RADIUS_M (0.3 m).
        out = _insert(sg, "chair", [2.0, 0.0, 0.5], score=0.95, tick=10)
        assert out.startswith("pending_")        # new pending candidate
        assert "sofa_1" in sg.objects
        assert sg.objects["sofa_1"].label == "sofa"


class TestCollapseCrossLabel:
    """Post-pass that catches late collisions the online path missed."""

    def test_collapses_two_colocated_promoted_objects(self):
        """Simulate the failure mode where two same-location objects both
        got promoted before either saw the other — collapse_cross_label
        cleans that up as a post-pass."""
        sg = SceneGraph()
        # Promote each at a distance greater than CROSS_LABEL_MERGE_RADIUS_M
        # so the online path does NOT pre-collapse them.
        _promote(sg, "armchair", [0.0, 0.0, 0.5], base_score=0.6)
        _promote(sg, "chair", [2.0, 0.0, 0.5], base_score=0.95)
        assert len(sg.objects) == 2

        # Now teleport one of them to simulate a post-hoc colocation
        # (what happens when running-mean updates slide two objects
        # towards the same true centroid).
        sg.objects["chair_1"].position_xyz = [0.05, 0.0, 0.5]

        collapsed = merge.collapse_cross_label(sg)
        assert collapsed == 1
        assert len(sg.objects) == 1
        # Highest-confidence label wins.
        obj = next(iter(sg.objects.values()))
        assert obj.label == "chair"

    def test_no_collapse_when_separated(self):
        sg = SceneGraph()
        _promote(sg, "chair", [0.0, 0.0, 0.5], base_score=0.9)
        _promote(sg, "sofa", [5.0, 0.0, 0.5], base_score=0.9)
        collapsed = merge.collapse_cross_label(sg)
        assert collapsed == 0
        assert len(sg.objects) == 2

    def test_n_observations_sums(self):
        sg = SceneGraph()
        _promote(sg, "chair", [0.0, 0, 0], base_score=0.9)       # n=3
        _promote(sg, "armchair", [5.0, 0, 0], base_score=0.7)    # n=3, far away
        # Teleport to simulate late colocation.
        sg.objects["armchair_1"].position_xyz = [0.05, 0.0, 0.0]
        merge.collapse_cross_label(sg)
        remaining = next(iter(sg.objects.values()))
        assert remaining.n_observations == 6
