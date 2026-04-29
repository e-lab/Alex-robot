"""Unit tests for P0.7 — 3D AABB IoU-based merge.

Replaces the earlier centroid-only merge: when a new detection carries a
real 3D AABB (from unprojecting every mask pixel), two observations of
the same physical object that overlap in 3D should merge even if their
centroids drifted past the per-label radius.
"""

from __future__ import annotations

import numpy as np

from scene_graph import SceneGraph
from scene_graph.association import merge, dedup_rules


def _insert(sg, label, xyz, mn, mx, score=0.9, tick=0, emb=None):
    return merge.insert_or_merge(
        sg, det_label=label,
        det_xyz=np.asarray(xyz, dtype=np.float32),
        det_bbox_min=np.asarray(mn, dtype=np.float32),
        det_bbox_max=np.asarray(mx, dtype=np.float32),
        det_score=score, det_embedding=emb, det_track_id=None, tick=tick,
    )


def _promote(sg, label, xyz, mn, mx, base_score=0.9, emb=None):
    for i in range(dedup_rules.MIN_SIGHTINGS):
        _insert(sg, label, xyz, mn, mx, score=base_score, tick=i, emb=emb)
    return next(oid for oid, o in sg.objects.items()
                if np.allclose(o.position_xyz, xyz, atol=0.3))


class TestBboxIoUOverlapMerges:
    """Two door observations with drifted centroids but overlapping AABBs
    should merge via the IoU tier even when centroid distance exceeds radius."""

    def test_drifted_centroid_but_overlapping_bbox_merges(self):
        sg = SceneGraph()
        # A door shape: 0.9 m wide, 0.1 m thick, 2 m tall, centred at x=0.
        door_bbox_a = ([-0.45, -0.05, 0.0], [+0.45, +0.05, 2.0])
        _promote(sg, "door", xyz=[0.0, 0.0, 1.0],
                 mn=door_bbox_a[0], mx=door_bbox_a[1])
        assert "door_1" in sg.objects

        # Second view: same door, mask caught only the upper half so
        # centroid rose and shifted slightly. BBox still overlaps heavily
        # with the original.
        door_bbox_b = ([-0.40, -0.05, 0.8], [+0.50, +0.05, 2.0])
        out = _insert(sg, "door",
                      xyz=[0.05, 0.0, 1.4],
                      mn=door_bbox_b[0], mx=door_bbox_b[1],
                      score=0.88, tick=10)
        assert out == "door_1"
        assert sg.objects["door_1"].n_observations == 4

    def test_disjoint_bbox_does_not_merge(self):
        sg = SceneGraph()
        _promote(sg, "door", xyz=[0.0, 0.0, 1.0],
                 mn=[-0.45, -0.05, 0.0], mx=[0.45, 0.05, 2.0])
        # A second "door" 5 m away with a non-overlapping bbox — new pending.
        out = _insert(sg, "door",
                      xyz=[5.0, 0.0, 1.0],
                      mn=[4.55, -0.05, 0.0], mx=[5.45, 0.05, 2.0],
                      score=0.9, tick=10)
        assert out.startswith("pending_")
        assert len(sg.objects) == 1

    def test_bbox_iou_works_without_embedding(self):
        """IoU tier should not require embeddings to be present."""
        sg = SceneGraph()
        _promote(sg, "door", xyz=[0.0, 0.0, 1.0],
                 mn=[-0.45, -0.05, 0.0], mx=[0.45, 0.05, 2.0], emb=None)
        # Nearly identical bbox → high IoU → merge regardless of dist.
        out = _insert(sg, "door",
                      xyz=[0.01, 0.0, 1.0],
                      mn=[-0.44, -0.05, 0.0], mx=[0.44, 0.05, 2.0],
                      score=0.9, tick=10, emb=None)
        assert out == "door_1"


class TestCentroidTierStillWins:
    """Strong centroid match should still merge — we haven't replaced it."""

    def test_within_radius_always_merges(self):
        sg = SceneGraph()
        _promote(sg, "chair", xyz=[0.0, 0.0, 0.5],
                 mn=[-0.2, -0.2, 0.2], mx=[0.2, 0.2, 0.8])
        out = _insert(sg, "chair",
                      xyz=[0.3, 0.0, 0.5],
                      mn=[0.1, -0.2, 0.2], mx=[0.5, 0.2, 0.8],
                      score=0.85, tick=10)
        assert out == "chair_1"
