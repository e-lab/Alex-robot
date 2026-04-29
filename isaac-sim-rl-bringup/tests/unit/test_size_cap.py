"""Unit tests for the Option-A size-cap guard on IoU-based merges.

Two mechanisms:
  - Online guard: `_find_nearby_in_bucket` rejects a candidate match that
    would inflate the target bbox past the per-label cap.
  - Post-save filter: `drop_oversized_objects` removes any already-oversized
    node as a safety net.
"""

from __future__ import annotations

import numpy as np

from scene_graph import SceneGraph
from scene_graph.association import merge, dedup_rules


def _insert(sg, label, xyz, mn, mx, score=0.9, tick=0):
    return merge.insert_or_merge(
        sg, det_label=label,
        det_xyz=np.asarray(xyz, dtype=np.float32),
        det_bbox_min=np.asarray(mn, dtype=np.float32),
        det_bbox_max=np.asarray(mx, dtype=np.float32),
        det_score=score, det_embedding=None, det_track_id=None, tick=tick,
    )


def _promote(sg, label, xyz, mn, mx):
    for i in range(dedup_rules.MIN_SIGHTINGS):
        _insert(sg, label, xyz, mn, mx, tick=i)
    return next(oid for oid, o in sg.objects.items() if o.label == label)


class TestMaxExtentLookup:
    def test_known_labels(self):
        assert dedup_rules.max_extent("door") == 1.5
        assert dedup_rules.max_extent("chair") == 1.5
        assert dedup_rules.max_extent("bottle") == 0.3

    def test_unknown_label_uses_default(self):
        assert dedup_rules.max_extent("unicycle") == dedup_rules.DEFAULT_MAX_EXTENT_M


class TestOnlineSizeGuard:
    def test_merge_rejected_when_would_oversize(self):
        sg = SceneGraph()
        # A door near the cap already (1.0 m wide).
        _promote(sg, "door", xyz=[0.0, 0.0, 1.0],
                 mn=[-0.5, -0.05, 0.0], mx=[0.5, 0.05, 2.0])
        assert "door_1" in sg.objects

        # A "door" detection 4 m away with a bbox that WOULD UNION to
        # >1.5 m wide → merge must be rejected, must create new pending.
        out = _insert(sg, "door",
                      xyz=[4.0, 0.0, 1.0],
                      mn=[3.5, -0.05, 0.0], mx=[4.5, 0.05, 2.0],
                      score=0.9, tick=10)
        assert out.startswith("pending_"), f"got {out}"
        # door_1 untouched — horizontal footprint still within cap.
        d = sg.objects["door_1"]
        ext = np.asarray(d.bbox_max_xyz) - np.asarray(d.bbox_min_xyz)
        assert max(float(ext[0]), float(ext[1])) <= 1.5

    def test_small_overlap_within_cap_still_merges(self):
        """Sanity: we didn't break the normal merge path."""
        sg = SceneGraph()
        _promote(sg, "door", xyz=[0.0, 0.0, 1.0],
                 mn=[-0.45, -0.05, 0.0], mx=[0.45, 0.05, 2.0])
        # Slightly drifted view, well within the cap.
        out = _insert(sg, "door",
                      xyz=[0.05, 0.0, 1.0],
                      mn=[-0.40, -0.05, 0.0], mx=[0.50, 0.05, 2.0],
                      score=0.9, tick=10)
        assert out == "door_1"


class TestDropOversizedObjects:
    def test_drops_hugely_oversized_node(self):
        """Severely-over-cap (× drop_factor) → dropped."""
        sg = SceneGraph()
        _promote(sg, "door", xyz=[0.0, 0.0, 1.0],
                 mn=[-0.45, -0.05, 0.0], mx=[0.45, 0.05, 2.0])
        # 6 m wide door — 4× over cap, definitely runaway.
        sg.objects["door_1"].bbox_min_xyz = [-3.0, -0.1, 0.0]
        sg.objects["door_1"].bbox_max_xyz = [+3.0, +0.1, 2.0]
        n = merge.drop_oversized_objects(sg)
        assert n == 1
        assert len(sg.objects) == 0

    def test_keeps_valid_nodes(self):
        sg = SceneGraph()
        _promote(sg, "chair", xyz=[0.0, 0.0, 0.5],
                 mn=[-0.3, -0.3, 0.0], mx=[0.3, 0.3, 0.9])
        n_drop = merge.drop_oversized_objects(sg)
        assert n_drop == 0
        assert len(sg.objects) == 1


class TestClipOversizedBboxes:
    """Two-tier: shrink mildly-oversized, drop hugely-oversized."""

    def test_mildly_oversized_gets_clipped_not_dropped(self):
        sg = SceneGraph()
        # Promote a chair with a normal bbox.
        _promote(sg, "chair", xyz=[0.0, 0.0, 0.5],
                 mn=[-0.3, -0.3, 0.0], mx=[0.3, 0.3, 0.9])
        # Inflate to 2.0 m wide (chair cap 1.5, drop_factor 2.0 → just below).
        sg.objects["chair_1"].bbox_min_xyz = [-1.0, -1.0, 0.0]
        sg.objects["chair_1"].bbox_max_xyz = [+1.0, +1.0, 0.9]

        n_clipped, n_dropped = merge.clip_oversized_bboxes(sg)
        assert n_clipped == 1
        assert n_dropped == 0
        assert len(sg.objects) == 1
        # Clipped to 1.5 m wide centred on centroid (0,0,0.5).
        mn = np.asarray(sg.objects["chair_1"].bbox_min_xyz)
        mx = np.asarray(sg.objects["chair_1"].bbox_max_xyz)
        ext = mx - mn
        assert max(ext[0], ext[1]) <= 1.5 + 1e-4

    def test_severely_oversized_is_dropped(self):
        sg = SceneGraph()
        _promote(sg, "chair", xyz=[0.0, 0.0, 0.5],
                 mn=[-0.3, -0.3, 0.0], mx=[0.3, 0.3, 0.9])
        # 5 m wide chair — 4× cap → dropped.
        sg.objects["chair_1"].bbox_min_xyz = [-2.5, -2.5, 0.0]
        sg.objects["chair_1"].bbox_max_xyz = [+2.5, +2.5, 0.9]
        n_clipped, n_dropped = merge.clip_oversized_bboxes(sg)
        assert n_clipped == 0
        assert n_dropped == 1
        assert len(sg.objects) == 0

    def test_within_cap_unchanged(self):
        sg = SceneGraph()
        _promote(sg, "chair", xyz=[0.0, 0.0, 0.5],
                 mn=[-0.3, -0.3, 0.0], mx=[0.3, 0.3, 0.9])
        before_mn = list(sg.objects["chair_1"].bbox_min_xyz)
        before_mx = list(sg.objects["chair_1"].bbox_max_xyz)
        n_clipped, n_dropped = merge.clip_oversized_bboxes(sg)
        assert n_clipped == 0 and n_dropped == 0
        assert sg.objects["chair_1"].bbox_min_xyz == before_mn
        assert sg.objects["chair_1"].bbox_max_xyz == before_mx
