"""Unit tests for P0.5 — embedding-boosted same-label merge.

When a new detection falls outside the per-label radius but within
`stretch × radius`, the merge must only happen if the SAM3 mask-pooled
embedding is close enough to the stored one (cosine ≥ threshold).
"""

from __future__ import annotations

import numpy as np
import pytest

from scene_graph import SceneGraph
from scene_graph.association import merge, dedup_rules


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _insert(sg, label, xyz, score=0.9, tick=0, emb=None):
    mn = np.asarray(xyz, dtype=np.float32) - 0.1
    mx = np.asarray(xyz, dtype=np.float32) + 0.1
    return merge.insert_or_merge(
        sg, det_label=label,
        det_xyz=np.asarray(xyz, dtype=np.float32),
        det_bbox_min=mn, det_bbox_max=mx,
        det_score=score, det_embedding=emb, det_track_id=None, tick=tick,
    )


def _promote(sg, label, xyz, emb=None, base_score=0.9):
    """Insert MIN_SIGHTINGS times to guarantee promotion."""
    for i in range(dedup_rules.MIN_SIGHTINGS):
        _insert(sg, label, xyz, score=base_score, tick=i, emb=emb)
    # Find the promoted id.
    for oid, o in sg.objects.items():
        if np.allclose(o.position_xyz, xyz, atol=0.2):
            return oid
    raise AssertionError("promotion failed")


class TestStretchRadiusRescue:
    """Detection just past the per-label radius, with matching embedding:
    should merge into existing instead of creating a new pending."""

    def test_door_drift_beyond_radius_but_similar_embedding_merges(self):
        """Stretch rescue: chair merge radius is 1.0 m; a detection 1.2 m
        away is past the radius but within 2×radius, and the merged bbox
        stays under the chair size cap (1.5 m). With matching embedding,
        it should merge."""
        sg = SceneGraph()
        emb = _unit([1.0, 0.0, 0.0])
        _promote(sg, "chair", [0.0, 0.0, 0.5], emb=emb)
        assert "chair_1" in sg.objects

        # 1.5 m away — chair radius 1.0, so past it; within 2×radius.
        out = _insert(sg, "chair", [1.2, 0.0, 0.5],
                      score=0.92, emb=emb, tick=10)
        assert out == "chair_1"
        assert sg.objects["chair_1"].n_observations == 4

    def test_door_drift_with_different_embedding_creates_new(self):
        sg = SceneGraph()
        emb_a = _unit([1.0, 0.0, 0.0])
        emb_b = _unit([0.0, 1.0, 0.0])   # orthogonal to emb_a → cos=0
        _promote(sg, "chair", [0.0, 0.0, 0.5], emb=emb_a)

        # Same geometry (within stretch) but orthogonal embedding → no merge.
        out = _insert(sg, "chair", [1.2, 0.0, 0.5],
                      score=0.92, emb=emb_b, tick=10)
        assert out.startswith("pending_"), f"got {out}"
        assert len(sg.objects) == 1         # chair_1 untouched

    def test_no_embedding_on_either_side_never_merges_at_stretch(self):
        """Safety: with no embedding data, we never rely on the stretch."""
        sg = SceneGraph()
        _promote(sg, "chair", [0.0, 0.0, 0.5], emb=None)
        out = _insert(sg, "chair", [1.2, 0.0, 0.5],
                      score=0.9, emb=None, tick=10)
        assert out.startswith("pending_")


class TestStrongMatchStillWorks:
    """Detection inside the per-label radius always merges, embedding or not."""

    def test_strong_match_no_embedding(self):
        sg = SceneGraph()
        _promote(sg, "chair", [0.0, 0.0, 0.5], emb=None)
        out = _insert(sg, "chair", [0.3, 0.0, 0.5], emb=None, tick=10)
        assert out == "chair_1"

    def test_strong_match_ignores_bad_embedding(self):
        """Even if embedding disagrees, a within-radius detection merges
        (geometric trust beats visual disagreement for the strong tier)."""
        sg = SceneGraph()
        emb_a = _unit([1.0, 0.0, 0.0])
        emb_b = _unit([0.0, 1.0, 0.0])    # orthogonal
        _promote(sg, "chair", [0.0, 0.0, 0.5], emb=emb_a)
        out = _insert(sg, "chair", [0.3, 0.0, 0.5], emb=emb_b, tick=10)
        assert out == "chair_1"


class TestEmbeddingStorage:
    """A detection's embedding is stored on the node after promotion."""

    def test_embedding_persists_through_promotion(self):
        sg = SceneGraph()
        emb = _unit([0.6, 0.8, 0.0])
        _promote(sg, "door", [0.0, 0.0, 1.0], emb=emb)
        saved = sg.objects["door_1"].concept_embedding
        assert saved is not None
        # Keep-best-score rule should leave the original embedding in place.
        assert np.allclose(saved, emb, atol=1e-6)
