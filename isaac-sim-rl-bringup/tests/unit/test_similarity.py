"""Unit tests for scene_graph.association.similarity + detection.embeddings.cosine."""

from __future__ import annotations

import numpy as np
import pytest

from scene_graph import ObjectNode
from scene_graph.association import similarity
from scene_graph.detection import embeddings


def _obj(pos, min_xyz, max_xyz, embedding=None):
    return ObjectNode(
        id="x", label="x",
        position_xyz=list(pos), bbox_min_xyz=list(min_xyz), bbox_max_xyz=list(max_xyz),
        confidence=1.0, concept_embedding=embedding,
    )


class TestSpatialScore:
    def test_overlapping_box_is_partial(self):
        o = _obj([0, 0, 0], [0, 0, 0], [1, 1, 1])
        det_mn = np.array([0.5, 0, 0]); det_mx = np.array([1.5, 1, 1])
        s = similarity.spatial_score(det_mn, det_mx, o)
        assert 0.0 < s < 1.0

    def test_disjoint_is_zero(self):
        o = _obj([0, 0, 0], [0, 0, 0], [1, 1, 1])
        s = similarity.spatial_score(np.array([5, 5, 5]), np.array([6, 6, 6]), o)
        assert s == 0.0


class TestVisualScore:
    def test_none_embedding_returns_zero(self):
        o = _obj([0, 0, 0], [0, 0, 0], [1, 1, 1], embedding=None)
        assert similarity.visual_score(np.array([1.0, 0.0, 0.0]), o) == 0.0

    def test_identical_normalised_vectors_give_1(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        o = _obj([0, 0, 0], [0, 0, 0], [1, 1, 1], embedding=v)
        assert similarity.visual_score(v, o) == pytest.approx(1.0)

    def test_orthogonal_vectors_give_0(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        o = _obj([0, 0, 0], [0, 0, 0], [1, 1, 1], embedding=b)
        assert similarity.visual_score(a, o) == pytest.approx(0.0)


class TestCentroidDistance:
    def test_known_distance(self):
        o = _obj([0, 0, 0], [0, 0, 0], [0, 0, 0])
        d = similarity.centroid_distance(np.array([3.0, 4.0, 0.0]), o)
        assert d == pytest.approx(5.0)
