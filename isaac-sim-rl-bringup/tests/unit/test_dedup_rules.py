"""Unit tests for scene_graph.association.dedup_rules."""

from __future__ import annotations

from scene_graph.association import dedup_rules


class TestMergeRadius:
    def test_known_labels(self):
        assert dedup_rules.merge_radius("door") == 1.5
        assert dedup_rules.merge_radius("chair") == 1.0
        assert dedup_rules.merge_radius("bottle") == 0.2

    def test_unknown_label_uses_default(self):
        assert dedup_rules.merge_radius("unicycle") == dedup_rules.DEFAULT_MERGE_RADIUS_M


class TestThresholds:
    def test_ordering_makes_sense(self):
        # Cross-label radius must be tighter than the default merge radius:
        # otherwise a wrong-label neighbour would swallow a legitimate
        # different-label object at the same XYZ.
        assert dedup_rules.CROSS_LABEL_MERGE_RADIUS_M < dedup_rules.DEFAULT_MERGE_RADIUS_M

    def test_cosine_threshold_strict(self):
        assert 0.5 < dedup_rules.EMBEDDING_COSINE_THRESHOLD <= 1.0

    def test_min_sightings_at_least_two(self):
        # A 1-frame flash should never create a persistent object.
        assert dedup_rules.MIN_SIGHTINGS >= 2
