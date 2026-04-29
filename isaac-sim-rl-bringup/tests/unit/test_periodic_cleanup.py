"""Unit tests for `merge.run_periodic_cleanup` and the new
`min_n_observations` guard on `clip_oversized_bboxes`.

Together these are W1 of the May-15 demo plan: move the offline
post-pass cleanup into the SAM3 loop on a cadence so a live consumer
sees ~30 stable objects mid-run instead of ~140 noisy candidates.
"""
from __future__ import annotations

import numpy as np
import pytest

from scene_graph.graph.scene_graph import SceneGraph
from scene_graph.graph.node_types import ObjectNode
from scene_graph.association import merge


def _mk(oid: str, label: str, mn, mx, conf=0.9, n_obs=5):
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


class TestClipMinObservationsGuard:
    """`clip_oversized_bboxes(min_n_observations=K)` must skip objects
    observed fewer than K times — the W1 risk mitigation against
    early-stage candidates with transiently-noisy AABBs being culled
    before they have a chance to grow into a sensible footprint."""

    def test_default_clips_all_oversized(self):
        sg = SceneGraph()
        # A book object with an absurd 3 m extent — would normally drop
        # because cap is ~0.5 m and drop_factor=2 → 1 m. n_obs=2 (small).
        sg.objects["book_runaway"] = _mk(
            "book_runaway", "book",
            mn=[0, 0, 0], mx=[3.0, 0.5, 0.3], n_obs=2)
        n_clipped, n_dropped = merge.clip_oversized_bboxes(sg)
        # Default min_n_observations=1 → object is considered → drops.
        assert n_dropped == 1
        assert "book_runaway" not in sg.objects

    def test_high_min_obs_skips_early_candidates(self):
        sg = SceneGraph()
        # Same runaway, n_obs=2 (early candidate).
        sg.objects["book_early"] = _mk(
            "book_early", "book",
            mn=[0, 0, 0], mx=[3.0, 0.5, 0.3], n_obs=2)
        n_clipped, n_dropped = merge.clip_oversized_bboxes(
            sg, min_n_observations=10)
        # Below threshold → preserved.
        assert n_dropped == 0
        assert "book_early" in sg.objects

    def test_high_min_obs_still_clips_mature(self):
        sg = SceneGraph()
        # Same runaway, but mature (n_obs=50) — should still drop.
        sg.objects["book_mature"] = _mk(
            "book_mature", "book",
            mn=[0, 0, 0], mx=[3.0, 0.5, 0.3], n_obs=50)
        n_clipped, n_dropped = merge.clip_oversized_bboxes(
            sg, min_n_observations=10)
        assert n_dropped == 1
        assert "book_mature" not in sg.objects


class TestRunPeriodicCleanup:
    """End-to-end check that `run_periodic_cleanup` collapses + clips
    + fuses in the documented order, returning per-pass counts."""

    def test_returns_per_pass_counts(self):
        sg = SceneGraph()
        # An overlapping same-label pair. Note `collapse_cross_label`
        # despite its name does collapse pairs *regardless of label*
        # within a 0.3 m radius (see _find_cross_label_nearest). So for
        # two chairs at the same centroid, the cross-label pass fuses
        # them first; the same-label-overlap pass is what runs *after*
        # for cases that geometric-distance missed.
        # Place the two chairs at the same centroid → cross-label fuses.
        sg.objects["chair_1"] = _mk(
            "chair_1", "chair",
            mn=[0.2, 0.2, 0.2], mx=[0.8, 0.8, 0.8])
        sg.objects["chair_2"] = _mk(
            "chair_2", "chair",
            mn=[0.0, 0.0, 0.0], mx=[1.0, 1.0, 1.0])
        # An oversized object well away so it doesn't interact.
        sg.objects["book_runaway"] = _mk(
            "book_runaway", "book",
            mn=[10, 10, 0], mx=[13.0, 10.5, 0.3], n_obs=20)

        stats = merge.run_periodic_cleanup(sg)

        # The two chairs collapse via cross-label (same centroid).
        assert stats["cross_label"] == 1
        # Oversized book drops because n_obs=20 ≥ min_n_observations=10.
        assert stats["dropped"] == 1
        # Sanity on remaining counts.
        assert isinstance(stats["clipped"], int)
        assert isinstance(stats["overlap"], int)
        # Net effect: 3 → 1 surviving (the merged chair).
        assert len(sg.objects) == 1

    def test_skips_early_candidate_clip(self):
        """An object with n_obs < 10 is preserved through the periodic
        cleanup, even if it's currently oversized."""
        sg = SceneGraph()
        sg.objects["book_early"] = _mk(
            "book_early", "book",
            mn=[0, 0, 0], mx=[3.0, 0.5, 0.3], n_obs=3)
        stats = merge.run_periodic_cleanup(sg)
        assert "book_early" in sg.objects
        assert stats["dropped"] == 0

    def test_no_op_on_clean_graph(self):
        """An already-clean graph (no duplicates, no oversize) sees no
        changes and returns all-zero counts."""
        sg = SceneGraph()
        sg.objects["chair_1"] = _mk(
            "chair_1", "chair",
            mn=[0, 0, 0], mx=[0.6, 0.6, 0.6], n_obs=20)
        sg.objects["sofa_1"] = _mk(
            "sofa_1", "sofa",
            mn=[3, 3, 0], mx=[5, 4, 0.8], n_obs=15)

        before = set(sg.objects.keys())
        stats = merge.run_periodic_cleanup(sg)
        after = set(sg.objects.keys())

        assert before == after
        assert all(v == 0 for v in stats.values()), stats

    def test_exposed_via_module_api(self):
        """`run_periodic_cleanup` is in __all__ so it's importable as a
        first-class API rather than an internal helper."""
        from scene_graph.association.merge import __all__ as merge_all
        assert "run_periodic_cleanup" in merge_all
