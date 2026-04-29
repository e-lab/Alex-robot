"""Tests for autonomy.target_picker.pick_goal_for_target.

Pure-logic tests using real ObjectNode + SceneGraph instances from the
vendored scene_graph package.
"""
from __future__ import annotations

import pytest

from autonomy.target_picker import pick_goal_for_target
from scene_graph.graph.node_types import ObjectNode
from scene_graph.graph.scene_graph import SceneGraph


def _make_obj(
    sg: SceneGraph,
    label: str,
    *,
    confidence: float = 0.8,
    n_observations: int = 5,
    xyz=(0.0, 0.0, 0.0),
    aliases=None,
):
    """Insert a fully-promoted ObjectNode into ``sg.objects`` with a
    deterministic id, so tests can call this multiple times for the same
    label without manually managing ids."""
    obj_id = sg.new_object_id(label)
    attrs = {}
    if aliases is not None:
        attrs["aliases"] = aliases
    obj = ObjectNode(
        id=obj_id,
        label=label,
        position_xyz=list(xyz),
        bbox_min_xyz=[xyz[0] - 0.1, xyz[1] - 0.1, xyz[2] - 0.1],
        bbox_max_xyz=[xyz[0] + 0.1, xyz[1] + 0.1, xyz[2] + 0.1],
        confidence=confidence,
        n_observations=n_observations,
        attrs=attrs,
    )
    sg.objects[obj_id] = obj
    return obj


class TestPickGoal:
    def test_empty_graph_returns_none(self):
        sg = SceneGraph()
        assert pick_goal_for_target(sg, "oven") is None

    def test_no_label_match_returns_none(self):
        sg = SceneGraph()
        _make_obj(sg, "chair")
        _make_obj(sg, "sofa")
        assert pick_goal_for_target(sg, "oven") is None

    def test_below_min_observations_returns_none(self):
        sg = SceneGraph()
        _make_obj(sg, "oven", confidence=0.9, n_observations=2)
        # min_observations=3 default => below bar
        assert pick_goal_for_target(sg, "oven") is None

    def test_below_lock_conf_returns_none(self):
        sg = SceneGraph()
        _make_obj(sg, "oven", confidence=0.4, n_observations=5)
        # lock_conf=0.6 default => below bar
        assert pick_goal_for_target(sg, "oven") is None

    def test_single_qualifying_match_returns_it(self):
        sg = SceneGraph()
        obj = _make_obj(sg, "oven", confidence=0.8, n_observations=5,
                        xyz=(2.0, 1.0, 0.7))
        result = pick_goal_for_target(sg, "oven")
        assert result is obj
        assert result.position_xyz == [2.0, 1.0, 0.7]

    def test_highest_confidence_wins_among_matches(self):
        sg = SceneGraph()
        _make_obj(sg, "oven", confidence=0.7, n_observations=5)
        winner = _make_obj(sg, "oven", confidence=0.95, n_observations=5)
        _make_obj(sg, "oven", confidence=0.65, n_observations=5)
        assert pick_goal_for_target(sg, "oven") is winner

    def test_case_insensitive_label_match(self):
        sg = SceneGraph()
        obj = _make_obj(sg, "Oven", confidence=0.8, n_observations=5)
        # Lookup is case-insensitive
        assert pick_goal_for_target(sg, "OVEN") is obj
        assert pick_goal_for_target(sg, "oven") is obj
        assert pick_goal_for_target(sg, "oVeN") is obj

    def test_alias_match_via_attrs(self):
        sg = SceneGraph()
        obj = _make_obj(sg, "stove", confidence=0.8, n_observations=5,
                        aliases=["oven", "range"])
        assert pick_goal_for_target(sg, "oven") is obj
        assert pick_goal_for_target(sg, "range") is obj

    def test_alias_string_form_accepted(self):
        # Some labellers might store a single string instead of a list.
        sg = SceneGraph()
        obj = _make_obj(sg, "stove", confidence=0.8, n_observations=5,
                        aliases="oven")
        assert pick_goal_for_target(sg, "oven") is obj

    def test_empty_target_label_returns_none(self):
        sg = SceneGraph()
        _make_obj(sg, "oven", confidence=0.8, n_observations=5)
        assert pick_goal_for_target(sg, "") is None

    def test_custom_thresholds(self):
        sg = SceneGraph()
        obj = _make_obj(sg, "oven", confidence=0.5, n_observations=2)
        # Default thresholds reject; custom (lock_conf=0.4, min_obs=2) accept.
        assert pick_goal_for_target(sg, "oven") is None
        assert pick_goal_for_target(
            sg, "oven", lock_conf=0.4, min_observations=2,
        ) is obj

    def test_pending_objects_not_considered(self):
        sg = SceneGraph()
        # Manually park an object in `pending` (the vendored bucket for
        # candidates that haven't cleared internal promotion). The picker
        # should ignore it even if it would otherwise qualify.
        obj = _make_obj(sg, "oven", confidence=0.9, n_observations=10)
        del sg.objects[obj.id]
        sg.pending[obj.id] = obj
        assert pick_goal_for_target(sg, "oven") is None
