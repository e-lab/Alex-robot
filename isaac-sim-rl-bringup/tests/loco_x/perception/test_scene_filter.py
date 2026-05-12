"""Tests for the spatial-temporal scene-graph filter (LA-2, D2).

The agent observation can't carry the full scene graph every turn —
once Alex has swept a room, ``list_scene()`` returns 20+ nodes each
~50 tokens, and the prompt swells past 1 k tokens just for nodes the
LLM doesn't currently care about. The D2 filter keeps the prompt
bounded by including a node only if:

* it's within ``radius_m`` of the robot (default 5.0 m), **or**
* it was seen within the last ``recency_s`` seconds (default 30.0), **or**
* it matches an *active task / find query* (never drop the thing the
  agent just asked about — even if far and stale).

Cap at ``max_nodes`` (default 20) by closest distance; surface the
hidden count to the LLM so it knows the unfiltered set exists.

The filter is a pure function — no bundle, no clock, no provider —
so the agent runner can reuse it for the observation render *and* for
cross-referencing with the planned path's staleness.
"""
from __future__ import annotations

import pytest

from loco_x.perception.scene_filter import (
    FilteredScene,
    FilterParams,
    filter_scene_nodes,
)


def _node(label, xy, last_seen=0.0, confidence=0.5):
    return {
        "label": label,
        "world_xy": tuple(xy),
        "last_seen": float(last_seen),
        "confidence": float(confidence),
    }


# ── Inclusion rules: distance OR recency OR active target ──────────────────
def test_filter_keeps_close_nodes() -> None:
    """Node within ``radius_m`` of the robot: kept regardless of age."""
    nodes = [
        _node("stove", (0.5, 0.5), last_seen=0.0),     # 0.7 m → close
        _node("microwave", (2.0, 2.0), last_seen=0.0), # 2.8 m → close
    ]
    out = filter_scene_nodes(
        nodes, robot_xy=(0.0, 0.0), now=200.0,        # both stale (200 s)
        active_labels=[],
        params=FilterParams(radius_m=5.0, recency_s=30.0, max_nodes=20),
    )
    labels = [n["label"] for n in out.kept]
    assert {"stove", "microwave"}.issubset(labels)


def test_filter_keeps_recent_far_nodes() -> None:
    """Node far away but seen within ``recency_s``: kept."""
    nodes = [_node("sink", (10.0, 0.0), last_seen=199.0)]
    out = filter_scene_nodes(
        nodes, robot_xy=(0.0, 0.0), now=200.0,
        active_labels=[],
        params=FilterParams(radius_m=5.0, recency_s=30.0, max_nodes=20),
    )
    assert {n["label"] for n in out.kept} == {"sink"}


def test_filter_drops_far_old_nodes() -> None:
    """Node far AND old AND not active: dropped — and recorded in the
    hidden count so the LLM knows it exists."""
    nodes = [
        _node("stove", (0.5, 0.5), last_seen=199.0),   # close → kept
        _node("attic_lamp", (15.0, 0.0), last_seen=0.0),  # far + old → dropped
    ]
    out = filter_scene_nodes(
        nodes, robot_xy=(0.0, 0.0), now=200.0,
        active_labels=[],
        params=FilterParams(radius_m=5.0, recency_s=30.0, max_nodes=20),
    )
    assert {n["label"] for n in out.kept} == {"stove"}
    assert out.hidden_count == 1


def test_filter_always_keeps_active_target_even_when_far_and_stale() -> None:
    """If the agent currently wants ``microwave``, the node must
    survive the filter even if 10 m away and seen 5 minutes ago. The
    LLM is in the middle of acting on it; hiding it would be cruel.

    ``active_labels`` is the contract — the runner populates it from
    ``bundle["goal_label"]`` and any pending ``find(label)`` queries.
    """
    nodes = [
        _node("microwave", (10.0, 0.0), last_seen=0.0),
        _node("garbage", (15.0, 0.0), last_seen=0.0),
    ]
    out = filter_scene_nodes(
        nodes, robot_xy=(0.0, 0.0), now=300.0,
        active_labels=["microwave"],
        params=FilterParams(radius_m=5.0, recency_s=30.0, max_nodes=20),
    )
    labels = {n["label"] for n in out.kept}
    assert "microwave" in labels       # active → kept despite far+stale
    assert "garbage" not in labels     # neither close nor recent nor active


# ── Cap and hidden count ───────────────────────────────────────────────────
def test_filter_caps_at_max_nodes_keeping_closest() -> None:
    """If more than ``max_nodes`` survive inclusion, keep the
    ``max_nodes`` closest by distance. Hidden count records the rest."""
    # 30 nodes, all close-and-recent → all survive inclusion.
    nodes = [
        _node(f"thing_{i:02d}", (float(i) * 0.1, 0.0), last_seen=199.0)
        for i in range(30)
    ]
    out = filter_scene_nodes(
        nodes, robot_xy=(0.0, 0.0), now=200.0,
        active_labels=[],
        params=FilterParams(radius_m=10.0, recency_s=30.0, max_nodes=10),
    )
    assert len(out.kept) == 10
    assert out.hidden_count == 20
    # Closest survives, farthest doesn't.
    assert any(n["label"] == "thing_00" for n in out.kept)
    assert not any(n["label"] == "thing_29" for n in out.kept)


def test_filter_hidden_count_is_zero_when_nothing_filtered() -> None:
    """No hidden line should be printed when the filter is identity."""
    nodes = [_node("stove", (0.5, 0.5), last_seen=199.0)]
    out = filter_scene_nodes(
        nodes, robot_xy=(0.0, 0.0), now=200.0,
        active_labels=[],
        params=FilterParams(radius_m=5.0, recency_s=30.0, max_nodes=20),
    )
    assert out.hidden_count == 0


# ── Order: kept list sorted by distance (closest first) ────────────────────
def test_filter_kept_list_is_sorted_by_distance_ascending() -> None:
    """Closer nodes come first in the rendered observation so the LLM
    sees the most-relevant context up top."""
    nodes = [
        _node("far", (3.0, 0.0), last_seen=199.0),
        _node("near", (0.5, 0.0), last_seen=199.0),
        _node("mid", (1.5, 0.0), last_seen=199.0),
    ]
    out = filter_scene_nodes(
        nodes, robot_xy=(0.0, 0.0), now=200.0,
        active_labels=[],
        params=FilterParams(radius_m=10.0, recency_s=30.0, max_nodes=10),
    )
    labels = [n["label"] for n in out.kept]
    assert labels == ["near", "mid", "far"]


# ── Empty / degenerate cases ───────────────────────────────────────────────
def test_filter_empty_input_returns_empty() -> None:
    out = filter_scene_nodes(
        [], robot_xy=(0.0, 0.0), now=0.0,
        active_labels=[],
        params=FilterParams(),
    )
    assert out.kept == []
    assert out.hidden_count == 0


def test_filter_default_params_match_plan_d2_values() -> None:
    """Spot-check the defaults match D2: radius 5 m, recency 30 s,
    max nodes 20. A drift here would change every observation render
    silently — pin the values."""
    p = FilterParams()
    assert p.radius_m == 5.0
    assert p.recency_s == 30.0
    assert p.max_nodes == 20
