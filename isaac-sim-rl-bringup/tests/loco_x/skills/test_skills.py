"""Tests for the LLM-callable skill library (LA-1, D4).

The skill library is the *single source of truth for the action
space*. Every function the LLM can name is here; every error path
returns the spatial-error-context dict shape spelled out in D4. The
sandbox (LA-1/D1) executes against the namespace this module exposes.

LA-1 covers the **API surface**:
* the registry composes the namespace the sandbox executes against,
* each skill returns a serializable dict in the D4 shape,
* failure dicts carry the right ``error_kind`` and spatial fields,
* meta skills (``finish``, ``fail``) set the agent's stop flag.

LA-5 will add the *blocking* await (skill queues a task → autonomy
loop completes it → skill returns the result). For LA-1 the skills
are non-blocking; they push to the queue and return immediately.
That keeps the tests pure-Python with no concurrency.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from loco_x.occupancy import CellState
from loco_x.skills import SkillRegistry, make_skills


# ── Bundle stub ─────────────────────────────────────────────────────────────
def _make_bundle(
    *,
    robot_xy: Tuple[float, float] = (0.0, 0.0),
    robot_yaw_rad: float = 0.0,
    scene_nodes: Optional[List[Dict[str, Any]]] = None,
    occ_provider: Optional[Any] = None,
) -> dict:
    """A bundle dict shaped like the autonomy bundle but with stubs.

    Skills only consume a small slice: ``task_queue``, ``robot_pose``,
    ``scene_nodes``, ``occ_provider``, ``goal_label``. The rest of the
    autonomy bundle is irrelevant to LA-1.
    """
    return {
        "task_queue": [],
        "task_result": None,
        "agent_should_stop": False,
        "task_result_status": None,
        "task_result_reason": None,
        "robot_pose": {"xy": robot_xy, "yaw_rad": robot_yaw_rad},
        "scene_nodes": scene_nodes or [],
        "occ_provider": occ_provider,
        "goal_label": None,
    }


class _FakeProvider:
    """Just enough of OccupancyProvider for the locomotion skills.

    Honours the LA-0a out-of-bounds contract: ``query()`` returns
    ``UNKNOWN`` for cells outside the grid. Inside the grid we
    consult ``state_at`` (a sparse override map) — anything missing
    defaults to FREE.
    """

    def __init__(self, *, state_at: Dict[Tuple[int, int], CellState] = None,
                 origin_xy: Tuple[float, float] = (-5.0, -5.0),
                 size: Tuple[float, float] = (10.0, 10.0),
                 res: float = 0.1):
        self._state = state_at or {}
        self._origin = origin_xy
        self._size = size
        self._res = res
        self._width = int(size[0] / res)
        self._height = int(size[1] / res)

    def _in_bounds(self, ix: int, iy: int) -> bool:
        return 0 <= ix < self._width and 0 <= iy < self._height

    def query(self, world_xy):
        ix = int((world_xy[0] - self._origin[0]) / self._res)
        iy = int((world_xy[1] - self._origin[1]) / self._res)
        if not self._in_bounds(ix, iy):
            return CellState.UNKNOWN
        return self._state.get((ix, iy), CellState.FREE)

    def origin_xy(self): return self._origin
    def resolution_m(self): return self._res

    def frontier_cells(self, *, from_xy=None, k=10, prefer_near=None):
        from loco_x.occupancy import FrontierCandidate
        return [FrontierCandidate(
            world_xy=(1.0, 0.5), info_gain=20.0,
            travel_distance=1.12, score=20.0 / (1 + 1.12),
        )]

    def visited_fraction(self): return 0.42


# ── Registry composition ───────────────────────────────────────────────────
def test_registry_exposes_all_skills_in_namespace() -> None:
    """``make_skills(bundle)`` returns a namespace dict with every
    skill name the agent's system prompt advertises. The sandbox
    executes against this dict — anything missing here is a
    runtime ``unknown call``."""
    bundle = _make_bundle()
    # include_stdlib=False so we're asserting the *skill-only* surface;
    # the stdlib union is tested separately.
    ns = make_skills(bundle, include_stdlib=False)
    expected = {
        # Locomotion (D4)
        "goto", "goto_xy", "face", "stop",
        # Perception (D4 + D12)
        "find", "peek", "survey", "list_scene", "describe_view",
        # Exploration (D5)
        "next_frontier", "visited_fraction",
        # Meta (D11)
        "finish", "fail",
    }
    missing = expected - set(ns)
    assert not missing, f"skills missing from namespace: {missing}"
    # And the skill-only surface really is skill-only — no stdlib leak.
    assert "print" not in ns
    assert "range" not in ns


def test_registry_includes_stdlib_helpers_by_default() -> None:
    """The default namespace includes a minimal stdlib subset so the
    LLM can write ``for label in [...]: goto(label)`` without the
    sandbox rejecting ``range`` / ``print`` / etc. as unknown calls."""
    bundle = _make_bundle()
    ns = make_skills(bundle)   # include_stdlib defaults to True
    for helper in ("range", "len", "min", "max", "print",
                   "abs", "round", "enumerate", "zip"):
        assert helper in ns, f"stdlib helper missing: {helper}"


def test_skill_registry_returns_serializable_dicts() -> None:
    """Every skill returns a dict with ``status`` and serializable
    fields. Surfaces in two places: the agent runner echoes the dict
    into the next observation, and tests can JSON-encode the result
    for snapshot comparison."""
    import json
    bundle = _make_bundle(occ_provider=_FakeProvider())
    ns = make_skills(bundle)
    # Skills the LLM calls with no args / safe args; we don't actually
    # need the autonomy loop to be running.
    results = [
        ns["stop"](),
        ns["face"](0.0),
        ns["goto_xy"](1.0, 0.5),
        ns["list_scene"](),
        ns["next_frontier"](),
        ns["visited_fraction"](),
    ]
    for r in results:
        assert isinstance(r, dict)
        assert "status" in r
        # JSON round-trip ⇒ all values are serializable.
        json.dumps(r)


# ── Locomotion (D4) ─────────────────────────────────────────────────────────
def test_goto_pushes_task_with_label() -> None:
    """``goto('stove')`` enqueues a locomotion task carrying the
    label *and* the cached world_xy lookup. The label must already
    exist in the scene graph (D4 — that's what the
    ``error[unknown_label]`` path is for); the autonomy loop drains
    the queue and seeds the goal lock without re-doing the lookup."""
    bundle = _make_bundle(scene_nodes=[
        {"label": "stove", "world_xy": (2.0, 0.0),
         "last_seen": 1.0, "confidence": 0.81},
    ])
    ns = make_skills(bundle)
    r = ns["goto"]("stove")
    assert r["status"] == "queued"
    assert len(bundle["task_queue"]) == 1
    task = bundle["task_queue"][0]
    assert task["kind"] == "goto"
    assert task["label"] == "stove"
    assert task["world_xy"] == (2.0, 0.0)


def test_goto_xy_pushes_task_with_coordinates() -> None:
    """``goto_xy(+1.5, +0.7)`` enqueues a coord-mode task. Coords are
    in **world frame** per D9 — there is no relative goto."""
    bundle = _make_bundle(occ_provider=_FakeProvider())
    ns = make_skills(bundle)
    r = ns["goto_xy"](1.5, 0.7)
    assert r["status"] == "queued"
    task = bundle["task_queue"][0]
    assert task["kind"] == "goto_xy"
    assert task["xy"] == (1.5, 0.7)


def test_face_pushes_yaw_task_in_radians() -> None:
    """``face(math.pi/2)`` enqueues a yaw task. Radians per D9 —
    the observation renders degrees, but skills consume radians so
    there's no per-call unit confusion."""
    import math
    bundle = _make_bundle()
    ns = make_skills(bundle)
    r = ns["face"](math.pi / 2)
    assert r["status"] == "queued"
    task = bundle["task_queue"][0]
    assert task["kind"] == "face"
    assert abs(task["yaw_rad"] - math.pi / 2) < 1e-9


def test_stop_pushes_safe_stop_task() -> None:
    """``stop()`` enqueues a (0,0,0,1) command. Convenience skill —
    rarely needed but explicit for the LLM's mental model."""
    bundle = _make_bundle()
    ns = make_skills(bundle)
    r = ns["stop"]()
    assert r["status"] == "queued"
    task = bundle["task_queue"][0]
    assert task["kind"] == "stop"


# ── Locomotion: spatial-error context (D4) ──────────────────────────────────
def test_goto_xy_out_of_bounds_returns_error_with_closest_reachable() -> None:
    """``goto_xy`` to a cell well outside the provider's grid: the
    skill returns ``error[out_of_bounds]`` *before* enqueuing — no
    task is added. The error dict carries the requested XY and a
    suggested-recovery field per D4."""
    bundle = _make_bundle(occ_provider=_FakeProvider())
    ns = make_skills(bundle)
    r = ns["goto_xy"](999.0, 999.0)
    assert r["status"] == "error"
    assert r["error_kind"] == "out_of_bounds"
    assert r["target"]["world_xy"] == [999.0, 999.0]
    # No task should have been enqueued for an immediately-invalid call.
    assert bundle["task_queue"] == []


def test_goto_unknown_label_returns_error_listing_nearest_seen() -> None:
    """``goto('kettle')`` when no node matches: ``error[unknown_label]``
    with ``nearest_seen_labels`` populated from the scene graph. The
    LLM uses this to ``peek`` or retry with a related label."""
    bundle = _make_bundle(scene_nodes=[
        {"label": "stove",     "world_xy": (2.0, 0.0), "last_seen": 1.0},
        {"label": "microwave", "world_xy": (2.5, 0.5), "last_seen": 1.0},
    ])
    ns = make_skills(bundle)
    r = ns["goto"]("kettle")
    assert r["status"] == "error"
    assert r["error_kind"] == "unknown_label"
    nearest = r.get("nearest_seen_labels") or []
    assert "stove" in nearest or "microwave" in nearest
    assert bundle["task_queue"] == []


# ── Perception (D4 + D12) ───────────────────────────────────────────────────
def test_find_returns_node_when_label_seen() -> None:
    """``find('stove')`` queries the scene graph and returns the node
    info — read-only, no task queued."""
    bundle = _make_bundle(scene_nodes=[
        {"label": "stove", "world_xy": (2.0, 0.0),
         "last_seen": 1.0, "confidence": 0.81},
    ])
    ns = make_skills(bundle)
    r = ns["find"]("stove")
    assert r["status"] == "ok"
    assert r["value"]["label"] == "stove"
    assert r["value"]["world_xy"] == [2.0, 0.0]


def test_find_unknown_label_returns_error_listing_nearest_seen() -> None:
    """Symmetric with ``goto`` for the unknown-label case — but
    without enqueuing anything."""
    bundle = _make_bundle(scene_nodes=[
        {"label": "stove", "world_xy": (2.0, 0.0), "last_seen": 1.0},
    ])
    ns = make_skills(bundle)
    r = ns["find"]("kettle")
    assert r["status"] == "error"
    assert r["error_kind"] == "unknown_label"
    assert "stove" in (r.get("nearest_seen_labels") or [])


def test_peek_pushes_head_yaw_task() -> None:
    """``peek('left')`` enqueues a head-yaw task. The autonomy loop
    issues the head movement; SAM3 + heightmap re-observe naturally
    on the next perception tick."""
    bundle = _make_bundle()
    ns = make_skills(bundle)
    r = ns["peek"]("left")
    assert r["status"] == "queued"
    task = bundle["task_queue"][0]
    assert task["kind"] == "peek"
    assert task["direction"] == "left"


def test_peek_unknown_direction_returns_error() -> None:
    """A typoed direction returns ``error[unknown_direction]`` —
    surfaces typos quickly instead of silently doing nothing."""
    bundle = _make_bundle()
    ns = make_skills(bundle)
    r = ns["peek"]("zenith")
    assert r["status"] == "error"
    assert r["error_kind"] == "unknown_direction"
    assert bundle["task_queue"] == []


def test_survey_pushes_sweep_task_with_default_angles() -> None:
    """``survey()`` enqueues a multi-angle sweep with the default
    angle list. The task carries the angles so the autonomy loop
    can execute them sequentially."""
    bundle = _make_bundle()
    ns = make_skills(bundle)
    r = ns["survey"]()
    assert r["status"] == "queued"
    task = bundle["task_queue"][0]
    assert task["kind"] == "survey"
    assert len(task["angles_deg"]) == 6   # full sweep default


def test_survey_quick_uses_three_angles() -> None:
    """``survey(quick=True)`` enqueues the 3-angle peek (D12)."""
    bundle = _make_bundle()
    ns = make_skills(bundle)
    r = ns["survey"](quick=True)
    task = bundle["task_queue"][0]
    assert len(task["angles_deg"]) == 3


def test_scan_enqueues_one_long_running_task() -> None:
    """``scan(target_label='stove')`` enqueues a single rotation
    task. The autonomy loop's scan handler then runs the rotation
    over many ticks and early-exits when the target appears."""
    bundle = _make_bundle()
    ns = make_skills(bundle)
    r = ns["scan"](target_label="stove", max_revolutions=1.0)
    assert r["status"] == "queued"
    assert r["kind"] == "scan"
    assert len(bundle["task_queue"]) == 1
    task = bundle["task_queue"][0]
    assert task["kind"] == "scan"
    assert task["target_label"] == "stove"
    assert task["max_revolutions"] == 1.0
    assert task["direction"] == 1


def test_scan_direction_negative_is_clockwise() -> None:
    """``scan(direction=-1)`` clamps to -1 (CW); positive direction
    is the default (CCW). The autonomy handler reads this to pick
    the sign of _cmd[2]."""
    bundle = _make_bundle()
    ns = make_skills(bundle)
    ns["scan"](target_label="microwave", direction=-1)
    assert bundle["task_queue"][0]["direction"] == -1


def test_survey_custom_angles_overrides_defaults() -> None:
    """``survey(angles_deg=[-90, 0, 90])`` overrides both defaults."""
    bundle = _make_bundle()
    ns = make_skills(bundle)
    r = ns["survey"](angles_deg=[-90.0, 0.0, 90.0])
    task = bundle["task_queue"][0]
    assert task["angles_deg"] == [-90.0, 0.0, 90.0]


def test_list_scene_returns_snapshot() -> None:
    """``list_scene()`` is read-only — returns a serialized snapshot
    of the scene graph. Used by the LLM as an "escape hatch" when
    the filtered observation hides nodes it wants to see."""
    bundle = _make_bundle(scene_nodes=[
        {"label": "stove", "world_xy": (2.0, 0.0),
         "last_seen": 1.0, "confidence": 0.81},
        {"label": "microwave", "world_xy": (2.5, 0.5),
         "last_seen": 1.0, "confidence": 0.7},
    ])
    ns = make_skills(bundle)
    r = ns["list_scene"]()
    assert r["status"] == "ok"
    assert len(r["value"]) == 2
    assert r["value"][0]["label"] == "stove"


def test_describe_view_stub_returns_disabled_when_no_vlm() -> None:
    """LA-1's ``describe_view`` is a stub — the real VLM client lands
    in LA-4. The stub returns ``error[vlm_disabled]`` so the agent
    knows the skill exists but isn't wired yet, rather than the
    sandbox rejecting an unknown call."""
    bundle = _make_bundle()
    ns = make_skills(bundle)
    r = ns["describe_view"]()
    assert r["status"] == "error"
    assert r["error_kind"] == "vlm_disabled"


# ── Exploration (D5) ────────────────────────────────────────────────────────
def test_next_frontier_dispatches_to_provider() -> None:
    """``next_frontier()`` returns the top-ranked frontier from the
    occupancy provider. Read-only — no task queued."""
    bundle = _make_bundle(occ_provider=_FakeProvider())
    ns = make_skills(bundle)
    r = ns["next_frontier"]()
    assert r["status"] == "ok"
    assert r["value"]["world_xy"] == [1.0, 0.5]


def test_next_frontier_prefer_near_passes_anchor_list() -> None:
    """D14.1 — ``prefer_near=['countertop']`` is forwarded to the
    provider. We can't easily test the boost arithmetic from here
    (that's covered in occupancy/test_semantic_and_variance.py), but
    we can assert the skill doesn't reject the kwarg."""
    bundle = _make_bundle(occ_provider=_FakeProvider())
    ns = make_skills(bundle)
    r = ns["next_frontier"](prefer_near=["countertop"])
    assert r["status"] == "ok"


def test_next_frontier_no_frontiers_returns_error() -> None:
    """When the provider returns ``[]`` (fully-known map, USD case),
    the skill surfaces ``error[no_frontiers]`` so the agent can
    consider Case A failure (D11)."""
    class _NoFrontiers(_FakeProvider):
        def frontier_cells(self, *, from_xy=None, k=10, prefer_near=None):
            return []
    bundle = _make_bundle(occ_provider=_NoFrontiers())
    ns = make_skills(bundle)
    r = ns["next_frontier"]()
    assert r["status"] == "error"
    assert r["error_kind"] == "no_frontiers"
    assert "visited_fraction" in r


def test_visited_fraction_dispatches_to_provider() -> None:
    bundle = _make_bundle(occ_provider=_FakeProvider())
    ns = make_skills(bundle)
    r = ns["visited_fraction"]()
    assert r["status"] == "ok"
    assert abs(r["value"] - 0.42) < 1e-9


# ── Meta (D11) ──────────────────────────────────────────────────────────────
def test_finish_sets_done_flag() -> None:
    """``finish('reached microwave')`` sets the agent's stop flag and
    records the task result. The runner picks this up and unwinds."""
    bundle = _make_bundle()
    ns = make_skills(bundle)
    r = ns["finish"]("reached microwave")
    assert r["status"] == "ok"
    assert bundle["agent_should_stop"] is True
    assert bundle["task_result_status"] == "succeeded"
    assert "reached microwave" in (bundle["task_result_reason"] or "")


def test_fail_sets_done_flag_with_reason() -> None:
    """``fail('microwave not in scene')`` sets the stop flag with a
    failed status. Per D11 the runner safe-stops _cmd and preserves
    FSM mode for postmortem."""
    bundle = _make_bundle()
    ns = make_skills(bundle)
    r = ns["fail"]("microwave not in scene after 0.92 coverage")
    assert r["status"] == "ok"
    assert bundle["agent_should_stop"] is True
    assert bundle["task_result_status"] == "failed"
    assert "0.92" in (bundle["task_result_reason"] or "")
