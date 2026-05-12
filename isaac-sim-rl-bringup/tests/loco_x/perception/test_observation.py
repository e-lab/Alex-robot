"""Tests for the agent-facing observation builder (LA-2, D2 + D9 + D13).

The observation is the text block the LLM sees each turn. The
contract is the example in D2 of the plan:

    coordinates: world frame, meters; yaw=0 means facing +X; ...
    pose: x=+1.23 y=-0.45 yaw=+30deg
    fsm: IDLE
    scene_graph (filtered, 4 nodes within 5m or seen <30s ago):
      - stove          world=(+3.1,+0.2)   rel=(forward 2.1m, right 0.9m) ...
      ...
      (K nodes hidden ...)
    exploration:
      visited 41% of accessible grid
      next frontier suggestion: world=(+0.8,+2.6)  rel=(...)
    path_staleness: max 22s at world=(...) on waypoint 3/5 (...)
    last action: goto("stove") -> ARRIVED at world=(...) after 47.2s

These tests pin:

* the dual-coord (D9) rendering — world + rel for every node, with
  the rel-coord yaw-sign canary,
* the D2 filter integration (kept count, hidden line),
* the D13 invariant (observation is a *snapshot*; doesn't peek at
  state mutating during the LLM call),
* the D11 signals woven into the text (visited_fraction,
  no_frontiers, path_staleness, fail context),
* the under-1k-token budget on a 30-node scene.
"""
from __future__ import annotations

import math

import pytest

from loco_x.occupancy import CellState
from loco_x.perception.observation import build_observation


def _node(label, xy, last_seen=0.0, confidence=0.5):
    return {
        "label": label,
        "world_xy": tuple(xy),
        "last_seen": float(last_seen),
        "confidence": float(confidence),
    }


def _bundle(
    *,
    robot_xy=(0.0, 0.0),
    robot_yaw_rad=0.0,
    fsm="IDLE",
    scene_nodes=None,
    occ_provider=None,
    goal_label=None,
    last_action=None,
    path=None,
    path_index=0,
):
    return {
        "robot_pose": {"xy": robot_xy, "yaw_rad": robot_yaw_rad},
        "fsm_mode": fsm,
        "scene_nodes": scene_nodes or [],
        "occ_provider": occ_provider,
        "goal_label": goal_label,
        "last_action": last_action,
        "path": path,
        "path_index": path_index,
        "agent_should_stop": False,
        "task_result_status": None,
        "task_result_reason": None,
        "task_queue": [],
    }


class _StubProvider:
    """Stub occupancy provider exposing only what the observation
    builder reads. Real :class:`HeightMapProvider` satisfies the same
    surface; we don't need the planner here."""

    def __init__(
        self,
        *,
        visited_fraction=0.41,
        frontier=(0.8, 2.6),
        path_staleness=None,
    ):
        self._vf = visited_fraction
        self._frontier = frontier
        self._path_staleness = path_staleness or (0.0, None)

    def visited_fraction(self):
        return self._vf

    def frontier_cells(self, *, from_xy=None, k=1, prefer_near=None):
        if self._frontier is None:
            return []
        from loco_x.occupancy import FrontierCandidate
        return [FrontierCandidate(
            world_xy=self._frontier, info_gain=20.0,
            travel_distance=2.7, score=7.4,
        )]

    def max_path_staleness(self, path_xys):
        return self._path_staleness

    def path_invalidated_by_new_obstacle(self, path_xys):
        return None


# ── Header + pose ──────────────────────────────────────────────────────────
def test_observation_header_pins_coordinate_convention() -> None:
    """The first line must state the world-frame convention so the
    LLM doesn't re-derive it from per-turn observations."""
    obs = build_observation(_bundle(), now=0.0)
    assert "world frame" in obs
    assert "+X" in obs
    assert "yaw=0" in obs


def test_observation_pose_line_uses_degrees() -> None:
    """The pose line shows yaw in degrees (D9: degrees in observation,
    radians in skill args). 30° must render as ``+30deg`` exactly."""
    obs = build_observation(
        _bundle(robot_xy=(1.23, -0.45), robot_yaw_rad=math.radians(30)),
        now=0.0,
    )
    # Match the renderable substring — exact format pinned for the LLM.
    assert "x=+1.23" in obs
    assert "y=-0.45" in obs
    assert "+30deg" in obs


def test_observation_fsm_line_present() -> None:
    """The current FSM mode appears so the LLM knows whether autonomy
    is mid-action."""
    obs = build_observation(_bundle(fsm="APPROACH"), now=0.0)
    assert "fsm: APPROACH" in obs


# ── Scene graph: kept count, world/rel coords, hidden line ─────────────────
def test_observation_renders_scene_nodes_with_world_and_rel_coords() -> None:
    """Every kept node shows both ``world=(x,y)`` and
    ``rel=(forward Nm, right Nm)`` — D9 contract."""
    nodes = [
        _node("stove", (2.0, 0.0), last_seen=0.0, confidence=0.81),
    ]
    obs = build_observation(
        _bundle(scene_nodes=nodes, robot_xy=(0.0, 0.0),
                robot_yaw_rad=0.0),
        now=0.0,
    )
    # The stove is 2 m straight ahead of the robot at world (+2, 0).
    assert "stove" in obs
    assert "world=(+2.0,+0.0)" in obs or "world=(+2.00,+0.00)" in obs
    assert "rel=(forward 2.0m" in obs or "rel=(forward 2.00m" in obs


def test_observation_renders_hidden_count_when_filter_drops_nodes() -> None:
    """If the D2 filter hides nodes, the ``(K nodes hidden ...)`` line
    must appear so the LLM can call ``list_scene()`` if it needs more."""
    nodes = [
        _node("stove", (1.0, 0.0), last_seen=0.0),     # close, kept
        _node("attic_lamp", (20.0, 0.0), last_seen=0.0),  # far + old, hidden
    ]
    obs = build_observation(
        _bundle(scene_nodes=nodes),
        now=200.0,  # past recency window
    )
    assert "stove" in obs
    assert "attic_lamp" not in obs
    assert "1 nodes hidden" in obs or "1 node hidden" in obs


def test_observation_under_token_limit_for_30_node_scene() -> None:
    """The D2 cap (max_nodes=20) + token-density target keeps the
    observation under ~1 k tokens for a 30-node scene. Tokens are
    hard to count without a tokenizer; we proxy with character count
    (~4 chars/token average) and bound at 4 k characters."""
    nodes = [
        _node(f"node_{i:02d}", (float(i) * 0.4, 0.0), last_seen=199.0)
        for i in range(30)
    ]
    obs = build_observation(
        _bundle(scene_nodes=nodes), now=200.0,
    )
    assert len(obs) < 4_000, f"observation grew to {len(obs)} chars"


# ── D9 rel-coord canary ────────────────────────────────────────────────────
def test_rel_coords_match_world_at_yaw_zero() -> None:
    """Node at world=(+1, 0) with yaw=0 must render as ``forward 1m``."""
    obs = build_observation(
        _bundle(
            robot_xy=(0.0, 0.0), robot_yaw_rad=0.0,
            scene_nodes=[_node("ahead", (1.0, 0.0))],
        ),
        now=0.0,
    )
    assert "forward 1.0m" in obs or "forward 1.00m" in obs


def test_rel_coords_match_world_under_rotation() -> None:
    """**The yaw-sign canary** (per D9). A node that was ``forward``
    when the robot faced +X must become ``right`` when the robot
    rotates +90° (now facing +Y).

    This catches the sign error that always shows up the first time
    somebody writes a 2D rotation. Sign convention:
        yaw=0  → robot faces +X (world)
        yaw=+90° → robot faces +Y (world)
    So a world point at (+1, 0) is *to the robot's right* when the
    robot is looking north (+Y).
    """
    node = _node("east_marker", (1.0, 0.0))
    # Robot faces +X.
    obs_zero = build_observation(
        _bundle(robot_yaw_rad=0.0, scene_nodes=[node]), now=0.0
    )
    assert "forward 1.0m" in obs_zero or "forward 1.00m" in obs_zero

    # Robot rotates +90° → now facing +Y. The east_marker is to the
    # robot's right.
    obs_plus_90 = build_observation(
        _bundle(robot_yaw_rad=math.radians(90), scene_nodes=[node]),
        now=0.0,
    )
    assert "right 1.0m" in obs_plus_90 or "right 1.00m" in obs_plus_90


def test_rel_coords_classify_all_four_directions() -> None:
    """A cross of four nodes at the cardinal world directions, viewed
    with the robot facing +X, must classify forward / behind / left
    / right correctly. Locks the sign of each axis projection."""
    nodes = [
        _node("east",  (+1.0, 0.0)),   # forward
        _node("west",  (-1.0, 0.0)),   # behind
        _node("north", (0.0, +1.0)),   # left
        _node("south", (0.0, -1.0)),   # right
    ]
    obs = build_observation(
        _bundle(scene_nodes=nodes, robot_yaw_rad=0.0),
        now=0.0,
    )
    # Each node should pair with the correct direction word.
    east_line = next(line for line in obs.splitlines() if "east " in line)
    west_line = next(line for line in obs.splitlines() if "west " in line)
    north_line = next(line for line in obs.splitlines() if "north " in line)
    south_line = next(line for line in obs.splitlines() if "south " in line)
    assert "forward" in east_line, east_line
    assert "behind" in west_line, west_line
    assert "left" in north_line, north_line
    assert "right" in south_line, south_line


def test_rel_coords_use_degrees_only_in_observation_not_radians() -> None:
    """Per D9: degrees in the observation, radians in skill args. The
    text must not show ``rad``; yaw must show ``deg``."""
    obs = build_observation(
        _bundle(robot_yaw_rad=math.radians(45)),
        now=0.0,
    )
    assert "deg" in obs
    assert "rad" not in obs.lower()


# ── D2 active-target survives filter ───────────────────────────────────────
def test_active_goal_label_survives_filter_when_far_and_stale() -> None:
    """If the agent has a ``goal_label`` lock, the corresponding node
    must appear even if 10 m away and ancient. The LLM is in the
    middle of acting on it."""
    nodes = [
        _node("microwave", (15.0, 0.0), last_seen=0.0),
    ]
    obs = build_observation(
        _bundle(scene_nodes=nodes, goal_label="microwave"),
        now=500.0,
    )
    assert "microwave" in obs


# ── D11 + D13 signals ──────────────────────────────────────────────────────
def test_observation_includes_visited_fraction_when_provider_present() -> None:
    """``exploration: visited X% of accessible grid``. The LLM uses
    this for the D11 Case A / C decision (give up vs keep peeking)."""
    obs = build_observation(
        _bundle(occ_provider=_StubProvider(visited_fraction=0.41)),
        now=0.0,
    )
    assert "visited 41%" in obs or "visited 41.0%" in obs


def test_observation_renders_next_frontier_when_available() -> None:
    """The provider's top frontier appears with both world and rel
    coords — same dual-view as scene nodes (D9)."""
    obs = build_observation(
        _bundle(occ_provider=_StubProvider(frontier=(0.8, 2.6))),
        now=0.0,
    )
    assert "next frontier" in obs
    assert "world=(+0.8" in obs
    # 2.6 m left, 0.8 m forward from origin with yaw=0.
    assert "rel=" in obs


def test_observation_renders_no_frontiers_warning_when_provider_returns_empty() -> None:
    """D11 Case A surfaces directly in the observation when frontier
    is empty: the LLM knows coverage has plateaued."""
    obs = build_observation(
        _bundle(occ_provider=_StubProvider(frontier=None,
                                           visited_fraction=0.92)),
        now=0.0,
    )
    assert "no reachable unknown" in obs or "no_frontiers" in obs


def test_observation_renders_path_staleness_when_above_freshness_window() -> None:
    """When a path cell has gone stale beyond ``path_freshness_s`` (15
    s), the observation surfaces ``path_staleness: max NNs at ...``
    so the agent can ``peek`` before walking. The path is NOT
    invalidated by staleness — it's a signal, not an action."""
    obs = build_observation(
        _bundle(
            occ_provider=_StubProvider(
                path_staleness=(22.0, (2.1, 0.3)),
            ),
            path=[(0.0, 0.0), (1.0, 0.0), (2.1, 0.3)],
            path_index=0,
        ),
        now=0.0,
        path_freshness_s=15.0,
    )
    assert "path_staleness" in obs
    assert "22" in obs   # 22 seconds


def test_observation_omits_path_staleness_when_below_freshness_window() -> None:
    """If everything on the path is fresh, the staleness line is
    suppressed so the LLM doesn't burn tokens reading "everything is
    fine" lines."""
    obs = build_observation(
        _bundle(
            occ_provider=_StubProvider(path_staleness=(3.0, (1.0, 0.0))),
            path=[(0.0, 0.0), (1.0, 0.0)],
        ),
        now=0.0,
        path_freshness_s=15.0,
    )
    assert "path_staleness" not in obs


# ── Last-action echo ───────────────────────────────────────────────────────
def test_observation_renders_last_action_when_present() -> None:
    """The runner stores the previous turn's skill result in
    ``bundle["last_action"]``; the observation echoes it verbatim
    (D4 → human-readable rendering of the spatial-error-context
    dict)."""
    last = {
        "status": "ok",
        "kind": "goto",
        "label": "stove",
        "summary": "ARRIVED at world=(+2.95,+0.18) after 47.2s",
    }
    obs = build_observation(_bundle(last_action=last), now=0.0)
    assert "last action" in obs
    assert "ARRIVED" in obs


def test_observation_renders_last_action_error_with_suggested_recovery() -> None:
    """When the last skill failed with a D4 error dict, the
    observation includes the error_kind, message, and
    suggested_recovery hint."""
    last = {
        "status": "error",
        "error_kind": "blocked",
        "message": "path blocked by 'table' at world=(+2.1,+0.4)",
        "suggested_recovery": "try_next_frontier",
    }
    obs = build_observation(_bundle(last_action=last), now=0.0)
    assert "blocked" in obs
    assert "try_next_frontier" in obs


def test_observation_omits_last_action_section_when_none() -> None:
    """First turn: no last action. The section is suppressed."""
    obs = build_observation(_bundle(last_action=None), now=0.0)
    assert "last action" not in obs


# ── D13 snapshot invariant ─────────────────────────────────────────────────
def test_observation_is_a_snapshot_not_a_live_view() -> None:
    """Per D13: the observation reflects bundle state at build time.
    If the bundle changes after ``build_observation`` returns, the
    returned string is unchanged. (Strings are immutable in Python, so
    this is structural; we assert by mutating the bundle and showing
    the string is unchanged.)"""
    bundle = _bundle(scene_nodes=[_node("stove", (1.0, 0.0))])
    obs1 = build_observation(bundle, now=0.0)
    bundle["scene_nodes"].append(_node("sink", (-1.0, 0.0)))
    # The previously-built observation does NOT pick up the new node.
    assert "sink" not in obs1
    # A fresh build does.
    obs2 = build_observation(bundle, now=0.0)
    assert "sink" in obs2
