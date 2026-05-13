"""Agent-facing observation builder (D2 + D9 + D11 + D13).

Renders the structured text the LLM sees each turn. The contract is
the example block in plan §D2: a one-line coordinate convention
header, the robot's pose in degrees, the FSM mode, the filtered scene
graph with dual coordinates (world + rel), an exploration summary,
optional path-staleness signal, and the previous turn's action result.

Design choices captured by tests in
:mod:`tests.loco_x.perception.test_observation`:

* The function is **pure** — no clock reads inside; ``now`` is
  injected. Per D13, the observation is a *snapshot* taken at
  build-time; world state continues evolving during the LLM call but
  the returned string doesn't change.
* Rel-coords (D9) are computed by a 5-line projection: rotate the
  (world − robot) vector by −yaw, then sign-classify into
  ``forward / behind / left / right``. The ``rel=`` segment is for
  the LLM's *reasoning only*; skills always consume world coords.
* Filtering (D2) is delegated to :func:`filter_scene_nodes` so the
  agent runner can reuse it (e.g. for matching scene-graph labels
  against ``max_path_staleness``).
* Numeric formatting is pinned by the tests — a drift in
  ``+1.23`` / ``+30deg`` / ``forward 2.0m`` would change every
  observation silently, so we test the substrings rather than the
  bytes.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .scene_filter import FilterParams, filter_scene_nodes


# ── Numeric formatters (kept tiny + locked by tests) ────────────────────────
def _f(v: float) -> str:
    """Signed float with two decimal places. Matches the ``+1.23`` /
    ``-0.45`` formatting in the D2 example. Two decimals is enough
    for 1 cm precision at a 5 cm grid — finer would be misleading."""
    return f"{v:+.2f}"


def _pct(v: float) -> str:
    return f"{v * 100:.0f}%"


# ── D9 rel-coord projection ────────────────────────────────────────────────
def _rel_coords(
    *, world_xy: Tuple[float, float],
    robot_xy: Tuple[float, float],
    robot_yaw_rad: float,
) -> Tuple[str, str]:
    """Project ``world_xy`` into the robot's body frame.

    Returns ``(forward_axis_text, lateral_axis_text)`` e.g.
    ``("forward 2.1m", "right 0.9m")``. The robot's heading is the
    body-frame +X axis; +Y is to the **left** of the body (right-
    handed world, yaw=0 means facing +X).

    Rotation: take the world-frame vector (target − robot) and rotate
    by ``-yaw`` to get body-frame coordinates::

        bx =  cos(-yaw)*dx - sin(-yaw)*dy =  cos(yaw)*dx + sin(yaw)*dy
        by =  sin(-yaw)*dx + cos(-yaw)*dy = -sin(yaw)*dx + cos(yaw)*dy

    Sign-classify each axis. The lateral axis is "left" for positive
    body-Y (the right-handed up direction is +Z; +Y is left when
    looking forward).
    """
    dx = world_xy[0] - robot_xy[0]
    dy = world_xy[1] - robot_xy[1]
    cyaw = math.cos(robot_yaw_rad)
    syaw = math.sin(robot_yaw_rad)
    bx = cyaw * dx + syaw * dy        # forward axis
    by = -syaw * dx + cyaw * dy       # left axis (right-handed)

    forward = "forward" if bx >= 0 else "behind"
    lateral = "left" if by >= 0 else "right"
    return f"{forward} {abs(bx):.1f}m", f"{lateral} {abs(by):.1f}m"


# ── Section renderers ──────────────────────────────────────────────────────
def _header_line() -> str:
    return (
        "coordinates: world frame, meters; yaw=0 means facing +X; "
        '"forward/right/left/behind" in rel= is measured from the '
        "robot's current heading."
    )


def _pose_line(robot_xy: Tuple[float, float], yaw_rad: float) -> str:
    yaw_deg = math.degrees(yaw_rad)
    return (
        f"pose: x={_f(robot_xy[0])} y={_f(robot_xy[1])} "
        f"yaw={yaw_deg:+.0f}deg"
    )


def _scene_node_line(
    node: Dict[str, Any],
    *,
    robot_xy: Tuple[float, float],
    robot_yaw_rad: float,
    now: float,
) -> str:
    label = str(node.get("label", "?"))
    xy = node.get("world_xy")
    last_seen = float(node.get("last_seen", 0.0))
    age = now - last_seen
    if xy is None:
        return f"  - {label:<14} world=(unknown)"
    fw, lat = _rel_coords(
        world_xy=xy, robot_xy=robot_xy, robot_yaw_rad=robot_yaw_rad,
    )
    conf = float(node.get("confidence", 0.0))
    return (
        f"  - {label:<14} "
        f"world=({_f(xy[0])},{_f(xy[1])})  "
        f"rel=({fw}, {lat})  "
        f"last_seen={age:.1f}s  conf={conf:.2f}"
    )


def _hidden_line(hidden_count: int) -> str:
    suffix = "node" if hidden_count == 1 else "nodes"
    return (
        f"  ({hidden_count} {suffix} hidden: distance>radius and "
        f"last_seen>recency — call list_scene() to see all)"
    )


def _last_action_lines(last_action: Dict[str, Any]) -> List[str]:
    """Render the previous turn's skill result. Two shapes:

    * ``status: ok`` — show the kind + a free-text summary if present.
    * ``status: error`` — show the error kind, message, and the
      suggested recovery hint (D4).
    """
    status = last_action.get("status", "?")
    if status == "error":
        kind = last_action.get("error_kind", "?")
        msg = last_action.get("message", "")
        rec = last_action.get("suggested_recovery")
        line = f"last action: error[{kind}]: {msg}"
        if rec:
            line += f" (suggested: {rec})"
        return [line]
    summary = last_action.get("summary") or last_action.get("kind") or "ok"
    return [f"last action: {summary}"]


# ── Top-level builder ──────────────────────────────────────────────────────
def build_observation(
    bundle: Dict[str, Any],
    *,
    now: float,
    filter_params: Optional[FilterParams] = None,
    path_freshness_s: float = 15.0,
) -> str:
    """Build the agent observation string for one tick.

    ``bundle`` is the autonomy bundle; we read a small slice
    (``robot_pose``, ``scene_nodes``, ``occ_provider``, ``goal_label``,
    ``last_action``, ``path``, ``fsm_mode``). ``now`` is injected per
    D8 / D13.
    """
    pose = bundle.get("robot_pose") or {"xy": (0.0, 0.0), "yaw_rad": 0.0}
    robot_xy = tuple(pose.get("xy") or (0.0, 0.0))
    robot_yaw = float(pose.get("yaw_rad") or 0.0)
    fsm = bundle.get("fsm_mode") or "IDLE"
    goal_label = bundle.get("goal_label")

    # D2 filter — always-keep the active goal label so the LLM never
    # loses sight of the thing it's acting on.
    active_labels: List[str] = []
    if goal_label:
        active_labels.append(str(goal_label))
    filtered = filter_scene_nodes(
        bundle.get("scene_nodes") or [],
        robot_xy=robot_xy,
        now=now,
        active_labels=active_labels,
        params=filter_params,
    )

    lines: List[str] = []
    lines.append(_header_line())
    # Surface the task target each turn so the LLM doesn't forget
    # what it's looking for. The autonomy script populates
    # bundle["task_target"] from cfg.autonomy.target at startup.
    task_target = bundle.get("task_target")
    if task_target:
        lines.append(f"task: walk to a scene-graph node labelled '{task_target}'")
    lines.append(_pose_line(robot_xy, robot_yaw))
    lines.append(f"fsm: {fsm}")

    # Scene graph block.
    if filtered.kept:
        params = filter_params or FilterParams()
        lines.append(
            f"scene_graph (filtered, {len(filtered.kept)} nodes "
            f"within {params.radius_m:.0f}m or seen <{params.recency_s:.0f}s ago):"
        )
        for node in filtered.kept:
            lines.append(_scene_node_line(
                node, robot_xy=robot_xy,
                robot_yaw_rad=robot_yaw, now=now,
            ))
        if filtered.hidden_count > 0:
            lines.append(_hidden_line(filtered.hidden_count))
    else:
        lines.append("scene_graph: (empty)")
        if filtered.hidden_count > 0:
            lines.append(_hidden_line(filtered.hidden_count))

    # Exploration block (if provider is wired).
    provider = bundle.get("occ_provider")
    if provider is not None:
        lines.append("exploration:")
        try:
            vf = float(provider.visited_fraction())
            lines.append(f"  visited {_pct(vf)} of accessible grid")
        except Exception:                # pragma: no cover - defensive
            pass

        # Top frontier (optional — USD provider returns []).
        try:
            cands = provider.frontier_cells(from_xy=robot_xy, k=1)
        except Exception:                # pragma: no cover - defensive
            cands = []
        if cands:
            c = cands[0]
            fw, lat = _rel_coords(
                world_xy=c.world_xy, robot_xy=robot_xy,
                robot_yaw_rad=robot_yaw,
            )
            lines.append(
                f"  next frontier suggestion: "
                f"world=({_f(c.world_xy[0])},{_f(c.world_xy[1])})  "
                f"rel=({fw}, {lat})"
            )
        else:
            # D11 Case A signal — surface directly so the LLM doesn't
            # have to call next_frontier() to learn coverage plateaued.
            lines.append(
                "  next_frontier: error[no_frontiers] — no reachable "
                "unknown regions remain"
            )

    # Path-staleness signal (D10): max staleness on the current path
    # cells, but only surfaced when it exceeds the freshness window
    # so the LLM doesn't burn tokens on "all-fresh" status lines.
    path = bundle.get("path")
    if path and provider is not None:
        try:
            staleness_s, stalest_xy = provider.max_path_staleness(path)
        except Exception:                # pragma: no cover - defensive
            staleness_s, stalest_xy = 0.0, None
        if staleness_s > path_freshness_s and stalest_xy is not None:
            idx = bundle.get("path_index", 0)
            lines.append(
                f"path_staleness: max {staleness_s:.0f}s at "
                f"world=({_f(stalest_xy[0])},{_f(stalest_xy[1])}) "
                f"on waypoint {idx + 1}/{len(path)} "
                f"(path window={path_freshness_s:.0f}s; consider "
                f"peek('forward') before goto)"
            )

    # Last-action echo.
    last_action = bundle.get("last_action")
    if last_action:
        lines.extend(_last_action_lines(last_action))

    return "\n".join(lines)


__all__ = ["build_observation"]
