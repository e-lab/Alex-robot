"""Locomotion skills: goto, goto_xy, face, stop (D4).

These skills don't drive the robot directly — they push a task onto
``bundle["task_queue"]`` and return ``status: queued``. The autonomy
loop drains the queue every tick: a ``goto(label)`` task is dispatched
to the FSM by looking up the label in the scene graph and seeding the
goal lock; a ``goto_xy(x, y)`` task seeds the goal lock directly; a
``face(yaw_rad)`` task rotates in place; ``stop()`` forces the safe-stop
``_cmd``.

Splitting the *decide-to-go* (skill) from the *go* (autonomy loop)
keeps the LLM-callable surface deterministic — no skill ever sleeps,
no skill ever blocks the agent thread on a sim tick. LA-5 will add a
*blocking* wrapper that awaits the result; LA-1 keeps everything
non-blocking so tests stay pure-Python.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional

from ._errors import error_dict, nearest_seen_labels, ok_dict, queued_dict


def _scene_lookup(bundle: dict, label: str) -> Optional[Dict[str, Any]]:
    """Return the first scene_nodes entry whose label matches, or
    ``None``. Recency-based lookup is the autonomy loop's job — for
    the skill we only need to know whether the label was ever seen."""
    for node in bundle.get("scene_nodes") or []:
        if node.get("label") == label:
            return node
    return None


def make_locomotion_skills(bundle: dict) -> Dict[str, Callable[..., Dict[str, Any]]]:
    """Build the four locomotion closures over the given bundle."""

    def goto(label: str) -> Dict[str, Any]:
        """Walk to a named scene-graph node.

        On unknown label, returns the D4 ``error[unknown_label]`` dict
        with ``nearest_seen_labels`` so the agent can retry with a
        related name or ``peek`` to extend coverage.
        """
        node = _scene_lookup(bundle, label)
        if node is None:
            return error_dict(
                error_kind="unknown_label",
                message=f"'{label}' not in scene graph",
                bundle=bundle,
                target_label=label,
                nearest_seen_labels=nearest_seen_labels(bundle),
                suggested_recovery="try_peek",
            )
        target_xy = node.get("world_xy")
        # Idempotent: if Phase 1-4 has already reached this target
        # (FSM is ARRIVED on the same label) the goto is a no-op.
        # Returns status=ok so the runner records "you're already
        # there" rather than queuing a redundant walk that the
        # planner will refuse with NO PATH or that will immediately
        # re-trigger ARRIVED.
        fsm_raw = str(bundle.get("fsm_mode_raw") or "").lower()
        if fsm_raw == "arrived" and bundle.get("goal_label") == label:
            return ok_dict(
                value={"label": label, "already_at_target": True},
                message=f"already ARRIVED at '{label}'; call finish() to end the task",
            )
        bundle["task_queue"].append({
            "kind": "goto",
            "label": label,
            "world_xy": tuple(target_xy) if target_xy is not None else None,
        })
        return queued_dict(kind="goto", label=label)

    def goto_xy(x: float, y: float) -> Dict[str, Any]:
        """Walk to a world XY. Rejects out-of-bounds inputs before
        enqueueing — saves the autonomy loop a useless replan tick.

        The bounds check uses the provider's ``query()`` Protocol
        method, which returns ``UNKNOWN`` for cells outside its grid
        (per the LA-0a out-of-bounds contract). We treat that as
        ``out_of_bounds`` here, *not* as "go explore" — the LLM should
        call ``next_frontier()`` for that.
        """
        from loco_x.occupancy import CellState  # local: avoid import cycle in tests

        target = (float(x), float(y))
        provider = bundle.get("occ_provider")
        if provider is not None:
            state = provider.query(target)
            # Out-of-bounds → CellState.UNKNOWN regardless of provider.
            # Inside-bounds OBSTACLE also rejects, but with a different
            # error_kind so the agent observation is more informative.
            if state == CellState.UNKNOWN:
                # Heuristic: if the target is within the grid bounds,
                # we'd see FREE/OBSTACLE/UNKNOWN-but-near-something.
                # Treating UNKNOWN from a query as "out of bounds OR
                # never observed" is a conservative reject — the LLM
                # can call ``next_frontier`` to find a reachable
                # unknown cell explicitly.
                return error_dict(
                    error_kind="out_of_bounds",
                    message=(
                        f"world XY {target} is outside the known map or "
                        f"in an unobserved region"
                    ),
                    bundle=bundle,
                    target_xy=target,
                    suggested_recovery="try_next_frontier",
                )
            if state == CellState.OBSTACLE:
                return error_dict(
                    error_kind="blocked",
                    message=f"world XY {target} is on an obstacle cell",
                    bundle=bundle,
                    target_xy=target,
                    suggested_recovery="try_next_frontier",
                )
        bundle["task_queue"].append({
            "kind": "goto_xy",
            "xy": target,
        })
        return queued_dict(kind="goto_xy", xy=list(target))

    def face(yaw_rad: float) -> Dict[str, Any]:
        """Rotate in place to the given world-frame yaw (radians).

        Returns ``status="ok"`` (already-there) instead of ``queued``
        when the robot is already within 5° of the target yaw —
        prevents the LLM from re-issuing identical face commands when
        ``last_action`` already reflects the rotation as complete.
        """
        target = float(yaw_rad)
        pose = bundle.get("robot_pose") or {}
        current = float(pose.get("yaw_rad", 0.0))
        # Smallest signed angular diff in [-pi, +pi].
        err = math.atan2(math.sin(target - current), math.cos(target - current))
        if abs(err) < math.radians(5.0):
            # Robot is already there; don't enqueue another rotation.
            return ok_dict(
                value={"yaw_rad": target, "already_at_target": True},
                message=f"already within {math.degrees(abs(err)):.1f}° of target yaw",
            )
        bundle["task_queue"].append({
            "kind": "face",
            "yaw_rad": target,
        })
        return queued_dict(kind="face", yaw_rad=target)

    def stop() -> Dict[str, Any]:
        """Force the safe-stop command for one tick. Rarely useful
        on its own, but explicit in the skill set so the LLM can
        emit it after a finish() without leaving _cmd stale."""
        bundle["task_queue"].append({"kind": "stop"})
        return queued_dict(kind="stop")

    return {
        "goto": goto,
        "goto_xy": goto_xy,
        "face": face,
        "stop": stop,
    }


__all__ = ["make_locomotion_skills"]
