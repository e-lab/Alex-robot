"""Helpers for the D4 spatial-error-context dict shape.

Every skill's failure return follows this shape:

    {
        "status": "error",
        "error_kind": <one of the documented kinds>,
        "message": <one-line human-readable>,
        # spatial fields (present when applicable):
        "target": {"label": ..., "world_xy": [x, y] | None},
        "robot_pose": {"world_xy": [x, y], "yaw_deg": float},
        "blocker": {"label": ..., "world_xy": [x, y]} | None,
        ...
        "suggested_recovery": "try_peek" | "try_next_frontier" |
                             "try_goto_xy" | "give_up" | None,
    }

The helper :func:`error_dict` accepts kwargs and builds the dict; that
keeps every skill's failure path one line of code and lets the agent
runner trust the shape. The agent observation echoes these dicts
verbatim into ``last action:`` lines.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _robot_pose_field(bundle: dict) -> Optional[Dict[str, Any]]:
    """Render ``bundle["robot_pose"]`` for the error dict.

    The runner sets ``robot_pose`` once per tick; skills read it
    rather than recomputing. Returns ``None`` only in tests that
    don't set the field — production bundles always have it.
    """
    pose = bundle.get("robot_pose")
    if pose is None:
        return None
    xy = pose.get("xy")
    yaw_rad = pose.get("yaw_rad", 0.0)
    return {
        "world_xy": list(xy) if xy is not None else None,
        "yaw_deg": math.degrees(float(yaw_rad)),
    }


def error_dict(
    *,
    error_kind: str,
    message: str,
    bundle: dict,
    target_label: Optional[str] = None,
    target_xy: Optional[Sequence[float]] = None,
    blocker_label: Optional[str] = None,
    blocker_xy: Optional[Sequence[float]] = None,
    last_progress_xy: Optional[Sequence[float]] = None,
    suggested_recovery: Optional[str] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Build a D4-shaped error dict.

    All XYs are passed in as plain tuples / lists and rendered as
    JSON-serializable lists so the agent observation can echo the
    dict without further conversion.
    """
    out: Dict[str, Any] = {
        "status": "error",
        "error_kind": error_kind,
        "message": message,
        "target": {
            "label": target_label,
            "world_xy": list(target_xy) if target_xy is not None else None,
        },
        "robot_pose": _robot_pose_field(bundle),
        "blocker": (
            {
                "label": blocker_label,
                "world_xy": list(blocker_xy) if blocker_xy is not None else None,
            }
            if blocker_label is not None or blocker_xy is not None
            else None
        ),
        "last_progress_xy": (
            list(last_progress_xy) if last_progress_xy is not None else None
        ),
        "suggested_recovery": suggested_recovery,
    }
    out.update(extra)
    return out


def ok_dict(*, value: Any = None, **extra: Any) -> Dict[str, Any]:
    """Build the common ``status: ok`` shape used by read-only skills."""
    out: Dict[str, Any] = {"status": "ok", "value": value}
    out.update(extra)
    return out


def queued_dict(*, kind: str, **extra: Any) -> Dict[str, Any]:
    """Build the common ``status: queued`` shape used by skills that
    push a task onto ``bundle["task_queue"]``."""
    out: Dict[str, Any] = {"status": "queued", "kind": kind}
    out.update(extra)
    return out


# Helper used by both ``goto`` and ``find`` to surface "nearest seen
# labels" when the LLM names something not in the scene graph.
def nearest_seen_labels(bundle: dict, *, limit: int = 5) -> List[str]:
    nodes = bundle.get("scene_nodes") or []
    if not nodes:
        return []
    # Order by recency (most recent first); fall back to insertion
    # order. The runner is responsible for keeping ``scene_nodes``
    # well-formed (label + last_seen at minimum).
    sorted_nodes = sorted(
        nodes,
        key=lambda n: float(n.get("last_seen", 0.0)),
        reverse=True,
    )
    seen: List[str] = []
    for node in sorted_nodes:
        label = node.get("label")
        if isinstance(label, str) and label not in seen:
            seen.append(label)
        if len(seen) >= limit:
            break
    return seen


__all__ = [
    "error_dict",
    "nearest_seen_labels",
    "ok_dict",
    "queued_dict",
]
