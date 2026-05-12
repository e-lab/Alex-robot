"""Perception skills: find, peek, survey, list_scene, describe_view (D4 + D12).

* ``find(label)`` and ``list_scene()`` are read-only — they consult
  ``bundle["scene_nodes"]`` and return a snapshot.
* ``peek(direction)`` and ``survey(...)`` push head-cam motion tasks
  to the queue; the autonomy loop executes them and the always-on
  perception fold-in (D13) updates SAM3 + heightmap naturally.
* ``describe_view()`` is a stub in LA-1 — the real VLM client lands
  in LA-4. Returning a clear ``error[vlm_disabled]`` lets the agent
  see the skill exists but isn't wired, rather than the sandbox
  rejecting an unknown call.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from ._errors import error_dict, nearest_seen_labels, ok_dict, queued_dict


_PEEK_DIRECTIONS = {"left", "right", "up", "down", "forward"}

# Default sweep angles (matched to loco_x.conf.agent.anthropic.yaml).
_DEFAULT_SURVEY_ANGLES = [-60.0, -30.0, 0.0, 30.0, 60.0, 90.0]
_QUICK_SURVEY_ANGLES = [-45.0, 0.0, 45.0]


def make_perception_skills(bundle: dict) -> Dict[str, Callable[..., Dict[str, Any]]]:
    """Build the perception skill closures for one tick."""

    def find(label: str) -> Dict[str, Any]:
        """Return the most recent scene-graph node matching ``label``.

        Read-only — no task queued. On miss returns the D4
        ``error[unknown_label]`` shape with ``nearest_seen_labels``
        so the agent can re-prompt SAM3 with a related vocabulary
        item or call ``peek`` to look elsewhere.
        """
        nodes = bundle.get("scene_nodes") or []
        matches = [n for n in nodes if n.get("label") == label]
        if not matches:
            return error_dict(
                error_kind="unknown_label",
                message=f"'{label}' not seen by SAM3 in current scene graph",
                bundle=bundle,
                target_label=label,
                nearest_seen_labels=nearest_seen_labels(bundle),
                suggested_recovery="try_peek",
            )
        # Pick the most recently seen instance.
        node = max(matches, key=lambda n: float(n.get("last_seen", 0.0)))
        xy = node.get("world_xy")
        return ok_dict(value={
            "label": label,
            "world_xy": list(xy) if xy is not None else None,
            "last_seen": float(node.get("last_seen", 0.0)),
            "confidence": float(node.get("confidence", 0.0)),
        })

    def peek(direction: str) -> Dict[str, Any]:
        """Move the head ±30° in the requested direction.

        Per D13, perception updates run every autonomy tick regardless
        of the agent's state — peek just biases the head so the next
        few perception ticks see a different sector. The autonomy
        loop is responsible for the head joint commands.
        """
        if direction not in _PEEK_DIRECTIONS:
            return error_dict(
                error_kind="unknown_direction",
                message=(
                    f"'{direction}' is not a peek direction; "
                    f"valid: {sorted(_PEEK_DIRECTIONS)}"
                ),
                bundle=bundle,
                suggested_recovery=None,
            )
        bundle["task_queue"].append({"kind": "peek", "direction": direction})
        return queued_dict(kind="peek", direction=direction)

    def survey(
        angles_deg: Optional[List[float]] = None,
        quick: bool = False,
    ) -> Dict[str, Any]:
        """Multi-angle head sweep (D12). Three modes:

        * ``angles_deg=None, quick=False`` → 6-angle full sweep.
        * ``angles_deg=None, quick=True`` → 3-angle quick sweep
          (good for corridors).
        * ``angles_deg=[...]`` → agent-supplied list, overrides both.
        """
        if angles_deg is not None:
            angles = [float(a) for a in angles_deg]
        elif quick:
            angles = list(_QUICK_SURVEY_ANGLES)
        else:
            angles = list(_DEFAULT_SURVEY_ANGLES)
        bundle["task_queue"].append({
            "kind": "survey",
            "angles_deg": angles,
        })
        return queued_dict(kind="survey", angles_deg=angles)

    def list_scene() -> Dict[str, Any]:
        """Snapshot of the scene graph.

        Returns a serializable list of ``{label, world_xy, last_seen,
        confidence}`` dicts. The agent observation usually shows a
        *filtered* view (D2 spatial-temporal filter); ``list_scene()``
        is the escape hatch when the LLM wants the unfiltered list.
        """
        nodes = bundle.get("scene_nodes") or []
        value = []
        for n in nodes:
            xy = n.get("world_xy")
            value.append({
                "label": n.get("label"),
                "world_xy": list(xy) if xy is not None else None,
                "last_seen": float(n.get("last_seen", 0.0)),
                "confidence": float(n.get("confidence", 0.0)),
            })
        return ok_dict(value=value)

    def describe_view() -> Dict[str, Any]:
        """VLM caption of the current head-cam frame (D12).

        LA-1 stub: the VLM client lives in :mod:`loco_x.llm` and
        lands in LA-4. Returning ``error[vlm_disabled]`` keeps the
        skill in the namespace (so the sandbox doesn't reject) and
        signals to the LLM that the feature exists but isn't wired
        in this build.
        """
        return error_dict(
            error_kind="vlm_disabled",
            message="describe_view requires a VLM client; not wired in LA-1",
            bundle=bundle,
            suggested_recovery=None,
        )

    return {
        "find": find,
        "peek": peek,
        "survey": survey,
        "list_scene": list_scene,
        "describe_view": describe_view,
    }


__all__ = ["make_perception_skills"]
