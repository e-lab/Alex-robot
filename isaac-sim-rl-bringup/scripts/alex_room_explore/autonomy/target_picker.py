"""Pick the autonomy goal from a SceneGraph by target label.

Pure function: given an in-memory SceneGraph and a target label string,
returns the best ObjectNode that matches, or None if nothing yet qualifies.

"Best" = highest confidence among objects that meet:
  - case-insensitive label match (and `attrs.aliases` if present)
  - confidence >= ``lock_conf``
  - n_observations >= ``min_observations``  (Phase-2 stability gate)

Phase-2 lock-on rationale (from PLAN/autonomous_navigation_plan.md):
single-frame SAM3 detections can mis-project (mask leaks behind glass, depth
holes); requiring N stable sightings prevents the goal from latching onto a
ghost. The vendored ObjectNode tracks ``n_observations`` for us — we just
read it.
"""
from __future__ import annotations

from typing import Optional


def pick_goal_for_target(
    sg,                         # scene_graph.SceneGraph (avoid hard import here)
    target_label: str,
    *,
    lock_conf: float = 0.6,
    min_observations: int = 3,
):
    """Return the highest-confidence qualifying ObjectNode for ``target_label``.

    Search order: ``sg.objects`` (promoted) only — pending candidates are
    intentionally excluded since they haven't cleared the vendored package's
    own promotion threshold yet.
    """
    if not target_label:
        return None
    target_lower = target_label.lower()

    best = None
    best_conf = -1.0
    for obj in sg.objects.values():
        if obj.label.lower() != target_lower:
            # Optional alias support — vendored ObjectNode has a free-form
            # attrs dict that future labellers may populate with synonyms.
            aliases = obj.attrs.get("aliases", []) if hasattr(obj, "attrs") else []
            if isinstance(aliases, str):
                aliases = [aliases]
            if not any(a.lower() == target_lower for a in aliases):
                continue

        if obj.confidence < lock_conf:
            continue
        if obj.n_observations < min_observations:
            continue

        if obj.confidence > best_conf:
            best = obj
            best_conf = obj.confidence

    return best


__all__ = ["pick_goal_for_target"]
