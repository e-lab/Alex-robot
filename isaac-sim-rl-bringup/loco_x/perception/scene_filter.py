"""D2 spatial-temporal scene-graph filter.

Pure function. Given the full scene graph + the robot's pose + the
current time + the agent's active labels, return:

* the list of nodes that should appear in the observation, sorted by
  distance to the robot (closest first),
* the count of nodes hidden by the filter (so the LLM sees a
  ``(K nodes hidden — call list_scene() to see all)`` line in the
  observation).

A node is *kept* if **any** of:

1. its world XY is within ``radius_m`` of the robot (default 5 m),
2. it was seen within the last ``recency_s`` seconds (default 30 s),
3. its label matches an *active* one (the goal lock, a pending
   ``find(label)`` query) — never drop the thing the agent just
   asked about.

After inclusion, cap at ``max_nodes`` (default 20) by closest
distance. Hidden count records everything dropped.

The filter is deliberately pure: no bundle, no provider, no clock —
the caller injects ``now`` so tests stay deterministic (D8) and the
agent runner can reuse the function for cross-referencing (e.g.
matching scene-graph labels against a planned path's
``max_path_staleness`` worst-cell).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# D2 defaults — must match ``AgentCfg.obs_*`` in the schema.
@dataclass(frozen=True)
class FilterParams:
    """Knobs for :func:`filter_scene_nodes`. Defaults match D2."""

    radius_m: float = 5.0
    recency_s: float = 30.0
    max_nodes: int = 20


@dataclass(frozen=True)
class FilteredScene:
    """Output of :func:`filter_scene_nodes`. The ``kept`` list is
    sorted by distance ascending (closest first); ``hidden_count``
    feeds the agent observation's ``(K nodes hidden ...)`` line."""

    kept: List[Dict[str, Any]]
    hidden_count: int


def filter_scene_nodes(
    nodes: Sequence[Dict[str, Any]],
    *,
    robot_xy: Tuple[float, float],
    now: float,
    active_labels: Iterable[str] = (),
    params: Optional[FilterParams] = None,
) -> FilteredScene:
    """Apply the D2 inclusion / cap filter.

    The input ``nodes`` is expected to be a list of dicts with at
    least ``label`` and ``world_xy`` keys; ``last_seen`` is read with
    a 0.0 default (older-than-everything → relies on the recency
    branch missing). Missing ``world_xy`` is treated as "infinitely
    far" (still allowed in via the active-label rule, but never close
    enough to satisfy the radius rule).
    """
    if params is None:
        params = FilterParams()
    active_set = set(active_labels) if active_labels else set()
    rx, ry = robot_xy

    decorated: List[Tuple[float, Dict[str, Any], bool]] = []
    for node in nodes:
        xy = node.get("world_xy")
        if xy is None:
            d = math.inf
        else:
            d = math.hypot(xy[0] - rx, xy[1] - ry)
        last_seen = float(node.get("last_seen", 0.0))
        age = now - last_seen

        kept = (
            d <= params.radius_m
            or age <= params.recency_s
            or node.get("label") in active_set
        )
        decorated.append((d, node, kept))

    # Sort by distance so the kept slice stays "closest first" and the
    # hidden-count breakdown is stable under cap.
    decorated.sort(key=lambda t: t[0])

    included: List[Dict[str, Any]] = [n for _, n, kept in decorated if kept]
    hidden_before_cap = sum(1 for _, _, kept in decorated if not kept)

    if len(included) > params.max_nodes:
        kept_after_cap = included[: params.max_nodes]
        hidden_after_cap = len(included) - params.max_nodes
    else:
        kept_after_cap = included
        hidden_after_cap = 0

    return FilteredScene(
        kept=kept_after_cap,
        hidden_count=hidden_before_cap + hidden_after_cap,
    )


__all__ = ["FilterParams", "FilteredScene", "filter_scene_nodes"]
