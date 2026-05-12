"""Exploration skills: next_frontier, visited_fraction (D5).

Both delegate to :class:`OccupancyProvider`. The provider already
encapsulates the info-gain ranking + D14.1 semantic-anchor boost; the
skill is a thin adapter that:

* extracts the robot's pose from ``bundle["robot_pose"]`` for the
  ``from_xy`` argument,
* serializes the resulting :class:`FrontierCandidate` into a dict,
* surfaces ``error[no_frontiers]`` (D11 Case A signal) when the
  provider returns ``[]``.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from ._errors import error_dict, ok_dict


def make_exploration_skills(bundle: dict) -> Dict[str, Callable[..., Dict[str, Any]]]:
    """Build the exploration skill closures for one tick."""

    def next_frontier(
        prefer_near: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Return the top-ranked frontier candidate.

        Forwards ``prefer_near`` (D14.1 semantic anchors) to the
        provider. If the provider doesn't accept the kwarg (older
        impls), we fall back to the no-anchors call.
        """
        provider = bundle.get("occ_provider")
        if provider is None:
            return error_dict(
                error_kind="no_provider",
                message="occupancy provider not configured",
                bundle=bundle,
                suggested_recovery=None,
            )
        pose = bundle.get("robot_pose") or {}
        from_xy = pose.get("xy")
        try:
            cands = provider.frontier_cells(
                from_xy=from_xy,
                k=1,
                prefer_near=prefer_near,
            )
        except TypeError:
            # Older provider without prefer_near (USD stub).
            cands = provider.frontier_cells(from_xy=from_xy, k=1)
        if not cands:
            # D11 Case A — coverage signal. The agent observation
            # surfaces ``visited_fraction`` so the LLM can decide
            # whether to fail() or keep peeking from current pose.
            return error_dict(
                error_kind="no_frontiers",
                message="no reachable unknown regions remain",
                bundle=bundle,
                suggested_recovery="give_up",
                visited_fraction=float(provider.visited_fraction()),
            )
        c = cands[0]
        return ok_dict(value={
            "world_xy": list(c.world_xy),
            "info_gain": float(c.info_gain),
            "travel_distance": float(c.travel_distance),
            "score": float(c.score),
        })

    def visited_fraction() -> Dict[str, Any]:
        """Fraction of accessible cells the height map has observed.

        Used by the agent observation builder *and* by the LLM
        directly when deciding whether to call ``fail()``.
        """
        provider = bundle.get("occ_provider")
        if provider is None:
            return error_dict(
                error_kind="no_provider",
                message="occupancy provider not configured",
                bundle=bundle,
                suggested_recovery=None,
            )
        return ok_dict(value=float(provider.visited_fraction()))

    return {
        "next_frontier": next_frontier,
        "visited_fraction": visited_fraction,
    }


__all__ = ["make_exploration_skills"]
