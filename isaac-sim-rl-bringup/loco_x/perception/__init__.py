"""Loco-X perception layer — D2 observation builder + scene filter.

The agent runner calls :func:`build_observation` once per tick to
render the LLM's view of the world. The filter is exposed separately
because the runner also uses it for cross-referencing (e.g. matching
scene-graph labels against a planned path's worst-stale cell).
"""
from .observation import build_observation
from .scene_filter import FilterParams, FilteredScene, filter_scene_nodes

__all__ = [
    "FilterParams",
    "FilteredScene",
    "build_observation",
    "filter_scene_nodes",
]
