"""Forward-cone obstacle distance from a depth image.

Used by the Phase 3.5 emergency brake — a last-resort safety check that
zeros ``_cmd`` when something very close (< 0.5 m) appears in the forward
cone. The deliberative planner (USD-derived occupancy + A*) is the primary
path around obstacles; this is purely "stop if the planner missed it."

Why median-of-lowest-quantile rather than min: a single near-pixel can come
from a SAM3 mask edge, sensor noise, or a depth-spike on a glossy surface,
and would falsely trip the brake. Taking the median of the lowest 20%
gives us "the typical near surface inside the cone" — robust to a few
spurious pixels but still dominated by close obstacles whenever a real one
fills more than 20 % of the cone.

Pure numpy. No Isaac, no torch — fully unit-testable.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


def _zone_distance(
    cone: np.ndarray,
    *,
    q_lowest: float,
    max_depth: float,
) -> float:
    """Median of the lowest ``q_lowest`` fraction of a depth slice. Returns
    ``+inf`` for an empty / all-invalid slice.
    """
    valid = np.isfinite(cone) & (cone > 0.05) & (cone < max_depth)
    if not np.any(valid):
        return float("inf")
    z = cone[valid]
    q = max(0.0, min(1.0, float(q_lowest)))
    if q == 0.0 or z.size == 1:
        return float(z.min())
    threshold = float(np.quantile(z, q))
    nearest = z[z <= threshold]
    if nearest.size == 0:
        return float(z.min())
    return float(np.median(nearest))


def _cone_bounds(
    depth: np.ndarray,
    K: np.ndarray,
    *,
    h_deg: float,
    v_deg: float,
) -> "tuple[int, int, int, int] | None":
    """Pixel-space bounds ``(u_lo, u_hi, v_lo, v_hi)`` of the cone.

    Returns ``None`` when the cone collapses to zero width (degenerate).
    """
    if depth.ndim == 3:
        depth = depth[..., 0]
    H, W = depth.shape
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    half_w = fx * math.tan(math.radians(h_deg))
    half_h = fy * math.tan(math.radians(v_deg))
    u_lo = max(0,     int(math.floor(cx - half_w)))
    u_hi = min(W - 1, int(math.ceil (cx + half_w)))
    v_lo = max(0,     int(math.floor(cy - half_h)))
    v_hi = min(H - 1, int(math.ceil (cy + half_h)))
    if u_lo >= u_hi or v_lo >= v_hi:
        return None
    return u_lo, u_hi, v_lo, v_hi


def forward_cone_distance(
    depth: np.ndarray,
    K: np.ndarray,
    *,
    h_deg: float = 20.0,
    v_deg: float = 10.0,
    q_lowest: float = 0.2,
    max_depth: float = 10.0,
) -> float:
    """Return the median of the lowest ``q_lowest`` fraction of depth values
    inside the central ±h_deg / ±v_deg cone.

    Parameters
    ----------
    depth
        ``(H, W)`` or ``(H, W, 1)`` float depth image, metres along the camera
        principal axis (Isaac's ``distance_to_image_plane``).
    K
        ``(3, 3)`` pinhole intrinsics. Only ``fx, fy, cx, cy`` are read.
    h_deg, v_deg
        Half-angles (deg) of the cone, horizontal and vertical respectively.
    q_lowest
        Quantile of nearest depths to keep before taking their median.
        ``0.2`` selects the closest 20 % of pixels in the cone — the part
        most likely to belong to an obstacle. Clamped to ``(0, 1]``.
    max_depth
        Pixels with depth ≥ this are treated as "no data" (clip-far / sky).

    Returns
    -------
    distance_m
        Cone obstacle distance in metres. ``+inf`` when the cone has no
        valid pixels (caller treats infinite distance as "clear").
    """
    if depth.ndim == 3:
        depth = depth[..., 0]
    bounds = _cone_bounds(depth, K, h_deg=h_deg, v_deg=v_deg)
    if bounds is None:
        return float("inf")
    u_lo, u_hi, v_lo, v_hi = bounds
    cone = depth[v_lo : v_hi + 1, u_lo : u_hi + 1]
    return _zone_distance(cone, q_lowest=q_lowest, max_depth=max_depth)


__all__ = ["forward_cone_distance"]
