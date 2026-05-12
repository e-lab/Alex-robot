"""Synthetic point-cloud / pose factories for LA-0b.1 testing.

These helpers fabricate :class:`PointCloud` + :class:`Pose` inputs that
the :class:`HeightMapProvider` can ingest without Isaac, without a
camera model, and without depth back-projection. They live in the
production package (not the test tree) so the LA-0b.2 integration
layer can reuse them when wiring elevation_mapping_cupy.

Design choice: the provider's ``update()`` accepts a **point cloud in
world coordinates** plus a robot pose, not a raw RGBD frame. The
back-projection step is handled by a separate, swappable helper (LA-0b.2
fills it in for the real head-cam path). That way the LA-0b.1 harness
can drive the *fold-in math* — max-height per cell, consistency gate,
decay — without needing a camera model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# ── Dataclasses ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Pose:
    """Minimal head-cam / body pose. World frame, meters / radians.

    The provider only reads ``xy`` for the drive-through stamp and
    ``yaw`` for the (future) FOV cone. Kept tiny so synthetic tests
    don't have to fabricate full SE(3).
    """

    xy: Tuple[float, float] = (0.0, 0.0)
    z: float = 0.0
    yaw_rad: float = 0.0


@dataclass
class PointCloud:
    """A set of 3D points in **world coordinates**, meters.

    ``points`` is an ``(N, 3)`` float array. Construct via the helpers
    below or directly from any depth-back-projection routine — the
    provider doesn't care how the points were produced.
    """

    points: np.ndarray  # (N, 3) float64
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(
                f"points must be (N, 3); got {self.points.shape}"
            )
        if self.points.dtype != np.float64:
            self.points = self.points.astype(np.float64, copy=False)


# ── Cloud factories ─────────────────────────────────────────────────────────
def flat_floor(
    *,
    x_range: Tuple[float, float] = (-5.0, 5.0),
    y_range: Tuple[float, float] = (-5.0, 5.0),
    n_points: int = 10_000,
    z: float = 0.0,
    seed: int = 0,
    timestamp: float = 0.0,
) -> PointCloud:
    """Random points uniformly sampled on a horizontal plane at ``z``.

    Default 10 k points on the 10x10 m floor — matches the canonical
    test from the plan. Deterministic via ``seed``.
    """
    rng = np.random.default_rng(seed)
    xs = rng.uniform(x_range[0], x_range[1], size=n_points)
    ys = rng.uniform(y_range[0], y_range[1], size=n_points)
    zs = np.full(n_points, z, dtype=np.float64)
    pts = np.stack([xs, ys, zs], axis=1)
    return PointCloud(points=pts, timestamp=timestamp)


def box(
    *,
    xy_min: Tuple[float, float],
    xy_max: Tuple[float, float],
    z_min: float,
    z_max: float,
    n_points: int = 500,
    seed: int = 0,
    timestamp: float = 0.0,
) -> PointCloud:
    """Random points filling an axis-aligned box.

    Used to stamp a "tall obstacle" or "low cable" onto a floor cloud
    via simple concatenation (see :func:`merge`). Points are uniform on
    the box surface in spirit — we use a volume sample because the
    max-height-per-cell aggregation only needs the *upper* surface to
    be present in the cloud, which a volume sample guarantees.
    """
    rng = np.random.default_rng(seed)
    xs = rng.uniform(xy_min[0], xy_max[0], size=n_points)
    ys = rng.uniform(xy_min[1], xy_max[1], size=n_points)
    zs = rng.uniform(z_min, z_max, size=n_points)
    pts = np.stack([xs, ys, zs], axis=1)
    return PointCloud(points=pts, timestamp=timestamp)


def add_gaussian_noise(
    cloud: PointCloud, *, sigma_m: float, seed: int = 0,
) -> PointCloud:
    """Add isotropic Gaussian noise to every point. Used for the
    consistency-gate test (5 noisy frames, σ=0.02 m)."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma_m, size=cloud.points.shape)
    return PointCloud(points=cloud.points + noise, timestamp=cloud.timestamp)


def merge(*clouds: PointCloud, timestamp: Optional[float] = None) -> PointCloud:
    """Concatenate clouds. Timestamp defaults to the first cloud's."""
    if not clouds:
        raise ValueError("merge requires at least one cloud")
    pts = np.concatenate([c.points for c in clouds], axis=0)
    t = clouds[0].timestamp if timestamp is None else timestamp
    return PointCloud(points=pts, timestamp=t)


def outlier_spike(
    *,
    xy: Tuple[float, float],
    z: float = 10.0,
    n_points: int = 3,
    timestamp: float = 0.0,
) -> PointCloud:
    """A handful of points well above the scene — the reflection-
    artifact case from the plan. With the consistency gate, a single
    cloud carrying these points must NOT flip the cell to OBSTACLE."""
    pts = np.array(
        [[xy[0], xy[1], z] for _ in range(n_points)], dtype=np.float64
    )
    return PointCloud(points=pts, timestamp=timestamp)


__all__ = [
    "Pose",
    "PointCloud",
    "flat_floor",
    "box",
    "add_gaussian_noise",
    "merge",
    "outlier_spike",
]
