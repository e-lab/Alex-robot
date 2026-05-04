"""USD → 2D occupancy grid (Phase 3.5a).

Walks a `Usd.Stage`, reads the world-space AABB of every drawable prim,
filters by Z-band and skip-list, and rasterises each surviving XY footprint
onto a binary grid. The output (`occ_grid`, `GridFrame`) is what the A*
planner consumes — Phase 3.5b.

Why USD-based: the room is a known sim asset. Its USD already encodes
every wall, counter, and chair pose to centimetre precision. We don't
need SAM3 or a scan run to learn the geometry; we just read it. SAM3
stays in the loop as the **goal lookup** ("where's the stove?"), not the
steering source.

Design notes:
- We use ``UsdGeom.BBoxCache`` for world-space AABBs. It handles xform
  composition, instancing, and inherited transforms — much safer than
  reading individual ``xformOpOrder`` entries.
- ``includedPurposes=[default]`` matches what the renderer draws. We
  skip ``proxy``, ``guide``, and ``render`` purposes by default; visual
  cones / debug markers don't become obstacles.
- Skip paths apply to whole subtrees: ``skip_prim_paths=["/World/Alex"]``
  excludes the robot's body even on a live Isaac stage.
- Grid origin is the **lower-left corner** of cell (0, 0). ``occ[iy, ix]``
  with ``iy`` indexing rows (Y axis) and ``ix`` indexing columns (X axis).
- No Isaac dependency. Pure ``pxr`` + ``numpy``. Tests build stages in-
  memory via ``Sdf.Layer.CreateAnonymous()``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

# pxr is provided by Isaac Lab's bundled pxr. It's also pip-installable
# (``usd-core``) for headless tests. We import lazily-friendly so that a
# pure-numpy import of this module's dataclass still works without pxr.
try:
    from pxr import Gf, Usd, UsdGeom  # type: ignore[import-not-found]
    _PXR_AVAILABLE = True
except ImportError:                   # pragma: no cover - tests skip without pxr
    _PXR_AVAILABLE = False

# UsdPhysics is optional — synthetic test stages (in-memory anonymous
# layers) typically don't apply CollisionAPI to their cubes, so the
# rasteriser falls back to visual-Gprim mode when this import fails or
# when the caller passes ``use_collision_api=False``.
try:
    from pxr import UsdPhysics  # type: ignore[import-not-found]
    _USDPHYSICS_AVAILABLE = True
except ImportError:                   # pragma: no cover
    _USDPHYSICS_AVAILABLE = False


# ── GridFrame ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class GridFrame:
    """Metadata that pins a 2D grid to world coordinates.

    Cell ``(ix, iy)`` covers the world rectangle::

        x ∈ [origin_x + ix*res,     origin_x + (ix+1)*res)
        y ∈ [origin_y + iy*res,     origin_y + (iy+1)*res)

    so cell (0, 0) is the lower-left of the grid. Resolution is uniform
    in X and Y. ``world_to_grid`` returns the integer indices of the cell
    containing a point; ``grid_to_world`` returns the cell **centre** —
    the convention used by waypoint extraction in the A* planner.
    """
    origin_x: float
    origin_y: float
    resolution_m: float
    width: int     # cells along X
    height: int    # cells along Y

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        ix = int(np.floor((x - self.origin_x) / self.resolution_m))
        iy = int(np.floor((y - self.origin_y) / self.resolution_m))
        return ix, iy

    def grid_to_world(self, ix: int, iy: int) -> Tuple[float, float]:
        # Centre of cell.
        wx = self.origin_x + (ix + 0.5) * self.resolution_m
        wy = self.origin_y + (iy + 0.5) * self.resolution_m
        return wx, wy

    def in_bounds(self, ix: int, iy: int) -> bool:
        return 0 <= ix < self.width and 0 <= iy < self.height


# ── Stage walking ────────────────────────────────────────────────────────────
def _is_under_any(prim_path: str, prefixes: Sequence[str]) -> bool:
    """True if ``prim_path`` equals or is a descendant of any prefix."""
    for pre in prefixes:
        if prim_path == pre or prim_path.startswith(pre + "/"):
            return True
    return False


def _iter_drawable_aabbs(
    stage: "Usd.Stage",
    *,
    skip_prim_paths: Sequence[str],
    use_collision_api: bool,
) -> Iterable[Tuple[str, "Gf.Range3d"]]:
    """Yield ``(prim_path, world_aabb)`` for every prim that should
    contribute to the occupancy grid.

    Two modes:

    * ``use_collision_api=True`` (default for real Isaac scenes) — only
      include prims with ``UsdPhysics.CollisionAPI``. iThor/AI2-THOR USD
      scenes decompose every furniture mesh into many small ``Cube`` /
      ``Capsule`` collision primitives (purpose=guide), so AABB
      rasterisation faithfully reproduces L-shaped counters and avoids
      the "hollow room shell" trap of visual meshes whose bounding box
      covers the entire interior.

    * ``use_collision_api=False`` — fall back to all visual ``Gprim``
      prims (purpose=default). Used by unit tests that build synthetic
      stages without applying the physics schema to their cubes.

    The BBoxCache's ``includedPurposes`` is set per-mode: collision
    geometry on iThor scenes is authored as purpose=guide, so we have to
    include guide explicitly when reading collision shapes. For visual-
    Gprim mode we keep the original purpose=default.
    """
    if use_collision_api:
        if not _USDPHYSICS_AVAILABLE:
            raise RuntimeError(
                "occupancy_from_usd(use_collision_api=True) requires "
                "UsdPhysics — pass use_collision_api=False or install "
                "the appropriate pxr build."
            )
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            includedPurposes=[
                UsdGeom.Tokens.default_,
                UsdGeom.Tokens.proxy,
                UsdGeom.Tokens.guide,
            ],
            useExtentsHint=True,
        )
    else:
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            includedPurposes=[UsdGeom.Tokens.default_],
            useExtentsHint=True,
        )

    for prim in stage.Traverse():
        path_str = str(prim.GetPath())
        if _is_under_any(path_str, skip_prim_paths):
            continue
        if use_collision_api:
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
        else:
            # Only **geometric** prims contribute. Xform / Scope / Camera
            # / Light don't carry geometry, and ``ComputeWorldBound`` on
            # a parent Xform would union its children.
            if not prim.IsA(UsdGeom.Gprim):
                continue
        try:
            bbox = bbox_cache.ComputeWorldBound(prim)
        except Exception:
            continue
        rng = bbox.ComputeAlignedRange()
        if rng.IsEmpty():
            continue
        yield path_str, rng


# ── Rasteriser ───────────────────────────────────────────────────────────────
def _rasterize_aabb(
    occ: np.ndarray,
    gf: GridFrame,
    *,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> None:
    """Mark all cells whose world rectangle intersects [xmin,xmax]x[ymin,ymax]."""
    ix_lo = max(0, int(np.floor((xmin - gf.origin_x) / gf.resolution_m)))
    iy_lo = max(0, int(np.floor((ymin - gf.origin_y) / gf.resolution_m)))
    ix_hi = min(gf.width  - 1, int(np.floor((xmax - gf.origin_x) / gf.resolution_m)))
    iy_hi = min(gf.height - 1, int(np.floor((ymax - gf.origin_y) / gf.resolution_m)))
    if ix_lo > ix_hi or iy_lo > iy_hi:
        return
    occ[iy_lo : iy_hi + 1, ix_lo : ix_hi + 1] = True


def _auto_bounds(
    aabbs: List[Tuple[float, float, float, float]],
    *,
    pad_m: float,
) -> Tuple[float, float, float, float]:
    if not aabbs:
        # Degenerate empty scene: produce a tiny 1x1m grid centred at origin
        return (-0.5, -0.5, 0.5, 0.5)
    xmins = [a[0] for a in aabbs]
    ymins = [a[1] for a in aabbs]
    xmaxs = [a[2] for a in aabbs]
    ymaxs = [a[3] for a in aabbs]
    return (min(xmins) - pad_m, min(ymins) - pad_m,
            max(xmaxs) + pad_m, max(ymaxs) + pad_m)


def occupancy_from_usd(
    stage: "Usd.Stage",
    *,
    z_band: Tuple[float, float] = (0.10, 1.50),
    resolution_m: float = 0.05,
    bounds_xy: Optional[Tuple[float, float, float, float]] = None,
    skip_prim_paths: Sequence[str] = (),
    auto_pad_m: float = 0.5,
    use_collision_api: bool = True,
) -> Tuple[np.ndarray, GridFrame]:
    """Rasterise USD geometry into a 2D occupancy grid.

    Parameters
    ----------
    stage
        Open USD stage (live Isaac stage or anonymous test stage — no
        difference to this function).
    z_band
        ``(z_lo, z_hi)`` — only AABBs whose Z range overlaps this band
        contribute to the grid. Default ``(0.10, 1.50)`` covers floor-
        clear-but-below-ceiling: drops rugs and ceiling lights, keeps
        counters / tables / chairs / walls.
    resolution_m
        Cell size in metres. Default 0.05 (5 cm) matches the scenegraph
        plan §3.5 convention.
    bounds_xy
        ``(xmin, ymin, xmax, ymax)`` of the grid in world coordinates.
        ``None`` → auto-fit the union of in-band AABBs plus ``auto_pad_m``.
    skip_prim_paths
        Sequence of USD prim paths whose subtree is excluded from the
        grid. Useful for ignoring the robot's own body
        (``["/World/Alex"]``) when rasterising a live Isaac stage.
    auto_pad_m
        Padding added on each side when ``bounds_xy is None``.
    use_collision_api
        ``True`` (default) — only include prims with
        ``UsdPhysics.CollisionAPI`` applied. iThor/AI2-THOR scenes
        decompose every furniture mesh into many small Cube/Capsule
        collision primitives, so AABB rasterisation correctly handles
        L-shaped counters and avoids the "hollow room shell" failure
        mode of visual meshes (whose bbox covers the entire interior).
        ``False`` — fall back to every visual ``Gprim``. Use this for
        synthetic test stages that don't apply the physics schema.

    Returns
    -------
    occ_grid
        Boolean array of shape ``(height, width)``; ``True`` = obstacle.
    grid_frame
        Metadata pinning the grid to world coordinates.
    """
    if not _PXR_AVAILABLE:
        raise RuntimeError(
            "occupancy_from_usd requires pxr (USD) — install usd-core or "
            "run from inside Isaac Lab's Python env."
        )

    # First pass: gather candidate AABBs for the auto-bounds calculation.
    candidates: List[Tuple[float, float, float, float]] = []  # (xmin,ymin,xmax,ymax)
    for path_str, rng in _iter_drawable_aabbs(
        stage,
        skip_prim_paths=tuple(skip_prim_paths),
        use_collision_api=use_collision_api,
    ):
        z_lo = rng.GetMin()[2]
        z_hi = rng.GetMax()[2]
        # Z-band overlap test — keep prims whose Z range intersects the band.
        if z_hi < z_band[0] or z_lo > z_band[1]:
            continue
        x_lo = rng.GetMin()[0]
        y_lo = rng.GetMin()[1]
        x_hi = rng.GetMax()[0]
        y_hi = rng.GetMax()[1]
        candidates.append((x_lo, y_lo, x_hi, y_hi))

    # Resolve grid frame.
    if bounds_xy is None:
        xmin, ymin, xmax, ymax = _auto_bounds(candidates, pad_m=auto_pad_m)
    else:
        xmin, ymin, xmax, ymax = bounds_xy

    width  = max(1, int(np.ceil((xmax - xmin) / resolution_m)))
    height = max(1, int(np.ceil((ymax - ymin) / resolution_m)))
    gf = GridFrame(
        origin_x=float(xmin),
        origin_y=float(ymin),
        resolution_m=float(resolution_m),
        width=width,
        height=height,
    )

    # Second pass: rasterise.
    occ = np.zeros((height, width), dtype=bool)
    for x_lo, y_lo, x_hi, y_hi in candidates:
        _rasterize_aabb(occ, gf, xmin=x_lo, ymin=y_lo, xmax=x_hi, ymax=y_hi)
    return occ, gf


# ── Persistence ──────────────────────────────────────────────────────────────
def save_occupancy_npz(path: str, occ: np.ndarray, gf: GridFrame) -> None:
    """Save grid + frame to a single NPZ file. Compatible round-trip with
    ``load_occupancy_npz``.
    """
    np.savez_compressed(
        path,
        occ=occ.astype(bool),
        origin_x=np.float64(gf.origin_x),
        origin_y=np.float64(gf.origin_y),
        resolution_m=np.float64(gf.resolution_m),
        width=np.int64(gf.width),
        height=np.int64(gf.height),
    )


def load_occupancy_npz(path: str) -> Tuple[np.ndarray, GridFrame]:
    """Inverse of ``save_occupancy_npz``."""
    d = np.load(path)
    gf = GridFrame(
        origin_x=float(d["origin_x"]),
        origin_y=float(d["origin_y"]),
        resolution_m=float(d["resolution_m"]),
        width=int(d["width"]),
        height=int(d["height"]),
    )
    return d["occ"].astype(bool), gf


def save_topdown_png(path: str, occ: np.ndarray, gf: GridFrame) -> None:
    """Render the occupancy grid as a PNG sidecar for human inspection.

    Free cells = white, occupied cells = black. Image rows are flipped so
    +Y points up in the saved image (standard top-down convention).
    No matplotlib dependency — uses PIL only.
    """
    try:
        from PIL import Image
    except ImportError as exc:                          # pragma: no cover
        raise RuntimeError("save_topdown_png requires Pillow") from exc

    # 0 = occupied (black), 255 = free (white). Flip rows so +Y is up.
    img = np.where(occ, 0, 255).astype(np.uint8)
    img = np.flipud(img)
    Image.fromarray(img, mode="L").save(path)


__all__ = [
    "GridFrame",
    "occupancy_from_usd",
    "save_occupancy_npz",
    "load_occupancy_npz",
    "save_topdown_png",
]
