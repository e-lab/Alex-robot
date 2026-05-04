"""Tests for the USD → 2D occupancy grid rasteriser (Phase 3.5a).

The module walks a `Usd.Stage`, reads the world-space AABB of every
`UsdGeomXformable` mesh / cube / sphere prim, filters by Z-band and
skip-list, and rasterises the surviving XY footprints onto a binary grid.
This is the **ground-truth map source** for the deliberative planner: in
the sim, the room USD already encodes every wall/counter/chair pose, so
we don't need SAM3 or a scan run to learn the geometry.

All tests build their stage in-memory with `Sdf.Layer.CreateAnonymous()`
and `UsdGeom.Cube.Define(...)`; no Isaac, no on-disk USD files. Pure pxr
+ numpy, fast.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

# pxr lives in the venv (Isaac Lab pulls it in). Skip the whole test file
# if the user's environment is bare-Python so pytest collection still works.
pxr = pytest.importorskip("pxr")
from pxr import Gf, Sdf, Usd, UsdGeom

from autonomy.usd_occupancy import (
    GridFrame,
    load_occupancy_npz,
    occupancy_from_usd,
    save_occupancy_npz,
    save_topdown_png,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────
def _make_stage() -> "Usd.Stage":
    """Anonymous USD stage with Z-up + meters set up the way Isaac uses it."""
    layer = Sdf.Layer.CreateAnonymous(".usda")
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    # Define a /World root to mirror Isaac scenes.
    UsdGeom.Xform.Define(stage, "/World")
    return stage


def _add_axis_aligned_cube(
    stage: "Usd.Stage",
    path: str,
    *,
    xy_min: tuple[float, float],
    xy_max: tuple[float, float],
    z_min: float,
    z_max: float,
    apply_collision_api: bool = False,
) -> "UsdGeom.Cube":
    """Add a unit cube prim and translate/scale it so its world AABB matches
    the requested rectangle.

    UsdGeom.Cube has its own intrinsic extent (size 2 by default, centred
    at origin). We scale it so the local `[-1, +1]` extent maps to the
    requested half-widths, then translate the centre.

    ``apply_collision_api`` flips on the ``UsdPhysics.CollisionAPI``
    schema, which is required for the production code path
    (``use_collision_api=True``) to pick the prim up.
    """
    cube = UsdGeom.Cube.Define(stage, path)
    sx = 0.5 * (xy_max[0] - xy_min[0])
    sy = 0.5 * (xy_max[1] - xy_min[1])
    sz = 0.5 * (z_max - z_min)
    cx = 0.5 * (xy_max[0] + xy_min[0])
    cy = 0.5 * (xy_max[1] + xy_min[1])
    cz = 0.5 * (z_max + z_min)
    xform = UsdGeom.Xformable(cube)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(cx, cy, cz))
    xform.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
    if apply_collision_api:
        from pxr import UsdPhysics  # type: ignore[import-not-found]
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


# ── GridFrame ────────────────────────────────────────────────────────────────
def test_grid_frame_world_to_grid_round_trip():
    """grid_to_world(world_to_grid(p)) ≈ p (within one cell)."""
    gf = GridFrame(origin_x=-1.0, origin_y=-2.0, resolution_m=0.05,
                   width=80, height=120)
    px, py = 0.30, -1.55
    ix, iy = gf.world_to_grid(px, py)
    wx, wy = gf.grid_to_world(ix, iy)
    assert abs(wx - px) <= gf.resolution_m
    assert abs(wy - py) <= gf.resolution_m


def test_grid_frame_in_bounds():
    gf = GridFrame(origin_x=0.0, origin_y=0.0, resolution_m=0.1,
                   width=10, height=20)
    assert gf.in_bounds(0, 0)
    assert gf.in_bounds(9, 19)
    assert not gf.in_bounds(-1, 0)
    assert not gf.in_bounds(0, -1)
    assert not gf.in_bounds(10, 0)
    assert not gf.in_bounds(0, 20)


def test_grid_frame_origin_is_lower_left():
    """Cell (0,0) corresponds to (origin_x, origin_y)."""
    gf = GridFrame(origin_x=2.5, origin_y=-1.0, resolution_m=0.1,
                   width=5, height=5)
    wx, wy = gf.grid_to_world(0, 0)
    # Cell centre convention: grid_to_world(ix, iy) = origin + (ix+0.5,iy+0.5)*res
    assert wx == pytest.approx(2.55, abs=1e-6)
    assert wy == pytest.approx(-0.95, abs=1e-6)


# ── occupancy_from_usd ───────────────────────────────────────────────────────
def test_empty_stage_produces_all_clear_grid():
    """Stage with no meshes → grid is all False (free)."""
    stage = _make_stage()
    occ, gf = occupancy_from_usd(
        stage, use_collision_api=False, bounds_xy=(-1.0, -1.0, 1.0, 1.0), resolution_m=0.1,
    )
    assert occ.shape == (gf.height, gf.width) == (20, 20)
    assert occ.dtype == bool
    assert not occ.any()


def test_cube_in_z_band_is_rasterized():
    """Cube spanning [0, 1] x [0, 1] x [0.3, 0.6] is fully inside the
    default Z-band [0.10, 1.50] → corresponding cells are True."""
    stage = _make_stage()
    _add_axis_aligned_cube(
        stage, "/World/Box1",
        xy_min=(0.0, 0.0), xy_max=(1.0, 1.0),
        z_min=0.3, z_max=0.6,
    )
    occ, gf = occupancy_from_usd(
        stage, use_collision_api=False, bounds_xy=(-1.0, -1.0, 2.0, 2.0), resolution_m=0.1,
    )
    # Cells at the centre of the box must be occupied
    ix, iy = gf.world_to_grid(0.5, 0.5)
    assert occ[iy, ix]
    # A cell well outside must be free
    ix2, iy2 = gf.world_to_grid(-0.5, -0.5)
    assert not occ[iy2, ix2]


def test_cube_above_z_band_is_skipped():
    """Cube whose Z range is entirely above z_band[1] (e.g. ceiling fixture)
    must NOT appear in the occupancy grid."""
    stage = _make_stage()
    _add_axis_aligned_cube(
        stage, "/World/Ceiling",
        xy_min=(-2.0, -2.0), xy_max=(2.0, 2.0),
        z_min=2.5, z_max=2.7,    # 2.5–2.7 m — well above z_band[1]=1.5
    )
    occ, gf = occupancy_from_usd(
        stage, use_collision_api=False,
        bounds_xy=(-3.0, -3.0, 3.0, 3.0),
        resolution_m=0.1,
        z_band=(0.10, 1.50),
    )
    assert not occ.any()


def test_cube_below_z_band_is_skipped():
    """Floor decoration entirely below z_band[0] must not block the planner."""
    stage = _make_stage()
    _add_axis_aligned_cube(
        stage, "/World/Rug",
        xy_min=(0.0, 0.0), xy_max=(1.0, 1.0),
        z_min=0.0, z_max=0.05,   # 0–5 cm — below z_band[0]=0.10
    )
    occ, gf = occupancy_from_usd(
        stage, use_collision_api=False,
        bounds_xy=(-1.0, -1.0, 2.0, 2.0),
        resolution_m=0.1,
        z_band=(0.10, 1.50),
    )
    assert not occ.any()


def test_cube_partially_in_z_band_is_rasterized():
    """A wall spanning floor-to-ceiling overlaps the z_band → must be rasterised."""
    stage = _make_stage()
    _add_axis_aligned_cube(
        stage, "/World/Wall",
        xy_min=(0.0, -2.0), xy_max=(0.1, 2.0),  # thin wall along Y
        z_min=0.0, z_max=3.0,
    )
    occ, gf = occupancy_from_usd(
        stage, use_collision_api=False,
        bounds_xy=(-1.0, -3.0, 1.0, 3.0),
        resolution_m=0.1,
    )
    # A cell on the wall (at x≈0.05, y=0) should be occupied
    ix, iy = gf.world_to_grid(0.05, 0.0)
    assert occ[iy, ix]


def test_two_cubes_both_appear():
    stage = _make_stage()
    _add_axis_aligned_cube(stage, "/World/A",
                           xy_min=(-1.0, 0.0), xy_max=(-0.5, 0.5),
                           z_min=0.3, z_max=0.7)
    _add_axis_aligned_cube(stage, "/World/B",
                           xy_min=( 0.5, 0.0), xy_max=( 1.0, 0.5),
                           z_min=0.3, z_max=0.7)
    occ, gf = occupancy_from_usd(
        stage, use_collision_api=False, bounds_xy=(-2.0, -1.0, 2.0, 1.0), resolution_m=0.1,
    )
    ixA, iyA = gf.world_to_grid(-0.75, 0.25)
    ixB, iyB = gf.world_to_grid( 0.75, 0.25)
    assert occ[iyA, ixA]
    assert occ[iyB, ixB]
    # And the gap between them is free
    ix_gap, iy_gap = gf.world_to_grid(0.0, 0.25)
    assert not occ[iy_gap, ix_gap]


def test_skip_prim_paths_excludes_cube():
    """Prims listed in skip_prim_paths must not contribute to the grid.

    Use case: the room USD contains the robot's spawn marker as a prim;
    we don't want it rasterised as an obstacle.
    """
    stage = _make_stage()
    _add_axis_aligned_cube(stage, "/World/RealObstacle",
                           xy_min=(0.0, 0.0), xy_max=(1.0, 1.0),
                           z_min=0.3, z_max=0.7)
    _add_axis_aligned_cube(stage, "/World/IgnoreMe",
                           xy_min=(2.0, 0.0), xy_max=(3.0, 1.0),
                           z_min=0.3, z_max=0.7)
    occ, gf = occupancy_from_usd(
        stage, use_collision_api=False,
        bounds_xy=(-1.0, -1.0, 4.0, 2.0),
        resolution_m=0.1,
        skip_prim_paths=["/World/IgnoreMe"],
    )
    # Real obstacle should appear
    ix, iy = gf.world_to_grid(0.5, 0.5)
    assert occ[iy, ix]
    # IgnoreMe should NOT
    ix2, iy2 = gf.world_to_grid(2.5, 0.5)
    assert not occ[iy2, ix2]


def test_skip_prim_paths_supports_subtree():
    """A skip path applies to all descendants too. Useful for skipping
    `/World/Alex/...` so the robot's own meshes don't appear as obstacles
    when the rasteriser runs on the live Isaac stage.
    """
    stage = _make_stage()
    UsdGeom.Xform.Define(stage, "/World/Alex")
    _add_axis_aligned_cube(stage, "/World/Alex/HEAD",
                           xy_min=(0.0, 0.0), xy_max=(0.5, 0.5),
                           z_min=1.4, z_max=1.6)
    occ, gf = occupancy_from_usd(
        stage, use_collision_api=False,
        bounds_xy=(-1.0, -1.0, 2.0, 2.0),
        resolution_m=0.1,
        skip_prim_paths=["/World/Alex"],
    )
    assert not occ.any()


def test_auto_bounds_when_bounds_xy_is_none():
    """If bounds_xy is None, the grid should auto-fit the union of all
    in-band AABBs (with a small padding)."""
    stage = _make_stage()
    _add_axis_aligned_cube(stage, "/World/Box",
                           xy_min=(1.0, 2.0), xy_max=(3.0, 4.0),
                           z_min=0.3, z_max=0.7)
    occ, gf = occupancy_from_usd(
        stage, use_collision_api=False, bounds_xy=None, resolution_m=0.1,
    )
    # Frame must contain the box and some padding around it
    assert gf.origin_x <= 1.0
    assert gf.origin_y <= 2.0
    assert gf.origin_x + gf.width  * gf.resolution_m >= 3.0
    assert gf.origin_y + gf.height * gf.resolution_m >= 4.0
    # Box centre is occupied
    ix, iy = gf.world_to_grid(2.0, 3.0)
    assert occ[iy, ix]


def test_returned_grid_dtype_and_shape():
    stage = _make_stage()
    _add_axis_aligned_cube(stage, "/World/Box",
                           xy_min=(0.0, 0.0), xy_max=(1.0, 1.0),
                           z_min=0.3, z_max=0.7)
    occ, gf = occupancy_from_usd(
        stage, use_collision_api=False, bounds_xy=(-1.0, -1.0, 2.0, 2.0), resolution_m=0.05,
    )
    assert occ.dtype == bool
    # Width / height come from bounds and resolution
    assert gf.width  == 60   # 3.0m / 0.05m
    assert gf.height == 60
    assert occ.shape == (gf.height, gf.width)


# ── Persistence ──────────────────────────────────────────────────────────────
def test_save_load_npz_round_trip(tmp_path):
    """save_occupancy_npz → load_occupancy_npz returns identical data."""
    occ = np.zeros((20, 30), dtype=bool)
    occ[5:10, 8:12] = True
    gf = GridFrame(origin_x=-1.5, origin_y=-1.0, resolution_m=0.1,
                   width=30, height=20)
    p = tmp_path / "scene.occupancy.npz"
    save_occupancy_npz(str(p), occ, gf)
    occ2, gf2 = load_occupancy_npz(str(p))
    assert np.array_equal(occ, occ2)
    assert occ2.dtype == bool
    assert gf2 == gf


# ── CollisionAPI mode (the production default) ───────────────────────────────
def test_collision_api_default_filters_to_collision_prims():
    """With ``use_collision_api=True`` (the default), only prims that have
    UsdPhysics.CollisionAPI applied contribute to the grid.

    iThor scenes apply CollisionAPI to many small per-piece Cube prims,
    so every collision shape rasterises tightly. This test mirrors that
    pattern with a synthetic stage: one cube has CollisionAPI (should
    appear), one doesn't (should not).
    """
    pytest.importorskip("pxr.UsdPhysics")
    stage = _make_stage()
    _add_axis_aligned_cube(stage, "/World/Solid",
                           xy_min=(0.0, 0.0), xy_max=(1.0, 1.0),
                           z_min=0.3, z_max=0.7,
                           apply_collision_api=True)
    _add_axis_aligned_cube(stage, "/World/Decorative",
                           xy_min=(2.0, 0.0), xy_max=(3.0, 1.0),
                           z_min=0.3, z_max=0.7,
                           apply_collision_api=False)
    occ, gf = occupancy_from_usd(
        stage,
        bounds_xy=(-1.0, -1.0, 4.0, 2.0),
        resolution_m=0.1,
        # Default: use_collision_api=True
    )
    # Solid (with CollisionAPI) → occupied
    ix, iy = gf.world_to_grid(0.5, 0.5)
    assert occ[iy, ix]
    # Decorative (no CollisionAPI) → free
    ix2, iy2 = gf.world_to_grid(2.5, 0.5)
    assert not occ[iy2, ix2]


def test_collision_api_skip_paths_still_apply():
    """skip_prim_paths must work even when filtering by CollisionAPI."""
    pytest.importorskip("pxr.UsdPhysics")
    stage = _make_stage()
    _add_axis_aligned_cube(stage, "/World/Keep",
                           xy_min=(0.0, 0.0), xy_max=(1.0, 1.0),
                           z_min=0.3, z_max=0.7,
                           apply_collision_api=True)
    _add_axis_aligned_cube(stage, "/World/Drop",
                           xy_min=(2.0, 0.0), xy_max=(3.0, 1.0),
                           z_min=0.3, z_max=0.7,
                           apply_collision_api=True)
    occ, gf = occupancy_from_usd(
        stage,
        bounds_xy=(-1.0, -1.0, 4.0, 2.0),
        resolution_m=0.1,
        skip_prim_paths=["/World/Drop"],
    )
    ix, iy = gf.world_to_grid(0.5, 0.5)
    assert occ[iy, ix]
    ix2, iy2 = gf.world_to_grid(2.5, 0.5)
    assert not occ[iy2, ix2]


def test_collision_api_z_band_filter_still_applies():
    """A collision cube above the z_band must still be filtered out."""
    pytest.importorskip("pxr.UsdPhysics")
    stage = _make_stage()
    _add_axis_aligned_cube(stage, "/World/Ceiling",
                           xy_min=(-1.0, -1.0), xy_max=(1.0, 1.0),
                           z_min=2.5, z_max=2.7,   # above z_band[1]=1.5
                           apply_collision_api=True)
    occ, gf = occupancy_from_usd(
        stage,
        bounds_xy=(-2.0, -2.0, 2.0, 2.0),
        resolution_m=0.1,
        z_band=(0.10, 1.50),
    )
    assert not occ.any()


def test_save_topdown_png_writes_correct_size(tmp_path):
    """PNG sidecar is the same shape as the grid (rows flipped)."""
    pil = pytest.importorskip("PIL")
    from PIL import Image
    occ = np.zeros((20, 30), dtype=bool)
    occ[0, 0]   = True   # lower-left in world coords → top-left would be wrong
    occ[-1, -1] = True
    gf = GridFrame(origin_x=0.0, origin_y=0.0, resolution_m=0.1,
                   width=30, height=20)
    p = tmp_path / "scene.topdown.png"
    save_topdown_png(str(p), occ, gf)
    assert p.exists()
    img = np.array(Image.open(p).convert("L"))
    assert img.shape == (20, 30)
    # Rows flipped: world (0,0) → bottom-left → image bottom-left
    # which numpy-rows-from-top means row index = height-1, col = 0.
    assert img[19, 0]  == 0    # occupied → black
    assert img[0,  29] == 0    # mirror corner
    # Free cells stay white
    assert img[10, 10] == 255
