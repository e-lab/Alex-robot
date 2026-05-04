"""Build the room USD's 2D occupancy grid (Phase 3.5a integration test).

Standalone CLI: opens the FloorPlan1 USD, runs ``occupancy_from_usd``,
and saves ``room.occupancy.npz`` + ``room.topdown.png`` next to it.
Quick eyeball check before wiring the rasteriser into the live runtime.

Run::

    venv/bin/python isaac-sim-rl-bringup/scripts/alex_room_explore/tools/build_room_occupancy.py

No Isaac required — just pxr + numpy + PIL.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path

# Add the autonomy package to sys.path so we can import without installing.
_HERE = Path(__file__).resolve()
_AUTONOMY_PARENT = _HERE.parents[1]   # scripts/alex_room_explore/
if str(_AUTONOMY_PARENT) not in sys.path:
    sys.path.insert(0, str(_AUTONOMY_PARENT))

import numpy as np
from pxr import Usd, UsdGeom

from autonomy.usd_occupancy import (
    occupancy_from_usd,
    save_occupancy_npz,
    save_topdown_png,
)


DEFAULT_USD = (
    "assets/usd/scenes/ithor/FloorPlan1_physics/scene.usda"
)


def _find_repo_root(start: Path) -> Path:
    """Walk upward to find the Alex-robot repo root (has assets/usd/...)."""
    p = start
    while p != p.parent:
        if (p / "assets" / "usd").is_dir():
            return p
        p = p.parent
    raise RuntimeError(f"Could not find repo root from {start}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--usd", type=str,
        help=f"Path to USD scene (default: <repo>/{DEFAULT_USD})",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Where to write room.occupancy.npz / room.topdown.png. "
             "Default: alongside the USD.",
    )
    parser.add_argument(
        "--resolution", type=float, default=0.05,
        help="Grid cell size in metres. Default 0.05.",
    )
    parser.add_argument(
        "--z-min", type=float, default=0.10,
        help="Lower Z bound (drops floor). Default 0.10 m.",
    )
    parser.add_argument(
        "--z-max", type=float, default=1.50,
        help="Upper Z bound (drops ceiling). Default 1.50 m.",
    )
    parser.add_argument(
        "--skip", type=str, action="append", default=[],
        help="Prim path subtree to exclude. Repeatable.",
    )
    args = parser.parse_args()

    repo_root = _find_repo_root(_HERE)
    usd_path = Path(args.usd) if args.usd else (repo_root / DEFAULT_USD)
    if not usd_path.is_file():
        print(f"USD not found: {usd_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else usd_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "room.occupancy.npz"
    out_png = out_dir / "room.topdown.png"

    print(f"[build_room_occupancy] opening {usd_path}")
    # Some referenced assets in FloorPlan1 may be missing on disk; pxr
    # logs warnings to stderr but the resolved geometry still loads. We
    # silence stderr only during the open() to keep the integration log
    # readable. Stage warnings about missing prims don't affect the map.
    with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
        stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        print(f"[build_room_occupancy] failed to open stage", file=sys.stderr)
        return 1

    print(f"  up axis     = {UsdGeom.GetStageUpAxis(stage)}")
    print(f"  meters/unit = {UsdGeom.GetStageMetersPerUnit(stage)}")

    n_gprim = sum(1 for p in stage.Traverse() if p.IsA(UsdGeom.Gprim))
    print(f"  Gprim count = {n_gprim}")

    print(f"[build_room_occupancy] rasterising "
          f"(z_band=[{args.z_min}, {args.z_max}]m, res={args.resolution}m, "
          f"skip={args.skip})")
    occ, gf = occupancy_from_usd(
        stage,
        z_band=(args.z_min, args.z_max),
        resolution_m=args.resolution,
        bounds_xy=None,
        skip_prim_paths=args.skip,
    )

    n_occupied = int(occ.sum())
    n_total = int(occ.size)
    print(f"  grid:    {gf.width} x {gf.height} cells "
          f"= {gf.width * gf.resolution_m:.2f} x {gf.height * gf.resolution_m:.2f} m")
    print(f"  origin:  ({gf.origin_x:+.2f}, {gf.origin_y:+.2f})")
    print(f"  occupied {n_occupied} / {n_total} cells "
          f"({100.0 * n_occupied / n_total:.1f}%)")

    save_occupancy_npz(str(out_npz), occ, gf)
    print(f"[build_room_occupancy] wrote {out_npz}")
    save_topdown_png(str(out_png), occ, gf)
    print(f"[build_room_occupancy] wrote {out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
