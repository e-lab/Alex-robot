"""Plan a path through the room.occupancy.npz and render the polyline
overlaid on the topdown PNG (Phase 3.5b integration test).

Usage::

    venv/bin/python isaac-sim-rl-bringup/scripts/alex_room_explore/tools/plan_path_demo.py \\
        --map /tmp/phase35a_v3/room.occupancy.npz \\
        --start  1.47 -0.24 \\
        --goal  -0.55 -2.36 \\
        --out   /tmp/phase35a_v3/room.path.png

No Isaac, no pxr — just the planner module + Pillow.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make autonomy package importable.
_HERE = Path(__file__).resolve()
_AUTONOMY_PARENT = _HERE.parents[1]
if str(_AUTONOMY_PARENT) not in sys.path:
    sys.path.insert(0, str(_AUTONOMY_PARENT))

import numpy as np
from PIL import Image, ImageDraw

from autonomy.planner import plan_path
from autonomy.usd_occupancy import GridFrame, load_occupancy_npz


def render_path_overlay(
    *,
    occ: np.ndarray,
    gf: GridFrame,
    path: "list[tuple[float, float]] | None",
    start_xy: tuple[float, float],
    goal_xy:  tuple[float, float],
    out_png_path: str,
    upscale: int = 4,
) -> None:
    """Render the occupancy grid with the planned path overlaid.

    - White / light grey: free
    - Black: obstacle
    - Red polyline: planned path
    - Green dot: start
    - Blue dot: goal (the requested goal, not the snap)

    Image rows flipped so +Y points up. ``upscale`` factor pixel-doubles
    the source grid for visibility (5 cm cells become tiny otherwise).
    """
    H, W = occ.shape
    img = np.where(occ, 50, 230).astype(np.uint8)
    img = np.flipud(img)
    pil = Image.fromarray(img, mode="L").convert("RGB").resize(
        (W * upscale, H * upscale), resample=Image.NEAREST,
    )
    draw = ImageDraw.Draw(pil)

    def world_to_image(x: float, y: float) -> tuple[int, int]:
        ix, iy = gf.world_to_grid(x, y)
        # Image rows flipped (+Y up); upscale.
        ix_px = int((ix + 0.5) * upscale)
        iy_px = int((H - 1 - iy + 0.5) * upscale)
        return ix_px, iy_px

    if path is not None and len(path) >= 2:
        pts = [world_to_image(x, y) for x, y in path]
        draw.line(pts, fill=(220, 30, 30), width=max(1, upscale // 2))
        for px, py in pts[1:-1]:
            r = max(1, upscale // 2)
            draw.ellipse((px - r, py - r, px + r, py + r), fill=(255, 100, 0))

    sx, sy = world_to_image(*start_xy)
    gx, gy = world_to_image(*goal_xy)
    rs = max(2, upscale)
    draw.ellipse((sx - rs, sy - rs, sx + rs, sy + rs), fill=(40, 200, 40))      # green = start
    draw.ellipse((gx - rs, gy - rs, gx + rs, gy + rs), fill=(40, 80, 230))      # blue  = goal
    pil.save(out_png_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True, help="Path to room.occupancy.npz")
    parser.add_argument("--start", nargs=2, type=float, required=True, metavar=("X", "Y"))
    parser.add_argument("--goal",  nargs=2, type=float, required=True, metavar=("X", "Y"))
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--inflation", type=float, default=0.40, help="Robot footprint clearance (m)")
    parser.add_argument("--no-smooth", action="store_true", help="Disable line-of-sight smoothing")
    args = parser.parse_args()

    occ, gf = load_occupancy_npz(args.map)
    print(f"[plan_path_demo] map: {gf.width}x{gf.height} cells, "
          f"{gf.width * gf.resolution_m:.2f} x {gf.height * gf.resolution_m:.2f} m  "
          f"({100.0 * occ.sum() / occ.size:.1f}% occupied)")

    start = (args.start[0], args.start[1])
    goal  = (args.goal[0],  args.goal[1])

    path = plan_path(
        start, goal, occ, gf,
        inflation_m=args.inflation,
        smooth=not args.no_smooth,
    )
    if path is None:
        print(f"[plan_path_demo] NO PATH from {start} → {goal}")
    else:
        # Path length in metres (sum of segment lengths).
        seg_lens = [
            ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
            for a, b in zip(path[:-1], path[1:])
        ]
        print(f"[plan_path_demo] path: {len(path)} waypoints, "
              f"length {sum(seg_lens):.2f} m, inflation {args.inflation:.2f} m")
        for i, (x, y) in enumerate(path):
            print(f"  [{i:2d}] ({x:+.2f}, {y:+.2f})")

    render_path_overlay(
        occ=occ, gf=gf, path=path,
        start_xy=start, goal_xy=goal,
        out_png_path=args.out,
    )
    print(f"[plan_path_demo] wrote {args.out}")
    return 0 if path is not None else 1


if __name__ == "__main__":
    sys.exit(main())
