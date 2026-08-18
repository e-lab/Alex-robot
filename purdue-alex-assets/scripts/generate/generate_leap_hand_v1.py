#!/usr/bin/env python3
"""Generate portable LEAP Hand V1 URDFs from immutable official sources."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = REPOSITORY_ROOT / "assets" / "end_effectors" / "leap_hand_v1"

_SIDE_FILES = {
    "left": (
        "dip.stl",
        "fingertip.stl",
        "mcp_joint.stl",
        "palm_lower_left.stl",
        "pip.stl",
        "thumb_dip.stl",
        "thumb_fingertip.stl",
        "thumb_left_temp_base.stl",
        "thumb_pip.stl",
    ),
    "right": (
        "dip.stl",
        "fingertip.stl",
        "mcp_joint.stl",
        "palm_lower.stl",
        "pip.stl",
        "thumb_dip.stl",
        "thumb_fingertip.stl",
        "thumb_pip.stl",
    ),
}


def portable_urdf(source: bytes, side: str) -> bytes:
    """Rewrite only official mesh URIs, preserving all physical XML bytes."""

    result = source
    for filename in _SIDE_FILES[side]:
        original = (
            f'filename="package:///{filename}"'.encode()
            if side == "left"
            else f'filename="{filename}"'.encode()
        )
        replacement = f'filename="../source/{side}/{filename}"'.encode()
        occurrences = result.count(original)
        if occurrences == 0:
            raise ValueError(f"official {side} URDF does not reference {filename}")
        result = result.replace(original, replacement)
    return result


def generated_urdfs() -> dict[Path, bytes]:
    outputs = {}
    for side in ("left", "right"):
        source = ASSET_ROOT / "source" / side / "robot.urdf"
        outputs[ASSET_ROOT / "urdf" / f"leap_hand_v1_{side}.urdf"] = portable_urdf(
            source.read_bytes(), side
        )
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="fail if tracked outputs are stale"
    )
    args = parser.parse_args()

    stale = []
    for destination, expected in generated_urdfs().items():
        if args.check:
            if not destination.is_file() or destination.read_bytes() != expected:
                stale.append(destination.relative_to(REPOSITORY_ROOT).as_posix())
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(expected)

    if stale:
        print("stale LEAP Hand V1 generated URDFs:", file=sys.stderr)
        for path in stale:
            print(f"  {path}", file=sys.stderr)
        return 1
    if args.check:
        print("PASS: LEAP Hand V1 portable URDFs match immutable sources")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
