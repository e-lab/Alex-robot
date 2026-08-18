#!/usr/bin/env python3
# Copyright (c) 2026, Patrizio Acquadro.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate deterministic convex colliders for the qualified WSG32 asset."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = REPOSITORY_ROOT / "assets" / "end_effectors" / "weiss_wsg32"
WSG_MESH_ROOT = ASSET_ROOT / "source" / "wsg_32_description" / "meshes"
UMI_FINGER = (
    ASSET_ROOT
    / "source"
    / "actuated_umi"
    / "3d-printables"
    / "actuated-UMI-v1"
    / "UMI-gripper-soft-gripper-finger.stl"
)
COLLISION_ROOT = ASSET_ROOT / "meshes" / "collision"


@dataclass(frozen=True)
class Collider:
    source: Path
    output: Path
    minimum_face_center_z_mm: float | None = None
    maximum_face_center_z_mm: float | None = None


COLLIDERS = (
    Collider(WSG_MESH_ROOT / "hand_link.STL", COLLISION_ROOT / "hand_link_convex.stl"),
    Collider(
        WSG_MESH_ROOT / "left_finger_link.STL",
        COLLISION_ROOT / "jaw_link_convex.stl",
    ),
    Collider(
        UMI_FINGER,
        COLLISION_ROOT / "umi_v1_mount_convex.stl",
        minimum_face_center_z_mm=-40.0,
    ),
    Collider(
        UMI_FINGER,
        COLLISION_ROOT / "umi_v1_contact_convex.stl",
        maximum_face_center_z_mm=-30.0,
    ),
)


def _convex_stl_bytes(collider: Collider) -> bytes:
    try:
        import trimesh
        from trimesh.exchange.stl import export_stl
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "convex generation requires trimesh; run with the Isaac Lab Python"
        ) from error

    mesh = trimesh.load(collider.source, force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"failed to load a non-empty mesh from {collider.source}")

    centers_z = mesh.triangles_center[:, 2]
    selected = centers_z == centers_z
    if collider.minimum_face_center_z_mm is not None:
        selected &= centers_z >= collider.minimum_face_center_z_mm
    if collider.maximum_face_center_z_mm is not None:
        selected &= centers_z <= collider.maximum_face_center_z_mm
    if not selected.any():
        raise ValueError(f"empty collider partition for {collider.output.name}")
    if not selected.all():
        mesh = mesh.submesh([selected.nonzero()[0]], append=True, repair=False)

    hull = mesh.convex_hull
    if not hull.is_convex or not hull.is_watertight or hull.volume <= 0.0:
        raise ValueError(f"invalid convex hull generated from {collider.source}")
    return export_stl(hull)


def _write_or_check(path: Path, content: bytes, *, check: bool) -> None:
    if check:
        if not path.is_file() or path.read_bytes() != content:
            raise ValueError(f"generated artifact is stale: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def generate(*, check: bool) -> None:
    for collider in COLLIDERS:
        _write_or_check(
            collider.output,
            _convex_stl_bytes(collider),
            check=check,
        )
    mode = "validated" if check else "generated"
    print(f"PASS: {mode} {len(COLLIDERS)} WSG32 convex colliders")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="fail if generated colliders are stale"
    )
    args = parser.parse_args()
    try:
        generate(check=args.check)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
