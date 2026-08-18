# Copyright (c) 2026, Patrizio Acquadro.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure URDF parsing and USD authoring for Alex Purdue fixed frames."""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Literal

from .._paths import REPOSITORY_ROOT

AlexPurdueVariant = Literal["source", "full_convex"]
AlexPurdueEndEffector = Literal["sake_ezgripper", "wsg32_umi_v1"]
PurdueFrameValue = str | tuple[float, float, float]

_DEFAULT_ASSET_ROOT = REPOSITORY_ROOT / "assets" / "robots" / "alex_purdue"
ALEX_PURDUE_ASSET_ROOT = (
    Path(os.environ.get("ALEX_PURDUE_ASSET_ROOT", _DEFAULT_ASSET_ROOT))
    .expanduser()
    .resolve()
)
"""Root of the simulator-neutral Alex Purdue asset."""

_ALEX_PURDUE_URDF_PATHS = {
    ("sake_ezgripper", "source"): (
        ALEX_PURDUE_ASSET_ROOT / "urdf" / "baseline" / "alex_purdue_full.urdf"
    ),
    ("sake_ezgripper", "full_convex"): (
        ALEX_PURDUE_ASSET_ROOT / "urdf" / "baseline" / "alex_purdue_full_convex.urdf"
    ),
    ("wsg32_umi_v1", "source"): (
        ALEX_PURDUE_ASSET_ROOT / "urdf" / "baseline" / "alex_purdue_wsg32_umi_v1.urdf"
    ),
    ("wsg32_umi_v1", "full_convex"): (
        ALEX_PURDUE_ASSET_ROOT
        / "urdf"
        / "baseline"
        / "alex_purdue_wsg32_umi_v1_full_convex.urdf"
    ),
}

_SPECIAL_FRAMES = {
    "left_ezgripper_palm_link": ("palm", "LEFT_EZGRIPPER_PALM_FRAME"),
    "right_ezgripper_palm_link": ("palm", "RIGHT_EZGRIPPER_PALM_FRAME"),
    "LEFT_EZGRIPPER_TCP_LINK": ("tcp", "LEFT_EZGRIPPER_TCP_FRAME"),
    "RIGHT_EZGRIPPER_TCP_LINK": ("tcp", "RIGHT_EZGRIPPER_TCP_FRAME"),
    "left_WSG32_BASE_LINK": ("base", "LEFT_WSG32_BASE_FRAME"),
    "right_WSG32_BASE_LINK": ("base", "RIGHT_WSG32_BASE_FRAME"),
    "left_WSG32_TCP_LINK": ("tcp", "LEFT_WSG32_TCP_FRAME"),
    "right_WSG32_TCP_LINK": ("tcp", "RIGHT_WSG32_TCP_FRAME"),
}


def _resolve_alex_purdue_urdf(
    asset_path: str | os.PathLike[str] | None,
    variant: AlexPurdueVariant,
    end_effector: AlexPurdueEndEffector = "sake_ezgripper",
) -> Path:
    if asset_path is not None and (
        variant != "full_convex" or end_effector != "sake_ezgripper"
    ):
        raise ValueError(
            "asset_path and a non-default Alex Purdue profile are mutually exclusive"
        )
    if asset_path is None:
        expected_end_effectors = sorted(
            {profile[0] for profile in _ALEX_PURDUE_URDF_PATHS}
        )
        expected_variants = sorted({profile[1] for profile in _ALEX_PURDUE_URDF_PATHS})
        if variant not in expected_variants:
            raise ValueError(
                f"unknown Alex Purdue variant {variant!r}; expected one of: "
                + ", ".join(expected_variants)
            )
        if end_effector not in expected_end_effectors:
            raise ValueError(
                f"unknown Alex Purdue end effector {end_effector!r}; expected one of: "
                + ", ".join(expected_end_effectors)
            )
        try:
            path = _ALEX_PURDUE_URDF_PATHS[(end_effector, variant)]
        except KeyError as error:
            raise ValueError(
                f"unknown Alex Purdue profile end_effector={end_effector!r}, "
                f"variant={variant!r}; expected end effectors: "
                f"{', '.join(expected_end_effectors)}; variants: "
                f"{', '.join(expected_variants)}"
            ) from error
    else:
        path = Path(asset_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Alex Purdue URDF does not exist: {path}")
    return path


def _vector(text: str | None) -> tuple[float, float, float]:
    values = tuple(float(item) for item in (text or "0 0 0").split())
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"expected a finite three-vector, got {text!r}")
    return values[0], values[1], values[2]


def alex_purdue_frame_specs(
    asset_path: str | os.PathLike[str] | None = None,
    *,
    variant: AlexPurdueVariant = "full_convex",
    end_effector: AlexPurdueEndEffector = "sake_ezgripper",
) -> tuple[dict[str, PurdueFrameValue], ...]:
    """Read the 11 sensor and four end-effector fixed transforms."""

    path = _resolve_alex_purdue_urdf(asset_path, variant, end_effector)
    robot = ET.parse(path).getroot()
    if robot.tag != "robot" or robot.get("name") != "AlexPurdue":
        raise ValueError(f"unexpected Alex Purdue URDF root in {path}")

    specs: list[dict[str, PurdueFrameValue]] = []
    seen: set[str] = set()
    for joint in robot.findall("joint"):
        if joint.get("type") != "fixed":
            continue
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_link = parent.get("link", "")
        source_link = child.get("link", "")
        searchable = f"{joint.get('name', '')} {source_link}".upper()
        special = _SPECIAL_FRAMES.get(source_link)
        if special is not None:
            kind, frame = special
        elif "ZED" in searchable or "CAMERA" in searchable:
            kind = "camera"
            frame = source_link.removesuffix("_LINK") + "_FRAME"
        elif "IMU" in searchable:
            kind = "imu"
            frame = source_link.removesuffix("_LINK") + "_FRAME"
        else:
            continue
        if not parent_link or not source_link:
            raise ValueError(
                f"incomplete Alex Purdue fixed frame joint {joint.get('name')!r}"
            )
        if frame in seen:
            raise ValueError(f"duplicate Alex Purdue frame {frame!r}")
        seen.add(frame)
        origin = joint.find("origin")
        specs.append(
            {
                "kind": kind,
                "source_link": source_link,
                "frame": frame,
                "parent_link": parent_link,
                "xyz_m": _vector(origin.get("xyz") if origin is not None else None),
                "rpy_rad": _vector(origin.get("rpy") if origin is not None else None),
            }
        )

    required = {
        "HEAD_ZED_X_MINI_FRAME",
        "HEAD_IMU_FRAME",
    }
    if end_effector == "sake_ezgripper":
        required.update(
            {
                "LEFT_EZGRIPPER_PALM_FRAME",
                "RIGHT_EZGRIPPER_PALM_FRAME",
                "LEFT_EZGRIPPER_TCP_FRAME",
                "RIGHT_EZGRIPPER_TCP_FRAME",
            }
        )
    else:
        required.update(
            {
                "LEFT_WSG32_BASE_FRAME",
                "RIGHT_WSG32_BASE_FRAME",
                "LEFT_WSG32_TCP_FRAME",
                "RIGHT_WSG32_TCP_FRAME",
            }
        )
    if len(specs) != 15 or not required.issubset(seen):
        raise ValueError(
            f"Alex Purdue fixed-frame contract drifted: count={len(specs)}, missing={sorted(required - seen)}"
        )
    return tuple(specs)


def author_alex_purdue_frames(
    stage: Any,
    robot_prim_path: str,
    asset_path: str | os.PathLike[str] | None = None,
    *,
    variant: AlexPurdueVariant = "full_convex",
    end_effector: AlexPurdueEndEffector = "sake_ezgripper",
) -> dict[str, str]:
    """Recreate merged Purdue sensor, palm, and TCP joints as coordinate-only Xforms."""

    from pxr import Gf, Usd, UsdGeom, UsdPhysics  # type: ignore

    robot = stage.GetPrimAtPath(robot_prim_path)
    if not robot or not robot.IsValid():
        raise ValueError(f"Alex Purdue robot prim does not exist: {robot_prim_path}")
    by_name: dict[str, list[Any]] = {}
    for prim in Usd.PrimRange(robot):
        by_name.setdefault(prim.GetName(), []).append(prim)

    urdf = ET.parse(
        _resolve_alex_purdue_urdf(asset_path, variant, end_effector)
    ).getroot()
    fixed_parents: dict[
        str, tuple[str, tuple[float, float, float], tuple[float, float, float]]
    ] = {}
    for joint in urdf.findall("joint"):
        if joint.get("type") != "fixed":
            continue
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        origin = joint.find("origin")
        fixed_parents[child.get("link", "")] = (
            parent.get("link", ""),
            _vector(origin.get("xyz") if origin is not None else None),
            _vector(origin.get("rpy") if origin is not None else None),
        )

    def set_transform(
        xform: Any,
        xyz_m: tuple[float, float, float],
        rpy_rad: tuple[float, float, float],
    ) -> None:
        xformable = UsdGeom.Xformable(xform.GetPrim())
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(*xyz_m))
        xformable.AddRotateXYZOp().Set(
            Gf.Vec3f(*(math.degrees(value) for value in rpy_rad))
        )

    resolved_links: dict[str, Any] = {}

    def resolve_link(link: str) -> Any:
        if link in resolved_links:
            return resolved_links[link]
        matches = by_name.get(link, [])
        if len(matches) == 1:
            resolved_links[link] = matches[0]
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"expected one Alex Purdue parent prim named {link!r}, got {len(matches)}"
            )
        try:
            parent_link, xyz_m, rpy_rad = fixed_parents[link]
        except KeyError as error:
            raise ValueError(
                f"cannot resolve merged Alex Purdue parent link {link!r}"
            ) from error
        parent_prim = resolve_link(parent_link)
        anchor_path = f"{parent_prim.GetPath()}/{link}_FRAME_ANCHOR"
        anchor = UsdGeom.Xform.Define(stage, anchor_path)
        set_transform(anchor, xyz_m, rpy_rad)
        anchor.GetPrim().SetCustomDataByKey("alexPurdueMergedLink", link)
        resolved_links[link] = anchor.GetPrim()
        return anchor.GetPrim()

    authored: dict[str, str] = {}
    authored_by_source_link: dict[str, Any] = {}
    for spec in alex_purdue_frame_specs(
        asset_path, variant=variant, end_effector=end_effector
    ):
        parent_link = str(spec["parent_link"])
        if parent_link in authored_by_source_link:
            parent_prim = authored_by_source_link[parent_link]
        else:
            parent_prim = resolve_link(parent_link)

        frame = str(spec["frame"])
        frame_path = f"{parent_prim.GetPath()}/{frame}"
        existing = stage.GetPrimAtPath(frame_path)
        if existing and existing.IsValid():
            if (
                not existing.IsA(UsdGeom.Xform)
                or existing.HasAPI(UsdPhysics.RigidBodyAPI)
                or existing.HasAPI(UsdPhysics.CollisionAPI)
                or existing.HasAPI(UsdPhysics.MassAPI)
            ):
                raise ValueError(
                    f"Alex Purdue frame path is not a coordinate-only Xform: {frame_path}"
                )
            xform = UsdGeom.Xform(existing)
        else:
            xform = UsdGeom.Xform.Define(stage, frame_path)
        xyz_m = spec["xyz_m"]
        rpy_rad = spec["rpy_rad"]
        if not isinstance(xyz_m, tuple) or not isinstance(rpy_rad, tuple):
            raise ValueError(
                f"Alex Purdue frame transform has invalid vectors: {spec!r}"
            )
        set_transform(xform, xyz_m, rpy_rad)
        xform.GetPrim().SetCustomDataByKey("alexPurdueFrameKind", str(spec["kind"]))
        xform.GetPrim().SetCustomDataByKey(
            "alexPurdueSourceLink", str(spec["source_link"])
        )
        authored[frame] = frame_path
        authored_by_source_link[str(spec["source_link"])] = xform.GetPrim()
    return authored


__all__ = [
    "AlexPurdueEndEffector",
    "AlexPurdueVariant",
    "alex_purdue_frame_specs",
    "author_alex_purdue_frames",
]
