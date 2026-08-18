# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure URDF parsing and USD authoring for Alex V2 sensor frames."""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Literal

from .._paths import REPOSITORY_ROOT

AlexV2Variant = Literal["standard", "forearm_convex", "full_convex"]
SensorFrameValue = str | tuple[float, float, float]

_DEFAULT_ASSET_ROOT = REPOSITORY_ROOT / "assets" / "robots" / "alex_v2"
ALEX_V2_ASSET_ROOT = (
    Path(os.environ.get("ALEX_V2_ASSET_ROOT", _DEFAULT_ASSET_ROOT))
    .expanduser()
    .resolve()
)
"""Root of the simulator-neutral Alex V2 asset."""

_ALEX_V2_URDF_PATHS = {
    "standard": ALEX_V2_ASSET_ROOT / "urdf" / "alex_v2.urdf",
    "forearm_convex": ALEX_V2_ASSET_ROOT / "urdf" / "alex_v2_forearm_convex.urdf",
    "full_convex": ALEX_V2_ASSET_ROOT / "urdf" / "alex_v2_full_convex.urdf",
}


def _resolve_alex_v2_urdf(
    asset_path: str | os.PathLike[str] | None,
    variant: AlexV2Variant,
) -> Path:
    if asset_path is not None and variant != "standard":
        raise ValueError(
            "asset_path and a non-standard Alex V2 variant are mutually exclusive"
        )
    if asset_path is None:
        try:
            path = _ALEX_V2_URDF_PATHS[variant]
        except KeyError as error:
            expected = ", ".join(sorted(_ALEX_V2_URDF_PATHS))
            raise ValueError(
                f"unknown Alex V2 variant {variant!r}; expected one of: {expected}"
            ) from error
    else:
        path = Path(asset_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Alex V2 URDF does not exist: {path}")
    return path


def _vector(text: str | None) -> tuple[float, float, float]:
    values = tuple(float(item) for item in (text or "0 0 0").split())
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"expected a finite three-vector, got {text!r}")
    return values[0], values[1], values[2]


def alex_v2_sensor_frame_specs(
    asset_path: str | os.PathLike[str] | None = None,
    *,
    variant: AlexV2Variant = "standard",
) -> tuple[dict[str, SensorFrameValue], ...]:
    """Read the 19 fixed camera and IMU transforms in URDF order.

    Positions are expressed in meters and roll-pitch-yaw values in radians.
    """

    path = _resolve_alex_v2_urdf(asset_path, variant)
    robot = ET.parse(path).getroot()
    if robot.tag != "robot" or robot.get("name") != "AlexV2":
        raise ValueError(f"unexpected Alex V2 URDF root in {path}")

    specs: list[dict[str, SensorFrameValue]] = []
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
        if "ZED" in searchable or "CAMERA" in searchable:
            kind = "camera"
        elif "IMU" in searchable:
            kind = "imu"
        else:
            continue
        if not parent_link or not source_link:
            raise ValueError(f"incomplete fixed sensor joint {joint.get('name')!r}")
        frame = source_link.removesuffix("_LINK") + "_FRAME"
        if frame in seen:
            raise ValueError(f"duplicate Alex V2 sensor frame {frame!r}")
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

    required = {"HEAD_ZED_X_MINI_FRAME", "HEAD_IMU_FRAME"}
    if len(specs) != 19 or not required.issubset(seen):
        raise ValueError(
            f"Alex V2 fixed sensor-frame contract drifted: count={len(specs)}, missing={sorted(required - seen)}"
        )
    return tuple(specs)


def author_alex_v2_sensor_frames(
    stage: Any,
    robot_prim_path: str,
    asset_path: str | os.PathLike[str] | None = None,
    *,
    variant: AlexV2Variant = "standard",
) -> dict[str, str]:
    """Recreate merged Alex camera and IMU joints as coordinate-only Xforms."""

    from pxr import Gf, Usd, UsdGeom, UsdPhysics  # type: ignore

    robot = stage.GetPrimAtPath(robot_prim_path)
    if not robot or not robot.IsValid():
        raise ValueError(f"Alex V2 robot prim does not exist: {robot_prim_path}")
    by_name: dict[str, list[Any]] = {}
    for prim in Usd.PrimRange(robot):
        by_name.setdefault(prim.GetName(), []).append(prim)

    authored: dict[str, str] = {}
    for spec in alex_v2_sensor_frame_specs(asset_path, variant=variant):
        parent_link = str(spec["parent_link"])
        parents = by_name.get(parent_link, [])
        if len(parents) != 1:
            raise ValueError(
                f"expected one Alex V2 parent prim named {parent_link!r}, got {len(parents)}"
            )
        frame = str(spec["frame"])
        frame_path = f"{parents[0].GetPath()}/{frame}"
        existing = stage.GetPrimAtPath(frame_path)
        if existing and existing.IsValid():
            if (
                not existing.IsA(UsdGeom.Xform)
                or existing.HasAPI(UsdPhysics.RigidBodyAPI)
                or existing.HasAPI(UsdPhysics.CollisionAPI)
                or existing.HasAPI(UsdPhysics.MassAPI)
            ):
                raise ValueError(
                    f"Alex V2 sensor frame path is not a coordinate-only Xform: {frame_path}"
                )
            xform = UsdGeom.Xform(existing)
        else:
            xform = UsdGeom.Xform.Define(stage, frame_path)
        xformable = UsdGeom.Xformable(xform.GetPrim())
        xformable.ClearXformOpOrder()
        xyz_m = spec["xyz_m"]
        rpy_rad = spec["rpy_rad"]
        if not isinstance(xyz_m, tuple) or not isinstance(rpy_rad, tuple):
            raise ValueError(f"Alex V2 sensor transform has invalid vectors: {spec!r}")
        xformable.AddTranslateOp().Set(Gf.Vec3d(*xyz_m))
        xformable.AddRotateXYZOp().Set(
            Gf.Vec3f(*(math.degrees(value) for value in rpy_rad))
        )
        xform.GetPrim().SetCustomDataByKey("alexV2SensorKind", str(spec["kind"]))
        xform.GetPrim().SetCustomDataByKey("alexV2SourceLink", str(spec["source_link"]))
        authored[frame] = frame_path
    return authored


__all__ = ["alex_v2_sensor_frame_specs", "author_alex_v2_sensor_frames"]
