# Copyright (c) 2026, Patrizio Acquadro.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone URDF integration for the Purdue Alex003 pedestal."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import isaaclab.sim as sim_utils
import yaml
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyBaseCfg

from .._paths import REPOSITORY_ROOT
from ..robots.purdue_frames import ALEX_PURDUE_ASSET_ROOT

PURDUE_ALEX003_MEASUREMENTS_PATH = REPOSITORY_ROOT / "measurements.yaml"
"""Canonical physical measurement record for the Purdue Alex003 installation."""

_ALEX_PURDUE_REFERENCE_URDF_PATH = (
    ALEX_PURDUE_ASSET_ROOT / "urdf" / "baseline" / "alex_purdue_full_convex.urdf"
)

PURDUE_ALEX003_PEDESTAL_ASSET_ROOT = (
    REPOSITORY_ROOT / "assets" / "platforms" / "purdue_alex003_pedestal"
)
"""Root directory of the standalone floor-standing pedestal asset."""

PURDUE_ALEX003_PEDESTAL_URDF_PATH = (
    PURDUE_ALEX003_PEDESTAL_ASSET_ROOT / "urdf" / "purdue_alex003_pedestal.urdf"
)
"""Canonical portable URDF for the Purdue Alex003 floor-standing pedestal."""

_PART_NAMES = ("lower_base", "column", "upper_mount")


@dataclass(frozen=True)
class PurduePedestalPartSpec:
    """One measured axis-aligned component of the floor-standing pedestal."""

    name: str
    size_xyz_m: tuple[float, float, float]
    center_xyz_m: tuple[float, float, float]
    z_bounds_m: tuple[float, float]


@dataclass(frozen=True)
class PurdueAlex003PedestalSpec:
    """Validated physical geometry and robot alignment for Purdue Alex003."""

    source_path: Path
    model_id: str
    parts: tuple[PurduePedestalPartSpec, ...]
    mounting_plane_world_z_m: float
    alex_root_world_z_m: float
    right_shoulder_y_world_z_m: float


def _mapping(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")
    return value


def _finite_number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{path} must be a finite number")
    return number


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-9)


def _joint_origin_z_m(path: Path, joint_name: str) -> float:
    if not path.is_file():
        raise FileNotFoundError(f"Alex Purdue reference URDF does not exist: {path}")
    joint = ET.parse(path).getroot().find(f"joint[@name='{joint_name}']")
    if joint is None:
        raise ValueError(f"Alex Purdue reference URDF has no {joint_name} joint")
    origin = joint.find("origin")
    if origin is None:
        raise ValueError(f"Alex Purdue {joint_name} joint has no origin")
    try:
        xyz = tuple(float(value) for value in origin.get("xyz", "").split())
    except ValueError as error:
        raise ValueError(f"Alex Purdue {joint_name} origin must be numeric") from error
    if len(xyz) != 3 or not all(math.isfinite(value) for value in xyz):
        raise ValueError(f"Alex Purdue {joint_name} origin must be a finite xyz vector")
    return xyz[2]


def load_purdue_alex003_pedestal_spec(
    measurement_path: str | Path | None = None,
) -> PurdueAlex003PedestalSpec:
    """Load and validate the measured pedestal stack and Alex world alignment."""

    path = Path(measurement_path or PURDUE_ALEX003_MEASUREMENTS_PATH).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Purdue Alex003 measurement record does not exist: {path}"
        )
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    record = _mapping(loaded, "measurement")

    if record.get("schema") != {
        "name": "purdue_alex003_physical_measurements",
        "version": 2,
    }:
        raise ValueError("unexpected Purdue Alex003 measurement schema")
    if set(record) != {"schema", "setup", "robot", "pedestal", "end_effectors"}:
        raise ValueError("unexpected Purdue Alex003 measurement sections")

    setup = _mapping(record.get("setup"), "setup")
    if setup.get("id") != "purdue_alex003":
        raise ValueError("setup.id must be 'purdue_alex003'")
    if setup.get("units") != {"length": "m", "angle": "rad", "mass": "kg"}:
        raise ValueError("Purdue Alex003 measurements must use SI units")
    if setup.get("coordinate_convention") != {
        "x": "forward",
        "y": "left",
        "z": "up",
        "handedness": "right_handed",
    }:
        raise ValueError("unexpected Purdue Alex003 coordinate convention")

    pedestal = _mapping(record.get("pedestal"), "pedestal")
    model_id = pedestal.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("pedestal.model_id must be a non-empty string")

    mounting_world_z = _mapping(
        pedestal.get("mounting_plane_world_z_m"),
        "pedestal.mounting_plane_world_z_m",
    )
    mounting_plane_world_z_m = _finite_number(
        mounting_world_z.get("value"), "pedestal.mounting_plane_world_z_m.value"
    )

    pedestal_parts = _mapping(pedestal.get("parts"), "pedestal.parts")
    if set(pedestal_parts) != set(_PART_NAMES):
        raise ValueError("pedestal.parts must define the complete measured stack")
    parts: list[PurduePedestalPartSpec] = []
    previous_maximum = 0.0
    for name in _PART_NAMES:
        part_path = f"pedestal.parts.{name}"
        part = _mapping(pedestal_parts.get(name), part_path)
        size = _mapping(part.get("size_m"), f"{part_path}.size_m")
        if set(size) != {"x", "y", "z"}:
            raise ValueError(f"{part_path}.size_m must define x, y, and z")
        depth_x = _finite_number(size.get("x"), f"{part_path}.size_m.x")
        width_y = _finite_number(size.get("y"), f"{part_path}.size_m.y")
        height_z = _finite_number(size.get("z"), f"{part_path}.size_m.z")
        if min(depth_x, width_y, height_z) <= 0.0:
            raise ValueError(f"{part_path} dimensions must be positive")

        minimum = previous_maximum
        maximum = minimum + height_z
        center = (minimum + maximum) / 2.0
        parts.append(
            PurduePedestalPartSpec(
                name=name,
                size_xyz_m=(depth_x, width_y, height_z),
                center_xyz_m=(0.0, 0.0, center),
                z_bounds_m=(minimum, maximum),
            )
        )
        previous_maximum = maximum

    if not _close(previous_maximum, mounting_plane_world_z_m):
        raise ValueError("pedestal stack does not terminate at the mounting plane")

    robot = _mapping(record.get("robot"), "robot")
    shoulder_world = _mapping(
        robot.get("right_shoulder_y_world_z_m"),
        "robot.right_shoulder_y_world_z_m",
    )
    right_shoulder_y_world_z_m = _finite_number(
        shoulder_world.get("value"),
        "robot.right_shoulder_y_world_z_m.value",
    )
    right_shoulder_y_from_root_m = _joint_origin_z_m(
        _ALEX_PURDUE_REFERENCE_URDF_PATH, "RIGHT_SHOULDER_Y"
    )
    alex_root_world_z_m = right_shoulder_y_world_z_m - right_shoulder_y_from_root_m

    return PurdueAlex003PedestalSpec(
        source_path=path,
        model_id=model_id,
        parts=tuple(parts),
        mounting_plane_world_z_m=mounting_plane_world_z_m,
        alex_root_world_z_m=alex_root_world_z_m,
        right_shoulder_y_world_z_m=right_shoulder_y_world_z_m,
    )


def _resolve_purdue_alex003_pedestal_urdf(
    asset_path: str | Path | None,
) -> Path:
    path = Path(asset_path or PURDUE_ALEX003_PEDESTAL_URDF_PATH).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Purdue Alex003 pedestal URDF does not exist: {path}")
    if path.suffix.lower() != ".urdf":
        raise ValueError(f"Purdue Alex003 pedestal asset must be a URDF: {path}")
    return path


def make_purdue_alex003_pedestal_cfg(
    prim_path: str = "{ENV_REGEX_NS}/PurduePedestal",
    *,
    asset_path: str | Path | None = None,
) -> AssetBaseCfg:
    """Return an independent fixed-base configuration for the standalone URDF."""

    urdf_path = _resolve_purdue_alex003_pedestal_urdf(asset_path)
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.UrdfFileCfg(
            asset_path=urdf_path.as_posix(),
            fix_base=True,
            link_density=1000.0,
            merge_fixed_joints=True,
            joint_drive=None,
            collision_from_visuals=False,
            collision_type="Convex Hull",
            self_collision=False,
            rigid_props=RigidBodyBaseCfg(disable_gravity=False),
            # The layered transformer drops PhysX data for this 0-DOF URDF.
            run_asset_transformer=False,
            run_multi_physics_conversion=False,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )


__all__ = [
    "PurdueAlex003PedestalSpec",
    "PurduePedestalPartSpec",
    "load_purdue_alex003_pedestal_spec",
    "make_purdue_alex003_pedestal_cfg",
]
