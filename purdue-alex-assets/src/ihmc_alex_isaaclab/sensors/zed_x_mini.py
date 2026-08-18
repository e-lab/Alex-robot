# Copyright (c) 2026, Patrizio Acquadro.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Opt-in official Stereolabs ZED X Mini Wide integration for Isaac Lab."""

from __future__ import annotations

import math
import os
from typing import Any, Literal

from ihmc_alex_isaaclab.robots.purdue_frames import alex_purdue_frame_specs

from .._usd import define_fixed_joint, existing_fixed_joint, finite_vector
from .zed_x_mini_dependency import (
    load_zed_x_mini_manifest,
    resolve_zed_isaac_sim_root,
)

ZedXMiniResolution = Literal["SVGA", "HD1080", "HD1200"]

_MANIFEST = load_zed_x_mini_manifest()
_ASSET = _MANIFEST["asset"]
_ALIGNMENT = _MANIFEST["legacy_alignment"]
_RESOLUTIONS = _MANIFEST["resolutions"]

ZED_X_MINI_MODEL = str(_ASSET["model"])
ZED_X_MINI_LENS = str(_ASSET["lens"])
ZED_X_MINI_MASS_KG = float(_ASSET["mass_kg"])
ZED_X_MINI_CENTER_OF_MASS_M = (-0.008286575, 0.0, 0.015909010)
ZED_X_MINI_DIAGONAL_INERTIA_KG_M2 = (
    0.000122873,
    0.000041011,
    0.000138406,
)
ZED_X_MINI_DEFAULT_RESOLUTION = "SVGA"
ZED_X_MINI_RESOLUTIONS = {
    name: tuple(int(value) for value in size) for name, size in _RESOLUTIONS.items()
}
ZED_X_MINI_USD_ROOT_FROM_LEGACY_FRAME_XYZ_M = tuple(
    float(value) for value in _ALIGNMENT["official_usd_root_from_legacy_frame_xyz_m"]
)
_MODEL_SUBPATH = "base_link/ZED_XM"
_LEFT_CAMERA_SUBPATH = f"{_MODEL_SUBPATH}/CameraLeft"
_RIGHT_CAMERA_SUBPATH = f"{_MODEL_SUBPATH}/CameraRight"
_MOUNT_JOINT_NAME = "AlexPurdueHeadZedXMiniFixedJoint"


def make_zed_x_mini_cfgs(
    prim_path: str,
    *,
    resolution: ZedXMiniResolution = ZED_X_MINI_DEFAULT_RESOLUTION,
    zed_isaac_sim_root: str | os.PathLike[str] | None = None,
    spawn_init_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[Any, Any, Any]:
    """Return independent official-model, left RGB/depth, and right RGB cfgs."""

    if not isinstance(prim_path, str) or not prim_path.strip():
        raise ValueError("prim_path must be a non-empty string")
    if resolution not in ZED_X_MINI_RESOLUTIONS:
        raise ValueError(
            f"unsupported ZED X Mini resolution {resolution!r}; expected one of: "
            + ", ".join(ZED_X_MINI_RESOLUTIONS)
        )
    initial_position = finite_vector(spawn_init_pos, length=3, name="spawn_init_pos")
    dependency = resolve_zed_isaac_sim_root(zed_isaac_sim_root)

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab.sim.schemas.schemas_cfg import RigidBodyBaseCfg

    width, height = ZED_X_MINI_RESOLUTIONS[resolution]
    root = prim_path.rstrip("/")
    model_cfg = AssetBaseCfg(
        prim_path=root,
        spawn=sim_utils.UsdFileCfg(
            usd_path=dependency.usd_path.as_posix(),
            rigid_props=RigidBodyBaseCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=ZED_X_MINI_MASS_KG),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=initial_position),
    )
    left_camera_cfg = CameraCfg(
        prim_path=f"{root}/{_LEFT_CAMERA_SUBPATH}",
        update_period=0.0,
        height=height,
        width=width,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=None,
    )
    right_camera_cfg = CameraCfg(
        prim_path=f"{root}/{_RIGHT_CAMERA_SUBPATH}",
        update_period=0.0,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=None,
    )
    return model_cfg, left_camera_cfg, right_camera_cfg


def _rotate_rpy(
    vector: tuple[float, float, float], rpy_rad: tuple[float, float, float]
) -> tuple[float, float, float]:
    roll, pitch, yaw = rpy_rad
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    x, y, z = vector
    return (
        cy * cp * x + (cy * sp * sr - sy * cr) * y + (cy * sp * cr + sy * sr) * z,
        sy * cp * x + (sy * sp * sr + cy * cr) * y + (sy * sp * cr - cy * sr) * z,
        -sp * x + cp * sr * y + cp * cr * z,
    )


def _quaternion_from_rpy(
    rpy_rad: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    roll, pitch, yaw = rpy_rad
    sr, cr = math.sin(roll / 2.0), math.cos(roll / 2.0)
    sp, cp = math.sin(pitch / 2.0), math.cos(pitch / 2.0)
    sy, cy = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _alex_head_mount() -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    matches = [
        spec
        for spec in alex_purdue_frame_specs(
            variant="full_convex", end_effector="wsg32_umi_v1"
        )
        if spec["frame"] == "HEAD_ZED_X_MINI_FRAME"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one Alex Purdue HEAD_ZED_X_MINI_FRAME, got {len(matches)}"
        )
    spec = matches[0]
    if spec["parent_link"] != "HEAD_LINK":
        raise ValueError(
            "Alex Purdue HEAD_ZED_X_MINI_FRAME is not attached to HEAD_LINK"
        )
    return tuple(spec["xyz_m"]), tuple(spec["rpy_rad"])


def _expected_mount_pose() -> tuple[
    tuple[float, float, float], tuple[float, float, float, float]
]:
    legacy_xyz, legacy_rpy = _alex_head_mount()
    rotated_offset = _rotate_rpy(
        ZED_X_MINI_USD_ROOT_FROM_LEGACY_FRAME_XYZ_M, legacy_rpy
    )
    position = tuple(
        legacy + offset
        for legacy, offset in zip(legacy_xyz, rotated_offset, strict=True)
    )
    return position, _quaternion_from_rpy(legacy_rpy)


def author_alex_purdue_zed_x_mini_mount(
    stage: Any, robot_prim_path: str, zed_prim_path: str
) -> str:
    """Idempotently fix the official ZED root to the Golden Robot ``HEAD_LINK``."""

    import isaaclab.sim as sim_utils
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    if stage is None:
        raise ValueError("USD stage is required to author the ZED X Mini mount")
    dependency = resolve_zed_isaac_sim_root()
    robot = stage.GetPrimAtPath(robot_prim_path)
    if not robot or not robot.IsValid():
        raise ValueError(f"Alex Purdue robot prim does not exist: {robot_prim_path}")
    heads = [prim for prim in Usd.PrimRange(robot) if prim.GetName() == "HEAD_LINK"]
    if len(heads) != 1:
        raise ValueError(
            f"expected one HEAD_LINK below {robot_prim_path}, got {len(heads)}"
        )
    head = heads[0]
    if not head.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError(f"Alex Purdue HEAD_LINK is not a rigid body: {head.GetPath()}")

    zed = stage.GetPrimAtPath(zed_prim_path)
    if not zed or not zed.IsValid():
        raise ValueError(f"ZED X Mini prim does not exist: {zed_prim_path}")
    if not zed.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError(f"ZED X Mini root is not a rigid body: {zed_prim_path}")
    model = stage.GetPrimAtPath(f"{zed_prim_path.rstrip('/')}/{_MODEL_SUBPATH}")
    if not model or not model.IsValid():
        raise ValueError(
            f"ZED prim is not the pinned {ZED_X_MINI_MODEL} asset: {zed_prim_path}"
        )
    lens = model.GetVariantSets().GetVariantSet("lens").GetVariantSelection()
    if lens != ZED_X_MINI_LENS:
        raise ValueError(
            f"ZED_XM lens must be {ZED_X_MINI_LENS!r}, got {lens or '<missing>'!r}"
        )
    for subpath in (_LEFT_CAMERA_SUBPATH, _RIGHT_CAMERA_SUBPATH):
        camera = stage.GetPrimAtPath(f"{zed_prim_path.rstrip('/')}/{subpath}")
        if not camera or not camera.IsA(UsdGeom.Camera):
            raise ValueError(
                f"official ZED_XM camera prim is missing: {camera.GetPath()}"
            )
    if dependency.usd_path.name != "ZED_XM.usdc":
        raise ValueError(
            f"validated upstream did not resolve ZED_XM.usdc: {dependency.usd_path}"
        )

    position, quaternion = _expected_mount_pose()
    expected_rotation = Gf.Quatf(quaternion[0], *quaternion[1:])
    identity = Gf.Quatf(1.0, 0.0, 0.0, 0.0)
    joint_path = Sdf.Path(f"{zed_prim_path.rstrip('/')}/{_MOUNT_JOINT_NAME}")
    existing_joint_path = existing_fixed_joint(
        stage,
        joint_path,
        head,
        zed,
        position,
        expected_rotation,
        (0.0, 0.0, 0.0),
        identity,
        label="ZED mount",
    )

    mass_api = UsdPhysics.MassAPI.Apply(zed)
    mass_api.CreateMassAttr().Set(ZED_X_MINI_MASS_KG)
    mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*ZED_X_MINI_CENTER_OF_MASS_M))
    mass_api.CreateDiagonalInertiaAttr().Set(
        Gf.Vec3f(*ZED_X_MINI_DIAGONAL_INERTIA_KG_M2)
    )
    mass_api.CreatePrincipalAxesAttr().Set(identity)

    head_world = UsdGeom.Xformable(head).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    mount = Gf.Transform()
    mount.SetTranslation(Gf.Vec3d(*position))
    mount.SetRotation(Gf.Rotation(Gf.Quatd(quaternion[0], *quaternion[1:])))
    desired_world = mount.GetMatrix() * head_world
    zed_parent = zed.GetParent()
    parent_world = UsdGeom.Xformable(zed_parent).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    desired_local = Gf.Transform(desired_world * parent_world.GetInverse())
    desired_local_quaternion = desired_local.GetRotation().GetQuat()
    if not sim_utils.standardize_xform_ops(
        zed,
        translation=tuple(desired_local.GetTranslation()),
        orientation=(
            *tuple(desired_local_quaternion.GetImaginary()),
            float(desired_local_quaternion.GetReal()),
        ),
    ):
        raise ValueError(f"ZED X Mini root is not transformable: {zed_prim_path}")
    if existing_joint_path is not None:
        return existing_joint_path
    return define_fixed_joint(
        stage,
        joint_path,
        head,
        zed,
        position,
        expected_rotation,
        (0.0, 0.0, 0.0),
        identity,
    )
