# Copyright (c) 2026, Patrizio Acquadro.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Official RealSense D405 and D435 geometry with aligned Isaac RGB-D."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyBaseCfg

from .._usd import (
    define_fixed_joint,
    existing_fixed_joint,
    finite_vector,
    matrix_pose,
    normalized_xyzw,
)
from .._paths import REPOSITORY_ROOT

REALSENSE_ASSET_ROOT = (REPOSITORY_ROOT / "assets" / "sensors" / "realsense").resolve()
REALSENSE_DEFAULT_USD_ROOT = (
    REPOSITORY_ROOT / "build" / "isaac" / "realsense"
).resolve()

D405_RESOLUTION = (1280, 720)
D405_FOV_DEG = (87.0, 58.0)
D405_CLIPPING_RANGE_M = (0.07, 1.0)
D435_RESOLUTION = (1920, 1080)
D435_FOV_DEG = (69.0, 42.0)
D435_CLIPPING_RANGE_M = (0.28, 3.0)

_MODEL_SUBPATH = "Geometry/base_link/camera_bottom_screw_frame/camera_link"
_BOTTOM_SCREW_SUBPATH = "Geometry/base_link/camera_bottom_screw_frame"
_COLOR_OPTICAL_SUBPATH = (
    f"{_MODEL_SUBPATH}/camera_color_frame/camera_color_optical_frame"
)
_CAMERA_SUBPATH = f"{_COLOR_OPTICAL_SUBPATH}/Camera"
_MOUNT_JOINT_NAME = "RealSenseBottomScrewMountJoint"

_MODEL_SPECIFICATIONS = {
    "d405": {
        "resolution": D405_RESOLUTION,
        "fov_deg": D405_FOV_DEG,
        "clipping_range_m": D405_CLIPPING_RANGE_M,
    },
    "d435": {
        "resolution": D435_RESOLUTION,
        "fov_deg": D435_FOV_DEG,
        "clipping_range_m": D435_CLIPPING_RANGE_M,
    },
}


def _spawn_realsense_from_urdf(
    prim_path: str,
    cfg: sim_utils.UrdfFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: object,
) -> object:
    """Spawn one converted URDF and expose its camera link as a rigid body."""

    from pxr import Usd, UsdGeom, UsdPhysics

    is_d435 = Path(cfg.asset_path).name == "realsense_d435.urdf"
    d435_mesh_usd = _convert_d435_mesh(cfg) if is_d435 else None

    prim = sim_utils.spawn_from_urdf(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    stage = sim_utils.get_current_stage()
    roots = sim_utils.find_matching_prims(prim_path, stage=stage)
    if not roots:
        raise RuntimeError(f"RealSense spawn produced no prims for {prim_path!r}")
    if d435_mesh_usd is not None:
        generated_usd = Path(cfg.usd_dir) / cfg.usd_file_name
        generated_stage = Usd.Stage.Open(generated_usd.as_posix())
        if generated_stage is None:
            raise RuntimeError(f"failed to open converted D435 USD: {generated_usd}")
        generated_visual = generated_stage.GetPrimAtPath(
            f"/{generated_stage.GetDefaultPrim().GetName()}/{_MODEL_SUBPATH}/d435_isaac"
        )
        _reference_d435_mesh(generated_visual, d435_mesh_usd)
        generated_stage.Save()
    for root in roots:
        body_path = f"{root.GetPath()}/{_MODEL_SUBPATH}"
        body = stage.GetPrimAtPath(body_path)
        if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(
                f"RealSense camera_link rigid body is missing: {body_path}"
            )
        if body.HasAPI(UsdPhysics.ArticulationRootAPI) and not body.RemoveAPI(
            UsdPhysics.ArticulationRootAPI
        ):
            raise RuntimeError(
                f"failed to remove standalone articulation API from {body_path}"
            )
        if d435_mesh_usd is not None:
            visual = stage.GetPrimAtPath(f"{body_path}/d435_isaac")
            _reference_d435_mesh(visual, d435_mesh_usd)
            meshes = [
                candidate
                for candidate in Usd.PrimRange(visual, Usd.TraverseInstanceProxies())
                if candidate.IsA(UsdGeom.Mesh)
            ]
            if not meshes:
                raise RuntimeError(
                    f"converted official D435 mesh did not compose below {visual.GetPath()}"
                )
    return prim


def _convert_d435_mesh(cfg: sim_utils.UrdfFileCfg) -> Path:
    """Convert the official DAE with Isaac's converter into ignored build USD."""

    from isaaclab.sim.converters import MeshConverter, MeshConverterCfg

    dae_path = REALSENSE_ASSET_ROOT / "d435" / "meshes" / "d435_isaac.dae"
    if not dae_path.is_file():
        raise FileNotFoundError(f"official D435 DAE does not exist: {dae_path}")
    converter = MeshConverter(
        MeshConverterCfg(
            asset_path=dae_path.as_posix(),
            usd_dir=(Path(cfg.usd_dir) / "mesh").as_posix(),
            usd_file_name="d435_isaac.usd",
            force_usd_conversion=cfg.force_usd_conversion,
            make_instanceable=False,
        )
    )
    output = Path(converter.usd_path).resolve()
    if not output.is_file():
        raise RuntimeError(f"Isaac D435 mesh conversion produced no USD: {output}")
    return output


def _reference_d435_mesh(visual: Any, mesh_usd: Path) -> None:
    """Add the generated official-mesh reference to an importer-created visual."""

    from pxr import Usd, UsdGeom

    if not visual or not visual.IsValid():
        raise RuntimeError("URDF importer did not create the D435 visual frame")
    existing_meshes = [
        candidate
        for candidate in Usd.PrimRange(visual, Usd.TraverseInstanceProxies())
        if candidate.IsA(UsdGeom.Mesh)
    ]
    if existing_meshes:
        return
    if not visual.GetReferences().AddReference(mesh_usd.as_posix()):
        raise RuntimeError(
            f"failed to reference converted official D435 mesh at {visual.GetPath()}"
        )


def _intrinsic_matrix(
    resolution: tuple[int, int], fov_deg: tuple[float, float]
) -> list[float]:
    width, height = resolution
    horizontal, vertical = (math.radians(value) for value in fov_deg)
    focal_x = width / (2.0 * math.tan(horizontal / 2.0))
    focal_y = height / (2.0 * math.tan(vertical / 2.0))
    # Omniverse requires square pixels. The mean minimizes the error across
    # the two nominal vendor FOV values without silently averaging at runtime.
    focal = (focal_x + focal_y) / 2.0
    return [
        focal,
        0.0,
        width / 2.0,
        0.0,
        focal,
        height / 2.0,
        0.0,
        0.0,
        1.0,
    ]


def _make_realsense_cfgs(
    model: str,
    prim_path: str,
    *,
    usd_dir: str | os.PathLike[str] | None,
    spawn_init_pos: tuple[float, float, float],
) -> tuple[AssetBaseCfg, CameraCfg]:
    if not isinstance(prim_path, str) or not prim_path.strip() or prim_path == "/":
        raise ValueError("prim_path must be a non-empty, non-root string")
    root = prim_path.rstrip("/")
    initial_position = finite_vector(spawn_init_pos, length=3, name="spawn_init_pos")
    asset_path = REALSENSE_ASSET_ROOT / model / "urdf" / f"realsense_{model}.urdf"
    if not asset_path.is_file():
        raise FileNotFoundError(
            f"RealSense {model.upper()} URDF does not exist: {asset_path}"
        )
    output_root = (
        REALSENSE_DEFAULT_USD_ROOT / model
        if usd_dir is None
        else Path(usd_dir).expanduser().resolve()
    )
    specification = _MODEL_SPECIFICATIONS[model]
    width, height = specification["resolution"]
    camera_spawn = sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
        intrinsic_matrix=_intrinsic_matrix(
            specification["resolution"], specification["fov_deg"]
        ),
        width=width,
        height=height,
        clipping_range=specification["clipping_range_m"],
    )
    model_cfg = AssetBaseCfg(
        prim_path=root,
        spawn=sim_utils.UrdfFileCfg(
            asset_path=asset_path.as_posix(),
            usd_dir=output_root.as_posix(),
            usd_file_name=f"realsense_{model}.usd",
            fix_base=False,
            merge_fixed_joints=False,
            replace_cylinders_with_capsules=False,
            collision_from_visuals=False,
            self_collision=False,
            make_instanceable=False,
            run_asset_transformer=model == "d405",
            joint_drive=None,
            rigid_props=RigidBodyBaseCfg(disable_gravity=False),
            func=_spawn_realsense_from_urdf,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=initial_position),
    )
    camera_cfg = CameraCfg(
        prim_path=f"{root}/{_CAMERA_SUBPATH}",
        update_period=0.0,
        width=width,
        height=height,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=camera_spawn,
        offset=CameraCfg.OffsetCfg(convention="ros"),
    )
    return model_cfg, camera_cfg


def make_realsense_d405_cfgs(
    prim_path: str,
    *,
    usd_dir: str | os.PathLike[str] | None = None,
    spawn_init_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[AssetBaseCfg, CameraCfg]:
    """Return independent official-model and aligned RGB-D configs for D405."""

    return _make_realsense_cfgs(
        "d405", prim_path, usd_dir=usd_dir, spawn_init_pos=spawn_init_pos
    )


def make_realsense_d435_cfgs(
    prim_path: str,
    *,
    usd_dir: str | os.PathLike[str] | None = None,
    spawn_init_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[AssetBaseCfg, CameraCfg]:
    """Return independent official-model and aligned RGB-D configs for D435."""

    return _make_realsense_cfgs(
        "d435", prim_path, usd_dir=usd_dir, spawn_init_pos=spawn_init_pos
    )


def author_realsense_mount(
    stage: Any,
    parent_body_prim_path: str,
    sensor_prim_path: str,
    *,
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> str:
    """Fix an official bottom-screw frame to a parent-local XYZW pose."""

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    if stage is None:
        raise ValueError("USD stage is required to author a RealSense mount")
    position = finite_vector(pos, length=3, name="pos")
    quaternion_xyzw = normalized_xyzw(rot, name="rot")
    parent = stage.GetPrimAtPath(parent_body_prim_path)
    if not parent or not parent.IsValid():
        raise ValueError(f"parent rigid body does not exist: {parent_body_prim_path}")
    if not parent.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError(f"parent prim is not a rigid body: {parent_body_prim_path}")
    root_path = sensor_prim_path.rstrip("/")
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        raise ValueError(f"RealSense root prim does not exist: {sensor_prim_path}")

    bottom_path = f"{root_path}/{_BOTTOM_SCREW_SUBPATH}"
    body_path = f"{root_path}/{_MODEL_SUBPATH}"
    optical_path = f"{root_path}/{_COLOR_OPTICAL_SUBPATH}"
    bottom = stage.GetPrimAtPath(bottom_path)
    body = stage.GetPrimAtPath(body_path)
    optical = stage.GetPrimAtPath(optical_path)
    if not bottom or not bottom.IsValid():
        raise ValueError(
            f"official camera_bottom_screw_frame is missing: {bottom_path}"
        )
    if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError(f"official camera_link rigid body is missing: {body_path}")
    if not optical or not optical.IsValid():
        raise ValueError(
            f"official camera_color_optical_frame is missing: {optical_path}"
        )
    for prim, label in (
        (parent, "parent rigid body"),
        (root, "RealSense root"),
        (root.GetParent(), "RealSense root parent"),
        (bottom, "camera_bottom_screw_frame"),
        (body, "camera_link"),
    ):
        if not UsdGeom.Xformable(prim):
            raise ValueError(f"{label} is not transformable: {prim.GetPath()}")

    time = Usd.TimeCode.Default()
    parent_world = UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(time)
    root_world = UsdGeom.Xformable(root).ComputeLocalToWorldTransform(time)
    bottom_world = UsdGeom.Xformable(bottom).ComputeLocalToWorldTransform(time)
    body_world = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(time)
    bottom_in_body = bottom_world * body_world.GetInverse()
    body_in_root = body_world * root_world.GetInverse()

    bottom_position, bottom_rotation = matrix_pose(bottom_in_body)
    parent_rotation = Gf.Quatf(
        quaternion_xyzw[3],
        quaternion_xyzw[0],
        quaternion_xyzw[1],
        quaternion_xyzw[2],
    )
    joint_path = Sdf.Path(f"{root_path}/{_MOUNT_JOINT_NAME}")
    existing_joint_path = existing_fixed_joint(
        stage,
        joint_path,
        parent,
        body,
        position,
        parent_rotation,
        bottom_position,
        bottom_rotation,
        label="RealSense mount",
    )

    mount = Gf.Transform()
    mount.SetTranslation(Gf.Vec3d(*position))
    mount.SetRotation(
        Gf.Rotation(
            Gf.Quatd(
                quaternion_xyzw[3],
                quaternion_xyzw[0],
                quaternion_xyzw[1],
                quaternion_xyzw[2],
            )
        )
    )
    desired_bottom_world = mount.GetMatrix() * parent_world
    desired_body_world = bottom_in_body.GetInverse() * desired_bottom_world
    desired_root_world = body_in_root.GetInverse() * desired_body_world
    root_parent_world = UsdGeom.Xformable(
        root.GetParent()
    ).ComputeLocalToWorldTransform(time)
    desired_root_local = desired_root_world * root_parent_world.GetInverse()
    root_position, root_rotation = matrix_pose(desired_root_local)
    if not sim_utils.standardize_xform_ops(
        root,
        translation=root_position,
        orientation=(
            *tuple(root_rotation.GetImaginary()),
            float(root_rotation.GetReal()),
        ),
    ):
        raise ValueError(f"RealSense root is not transformable: {sensor_prim_path}")

    if existing_joint_path is not None:
        return existing_joint_path
    return define_fixed_joint(
        stage,
        joint_path,
        parent,
        body,
        position,
        parent_rotation,
        bottom_position,
        bottom_rotation,
    )


__all__ = [
    "author_realsense_mount",
    "make_realsense_d405_cfgs",
    "make_realsense_d435_cfgs",
]
