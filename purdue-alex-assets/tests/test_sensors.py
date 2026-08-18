# Copyright (c) 2026, Patrizio Acquadro.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sensor configuration, dependency, and mount contracts."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

import ihmc_alex_isaaclab.sensors.zed_x_mini_dependency as zed_dependency_module
from ihmc_alex_isaaclab.sensors.realsense import (
    D405_CLIPPING_RANGE_M,
    D405_FOV_DEG,
    D405_RESOLUTION,
    D435_CLIPPING_RANGE_M,
    D435_FOV_DEG,
    D435_RESOLUTION,
    REALSENSE_ASSET_ROOT,
    REALSENSE_DEFAULT_USD_ROOT,
    author_realsense_mount,
    make_realsense_d405_cfgs,
    make_realsense_d435_cfgs,
)
from ihmc_alex_isaaclab.sensors.zed_x_mini import (
    ZED_X_MINI_CENTER_OF_MASS_M,
    ZED_X_MINI_DEFAULT_RESOLUTION,
    ZED_X_MINI_DIAGONAL_INERTIA_KG_M2,
    ZED_X_MINI_LENS,
    ZED_X_MINI_MASS_KG,
    ZED_X_MINI_MODEL,
    ZED_X_MINI_RESOLUTIONS,
    author_alex_purdue_zed_x_mini_mount,
    make_zed_x_mini_cfgs,
)
from ihmc_alex_isaaclab.sensors.zed_x_mini_dependency import (
    DEFAULT_ZED_ISAAC_SIM_ROOT,
    load_zed_x_mini_manifest,
    resolve_zed_isaac_sim_root,
    validate_zed_isaac_sim_root,
)


@pytest.mark.parametrize(
    ("model", "factory", "resolution", "fov_deg", "clipping_range_m"),
    (
        (
            "d405",
            make_realsense_d405_cfgs,
            D405_RESOLUTION,
            D405_FOV_DEG,
            D405_CLIPPING_RANGE_M,
        ),
        (
            "d435",
            make_realsense_d435_cfgs,
            D435_RESOLUTION,
            D435_FOV_DEG,
            D435_CLIPPING_RANGE_M,
        ),
    ),
)
def test_factories_lock_separate_urdf_usd_and_aligned_rgbd_contracts(
    model: str,
    factory: object,
    resolution: tuple[int, int],
    fov_deg: tuple[float, float],
    clipping_range_m: tuple[float, float],
    tmp_path: Path,
) -> None:
    output = tmp_path / model
    model_cfg, camera_cfg = factory(
        f"{{ENV_REGEX_NS}}/{model.upper()}",
        usd_dir=output,
        spawn_init_pos=(1, 2.0, 3),
    )
    assert model_cfg.prim_path.endswith(f"/{model.upper()}")
    assert (
        model_cfg.spawn.asset_path
        == (
            REALSENSE_ASSET_ROOT / model / "urdf" / f"realsense_{model}.urdf"
        ).as_posix()
    )
    assert model_cfg.spawn.usd_dir == output.resolve().as_posix()
    assert model_cfg.spawn.usd_file_name == f"realsense_{model}.usd"
    assert model_cfg.spawn.func.__name__ == "_spawn_realsense_from_urdf"
    assert model_cfg.spawn.fix_base is False
    assert model_cfg.spawn.merge_fixed_joints is False
    assert model_cfg.spawn.make_instanceable is False
    assert model_cfg.spawn.run_asset_transformer is (model == "d405")
    assert model_cfg.spawn.collision_from_visuals is False
    assert model_cfg.spawn.self_collision is False
    assert model_cfg.spawn.joint_drive is None
    assert model_cfg.init_state.pos == (1.0, 2.0, 3.0)

    assert camera_cfg.prim_path.endswith(
        "/Geometry/base_link/camera_bottom_screw_frame/camera_link/"
        "camera_color_frame/camera_color_optical_frame/Camera"
    )
    assert (camera_cfg.width, camera_cfg.height) == resolution
    assert camera_cfg.data_types == ["rgb", "distance_to_image_plane"]
    assert camera_cfg.spawn.clipping_range == clipping_range_m
    assert camera_cfg.spawn.projection_type == "pinhole"
    assert camera_cfg.offset.pos == (0.0, 0.0, 0.0)
    assert camera_cfg.offset.rot == (0.0, 0.0, 0.0, 1.0)
    assert camera_cfg.offset.convention == "ros"

    focal_length = camera_cfg.spawn.focal_length
    horizontal_fov = math.degrees(
        2.0 * math.atan(camera_cfg.spawn.horizontal_aperture / (2.0 * focal_length))
    )
    vertical_fov = math.degrees(
        2.0 * math.atan(camera_cfg.spawn.vertical_aperture / (2.0 * focal_length))
    )
    assert horizontal_fov == pytest.approx(fov_deg[0], abs=1.2)
    assert vertical_fov == pytest.approx(fov_deg[1], abs=1.2)


def test_default_outputs_are_model_local_and_factories_return_independent_cfgs() -> (
    None
):
    first_model, first_camera = make_realsense_d405_cfgs("/World/First")
    second_model, second_camera = make_realsense_d405_cfgs("/World/Second")
    d435_model, _ = make_realsense_d435_cfgs("/World/D435")
    assert first_model is not second_model and first_camera is not second_camera
    first_camera.data_types.append("normals")
    assert second_camera.data_types == ["rgb", "distance_to_image_plane"]
    assert first_model.spawn.usd_dir == (REALSENSE_DEFAULT_USD_ROOT / "d405").as_posix()
    assert d435_model.spawn.usd_dir == (REALSENSE_DEFAULT_USD_ROOT / "d435").as_posix()


@pytest.mark.parametrize(
    "factory", (make_realsense_d405_cfgs, make_realsense_d435_cfgs)
)
def test_factories_reject_invalid_inputs(factory: object) -> None:
    with pytest.raises(ValueError, match="prim_path"):
        factory("")
    with pytest.raises(ValueError, match="prim_path"):
        factory("/")
    with pytest.raises(TypeError, match="3-element real vector"):
        factory("/World/Camera", spawn_init_pos=(0.0, 0.0))
    with pytest.raises(ValueError, match="finite"):
        factory("/World/Camera", spawn_init_pos=(0.0, math.nan, 0.0))


def _set_translation(prim: Usd.Prim, xyz: tuple[float, float, float]) -> None:
    UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*xyz))


def _mount_stage(
    camera_link_offset: tuple[float, float, float],
) -> tuple[Usd.Stage, str, str, str]:
    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    robot = UsdGeom.Xform.Define(stage, "/World/Robot").GetPrim()
    parent = UsdGeom.Xform.Define(stage, "/World/Robot/HEAD_LINK").GetPrim()
    _set_translation(parent, (0.6, -0.2, 1.4))
    UsdPhysics.RigidBodyAPI.Apply(parent)
    sensor_parent = UsdGeom.Xform.Define(stage, "/World/Sensors").GetPrim()
    sensor = UsdGeom.Xform.Define(stage, "/World/Sensors/RealSense").GetPrim()
    _set_translation(sensor, (-0.3, 0.5, 0.8))
    geometry = UsdGeom.Xform.Define(
        stage, "/World/Sensors/RealSense/Geometry"
    ).GetPrim()
    base = UsdGeom.Xform.Define(
        stage, "/World/Sensors/RealSense/Geometry/base_link"
    ).GetPrim()
    bottom = UsdGeom.Xform.Define(
        stage,
        "/World/Sensors/RealSense/Geometry/base_link/camera_bottom_screw_frame",
    ).GetPrim()
    body_path = (
        "/World/Sensors/RealSense/Geometry/base_link/"
        "camera_bottom_screw_frame/camera_link"
    )
    body = UsdGeom.Xform.Define(stage, body_path).GetPrim()
    _set_translation(body, camera_link_offset)
    UsdPhysics.RigidBodyAPI.Apply(body)
    color = UsdGeom.Xform.Define(stage, f"{body_path}/camera_color_frame").GetPrim()
    optical = UsdGeom.Xform.Define(
        stage, f"{body_path}/camera_color_frame/camera_color_optical_frame"
    ).GetPrim()
    camera_path = f"{optical.GetPath()}/Camera"
    UsdGeom.Camera.Define(stage, camera_path)
    assert all((world, robot, sensor_parent, geometry, base, bottom, color, optical))
    return stage, parent.GetPath().pathString, sensor.GetPath().pathString, camera_path


def _translation(matrix: Gf.Matrix4d) -> tuple[float, float, float]:
    return tuple(Gf.Transform(matrix).GetTranslation())


@pytest.mark.parametrize(
    "camera_link_offset",
    ((0.01085, 0.009, 0.021), (0.0106, 0.0175, 0.0125)),
)
def test_mount_targets_official_bottom_screw_frame_idempotently(
    camera_link_offset: tuple[float, float, float],
) -> None:
    stage, parent_path, sensor_path, camera_path = _mount_stage(camera_link_offset)
    camera_local_before = UsdGeom.Xformable(
        stage.GetPrimAtPath(camera_path)
    ).GetLocalTransformation()
    position = (0.20, -0.08, 0.03)
    angle = 0.24
    rotation_xyzw = (0.0, math.sin(angle / 2.0), 0.0, math.cos(angle / 2.0))

    joint_path = author_realsense_mount(
        stage,
        parent_path,
        sensor_path,
        pos=position,
        rot=rotation_xyzw,
    )
    assert (
        author_realsense_mount(
            stage,
            parent_path,
            sensor_path,
            pos=position,
            rot=rotation_xyzw,
        )
        == joint_path
    )
    joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath(joint_path))
    body_path = (
        f"{sensor_path}/Geometry/base_link/camera_bottom_screw_frame/camera_link"
    )
    assert joint.GetBody0Rel().GetTargets() == [Sdf.Path(parent_path)]
    assert joint.GetBody1Rel().GetTargets() == [Sdf.Path(body_path)]
    assert tuple(joint.GetLocalPos0Attr().Get()) == pytest.approx(position)
    parent_rotation = joint.GetLocalRot0Attr().Get()
    assert tuple(parent_rotation.GetImaginary()) == pytest.approx(rotation_xyzw[:3])
    assert float(parent_rotation.GetReal()) == pytest.approx(rotation_xyzw[3])
    assert tuple(joint.GetLocalPos1Attr().Get()) == pytest.approx(
        tuple(-value for value in camera_link_offset)
    )

    parent_world = UsdGeom.Xformable(
        stage.GetPrimAtPath(parent_path)
    ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    bottom = stage.GetPrimAtPath(
        f"{sensor_path}/Geometry/base_link/camera_bottom_screw_frame"
    )
    bottom_world = UsdGeom.Xformable(bottom).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    mount = Gf.Transform()
    mount.SetTranslation(Gf.Vec3d(*position))
    mount.SetRotation(Gf.Rotation(Gf.Quatd(rotation_xyzw[3], *rotation_xyzw[:3])))
    expected_bottom_world = mount.GetMatrix() * parent_world
    assert _translation(bottom_world) == pytest.approx(
        _translation(expected_bottom_world), abs=1.0e-8
    )
    assert (
        UsdGeom.Xformable(stage.GetPrimAtPath(camera_path)).GetLocalTransformation()
        == camera_local_before
    )
    assert (
        sum(
            prim.IsA(UsdPhysics.FixedJoint)
            for prim in Usd.PrimRange(stage.GetPrimAtPath(sensor_path))
        )
        == 1
    )


def test_mount_rejects_invalid_frames_pose_and_existing_joint() -> None:
    stage, parent_path, sensor_path, _ = _mount_stage((0.01085, 0.009, 0.021))
    with pytest.raises(ValueError, match="USD stage"):
        author_realsense_mount(None, parent_path, sensor_path)
    with pytest.raises(ValueError, match="parent rigid body does not exist"):
        author_realsense_mount(stage, "/World/Missing", sensor_path)
    with pytest.raises(ValueError, match="RealSense root prim does not exist"):
        author_realsense_mount(stage, parent_path, "/World/Missing")
    with pytest.raises(TypeError, match="3-element real vector"):
        author_realsense_mount(stage, parent_path, sensor_path, pos=(0.0, 0.0))
    with pytest.raises(ValueError, match="finite"):
        author_realsense_mount(
            stage, parent_path, sensor_path, pos=(0.0, math.inf, 0.0)
        )
    with pytest.raises(ValueError, match="normalized XYZW"):
        author_realsense_mount(
            stage, parent_path, sensor_path, rot=(0.0, 0.0, 0.0, 2.0)
        )

    joint_path = author_realsense_mount(stage, parent_path, sensor_path)
    joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath(joint_path))
    joint.GetBody0Rel().SetTargets([Sdf.Path("/World/Other")])
    with pytest.raises(ValueError, match="existing RealSense mount is incompatible"):
        author_realsense_mount(stage, parent_path, sensor_path)


@pytest.mark.parametrize(
    ("resolution", "size"),
    [("SVGA", (960, 600)), ("HD1080", (1920, 1080)), ("HD1200", (1920, 1200))],
)
def test_zed_factory_uses_manifest_contract(
    resolution: str, size: tuple[int, int]
) -> None:
    model, left, right = make_zed_x_mini_cfgs(
        "{ENV_REGEX_NS}/ZED_X_Mini",
        resolution=resolution,
        spawn_init_pos=(1, 2.0, 3),
    )
    manifest = load_zed_x_mini_manifest()
    dependency = validate_zed_isaac_sim_root(DEFAULT_ZED_ISAAC_SIM_ROOT)
    assert ZED_X_MINI_MODEL == manifest["asset"]["model"] == "ZED_XM"
    assert ZED_X_MINI_LENS == manifest["asset"]["lens"] == "Wide"
    assert ZED_X_MINI_MASS_KG == pytest.approx(manifest["asset"]["mass_kg"])
    assert manifest["asset"]["baseline_m"] == pytest.approx(0.050)
    assert ZED_X_MINI_DEFAULT_RESOLUTION == "SVGA"
    assert ZED_X_MINI_RESOLUTIONS[resolution] == size
    assert model.spawn.usd_path == dependency.usd_path.as_posix()
    assert model.spawn.mass_props.mass == pytest.approx(ZED_X_MINI_MASS_KG)
    assert model.init_state.pos == (1.0, 2.0, 3.0)
    assert left.prim_path.endswith("/base_link/ZED_XM/CameraLeft")
    assert right.prim_path.endswith("/base_link/ZED_XM/CameraRight")
    assert (left.width, left.height) == size
    assert (right.width, right.height) == size
    assert left.data_types == ["rgb", "distance_to_image_plane"]
    assert right.data_types == ["rgb"]


def test_zed_factory_returns_independent_cfgs_and_rejects_invalid_inputs(
    tmp_path: Path,
) -> None:
    first = make_zed_x_mini_cfgs("/World/First")
    second = make_zed_x_mini_cfgs("/World/Second")
    assert all(left is not right for left, right in zip(first, second, strict=True))
    first[1].data_types.append("normals")
    assert second[1].data_types == ["rgb", "distance_to_image_plane"]

    with pytest.raises(ValueError, match="prim_path"):
        make_zed_x_mini_cfgs("")
    with pytest.raises(ValueError, match="unsupported ZED X Mini resolution"):
        make_zed_x_mini_cfgs("/World/ZED", resolution="HD720")
    with pytest.raises(TypeError, match="3-element real vector"):
        make_zed_x_mini_cfgs("/World/ZED", spawn_init_pos=(0.0, 0.0))
    with pytest.raises(ValueError, match="finite"):
        make_zed_x_mini_cfgs("/World/ZED", spawn_init_pos=(0.0, math.nan, 0.0))
    with pytest.raises(FileNotFoundError, match="root does not exist"):
        make_zed_x_mini_cfgs("/World/ZED", zed_isaac_sim_root=tmp_path / "missing")


def test_zed_dependency_resolution_precedence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[Path] = []

    def capture(root: str | Path) -> Path:
        seen.append(Path(root))
        return Path(root)

    monkeypatch.setattr(zed_dependency_module, "validate_zed_isaac_sim_root", capture)
    environment_root = tmp_path / "environment"
    explicit_root = tmp_path / "explicit"
    monkeypatch.setenv("ZED_ISAAC_SIM_ROOT", environment_root.as_posix())
    assert resolve_zed_isaac_sim_root(explicit_root) == explicit_root
    assert resolve_zed_isaac_sim_root() == environment_root
    monkeypatch.delenv("ZED_ISAAC_SIM_ROOT")
    assert resolve_zed_isaac_sim_root() == DEFAULT_ZED_ISAAC_SIM_ROOT
    assert seen == [explicit_root, environment_root, DEFAULT_ZED_ISAAC_SIM_ROOT]


def _zed_mount_stage(*, lens: str = "Wide") -> tuple[Usd.Stage, str, str]:
    dependency = validate_zed_isaac_sim_root(DEFAULT_ZED_ISAAC_SIM_ROOT)
    stage = Usd.Stage.CreateInMemory()
    robot_path = "/World/Robot"
    zed_path = "/World/ZED_X_Mini"
    robot = UsdGeom.Xform.Define(stage, robot_path).GetPrim()
    head = UsdGeom.Xform.Define(stage, f"{robot_path}/HEAD_LINK").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(head)
    zed = stage.DefinePrim(zed_path, "Xform")
    assert zed.GetReferences().AddReference(dependency.usd_path.as_posix())
    model = stage.GetPrimAtPath(f"{zed_path}/base_link/ZED_XM")
    model.GetVariantSets().GetVariantSet("lens").SetVariantSelection(lens)
    assert robot and model
    return stage, robot_path, zed_path


def test_zed_mount_is_idempotent_and_preserves_camera_transforms() -> None:
    stage, robot_path, zed_path = _zed_mount_stage()
    camera_paths = (
        f"{zed_path}/base_link/ZED_XM/CameraLeft",
        f"{zed_path}/base_link/ZED_XM/CameraRight",
    )
    before = [
        UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetLocalTransformation()
        for path in camera_paths
    ]

    joint_path = author_alex_purdue_zed_x_mini_mount(stage, robot_path, zed_path)
    assert (
        author_alex_purdue_zed_x_mini_mount(stage, robot_path, zed_path) == joint_path
    )
    joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath(joint_path))
    assert joint.GetBody0Rel().GetTargets() == [Sdf.Path(f"{robot_path}/HEAD_LINK")]
    assert joint.GetBody1Rel().GetTargets() == [Sdf.Path(zed_path)]
    assert tuple(joint.GetLocalPos0Attr().Get()) == pytest.approx(
        (0.1030928129, 0.0096385, -0.0415191024), abs=1.0e-8
    )
    mass = UsdPhysics.MassAPI(stage.GetPrimAtPath(zed_path))
    assert mass.GetMassAttr().Get() == pytest.approx(ZED_X_MINI_MASS_KG)
    assert tuple(mass.GetCenterOfMassAttr().Get()) == pytest.approx(
        ZED_X_MINI_CENTER_OF_MASS_M
    )
    assert tuple(mass.GetDiagonalInertiaAttr().Get()) == pytest.approx(
        ZED_X_MINI_DIAGONAL_INERTIA_KG_M2
    )
    after = [
        UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetLocalTransformation()
        for path in camera_paths
    ]
    assert after == before


def test_zed_mount_rejects_incompatible_stage_content() -> None:
    stage, robot_path, zed_path = _zed_mount_stage()
    with pytest.raises(ValueError, match="robot prim does not exist"):
        author_alex_purdue_zed_x_mini_mount(stage, "/World/Missing", zed_path)
    with pytest.raises(ValueError, match="ZED X Mini prim does not exist"):
        author_alex_purdue_zed_x_mini_mount(stage, robot_path, "/World/Missing")

    narrow_stage, narrow_robot, narrow_zed = _zed_mount_stage(lens="Narrow")
    with pytest.raises(ValueError, match="lens must be 'Wide'"):
        author_alex_purdue_zed_x_mini_mount(narrow_stage, narrow_robot, narrow_zed)

    joint_path = author_alex_purdue_zed_x_mini_mount(stage, robot_path, zed_path)
    joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath(joint_path))
    joint.GetBody0Rel().SetTargets([Sdf.Path("/World/Other")])
    with pytest.raises(ValueError, match="existing ZED mount is incompatible"):
        author_alex_purdue_zed_x_mini_mount(stage, robot_path, zed_path)
