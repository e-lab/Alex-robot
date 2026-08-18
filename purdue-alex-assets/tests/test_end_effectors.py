# Copyright (c) 2026, Patrizio Acquadro.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-effector configuration, command, and mount contracts."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from isaaclab.sim.schemas.schemas_cfg import (
    ArticulationRootPropertiesCfg,
    CollisionPropertiesCfg,
    RigidBodyBaseCfg,
    RigidBodyPropertiesCfg,
)
from isaaclab.sim.spawners.materials.physics_materials_cfg import (
    RigidBodyMaterialBaseCfg,
)
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from ihmc_alex_isaaclab.end_effectors.leap_hand_v1 import (
    LEAP_HAND_V1_ASSET_ROOT,
    LEAP_HAND_V1_DEFAULT_USD_ROOT,
    LEAP_HAND_V1_JOINT_LIMITS,
    author_leap_hand_v1_mount,
    leap_hand_v1_joint_targets,
    make_leap_hand_v1_cfg,
)
from ihmc_alex_isaaclab.end_effectors.weiss_wsg32 import (
    alex_purdue_wsg32_targets,
    make_wsg32_umi_v1_cfg,
)
from ihmc_alex_isaaclab.robots.alex_purdue import make_alex_purdue_cfg
from ihmc_alex_isaaclab.robots.purdue_frames import ALEX_PURDUE_ASSET_ROOT


@pytest.mark.parametrize("side", ("left", "right"))
def test_factory_is_urdf_first_collision_ready_and_independent(
    side: str, tmp_path: Path
) -> None:
    custom_root = tmp_path / side
    first = make_leap_hand_v1_cfg(
        side, f"/World/{side.title()}", fix_base=False, usd_dir=custom_root
    )
    second = make_leap_hand_v1_cfg(side)
    assert first is not second and first.spawn is not second.spawn
    assert first.prim_path == f"/World/{side.title()}"
    assert (
        first.spawn.asset_path
        == (LEAP_HAND_V1_ASSET_ROOT / "urdf" / f"leap_hand_v1_{side}.urdf").as_posix()
    )
    assert first.spawn.usd_dir == custom_root.resolve().as_posix()
    assert first.spawn.usd_file_name == f"leap_hand_v1_{side}.usd"
    assert first.spawn.fix_base is False and second.spawn.fix_base is True
    assert first.spawn.merge_fixed_joints is False
    assert first.spawn.collision_from_visuals is False
    assert first.spawn.collision_type == "Convex Decomposition"
    assert first.spawn.self_collision is True
    assert first.spawn.make_instanceable is False
    assert first.spawn.robot_type == "End Effector"
    assert first.spawn.activate_contact_sensors is True
    assert first.spawn.run_asset_transformer is False
    assert first.spawn.run_multi_physics_conversion is False
    assert first.spawn.func.__name__ == "_spawn_leap_hand_v1"
    assert first.spawn.joint_drive.gains.stiffness == 0.0
    assert first.spawn.joint_drive.gains.damping == 0.0

    assert isinstance(first.spawn.rigid_props, RigidBodyPropertiesCfg)
    assert first.spawn.rigid_props.disable_gravity is False
    assert first.spawn.rigid_props.angular_damping == 0.01
    assert isinstance(first.spawn.collision_props, CollisionPropertiesCfg)
    assert first.spawn.collision_props.contact_offset == 0.002
    assert first.spawn.collision_props.rest_offset == 0.0
    assert isinstance(first.spawn.articulation_props, ArticulationRootPropertiesCfg)
    assert first.spawn.articulation_props.articulation_enabled is True
    assert first.spawn.articulation_props.fix_root_link is False
    assert first.spawn.articulation_props.enabled_self_collisions is True
    assert first.spawn.articulation_props.solver_position_iteration_count == 8
    assert first.spawn.articulation_props.solver_velocity_iteration_count == 4
    assert isinstance(first.spawn.physics_material, RigidBodyMaterialBaseCfg)
    assert first.spawn.physics_material.static_friction == 1.0
    assert first.spawn.physics_material.dynamic_friction == 1.0
    assert first.spawn.physics_material.restitution == 0.0

    actuator = first.actuators["fingers"]
    assert actuator.joint_names_expr == ["a_.*"]
    assert actuator.stiffness == 3.0
    assert actuator.damping == 0.1
    assert actuator.effort_limit == 0.5
    assert actuator.effort_limit_sim == 0.95
    assert actuator.velocity_limit == 8.48
    assert actuator.velocity_limit_sim == 8.48
    assert actuator.armature == 0.001
    assert actuator.friction == 0.01
    first.actuators["fingers"].stiffness = 9.0
    assert second.actuators["fingers"].stiffness == 3.0


def test_default_left_and_right_configs_can_coexist() -> None:
    left = make_leap_hand_v1_cfg("left")
    right = make_leap_hand_v1_cfg("right")
    assert left.prim_path == "{ENV_REGEX_NS}/LeapHandV1Left"
    assert right.prim_path == "{ENV_REGEX_NS}/LeapHandV1Right"
    assert left.prim_path != right.prim_path
    assert left.spawn.asset_path.endswith("leap_hand_v1_left.urdf")
    assert right.spawn.asset_path.endswith("leap_hand_v1_right.urdf")
    assert left.spawn.usd_dir == (LEAP_HAND_V1_DEFAULT_USD_ROOT / "left").as_posix()
    assert right.spawn.usd_dir == (LEAP_HAND_V1_DEFAULT_USD_ROOT / "right").as_posix()


def test_factory_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="side"):
        make_leap_hand_v1_cfg("LEFT")
    with pytest.raises(ValueError, match="prim_path"):
        make_leap_hand_v1_cfg("left", "")
    with pytest.raises(ValueError, match="prim_path"):
        make_leap_hand_v1_cfg("right", "/")
    with pytest.raises(TypeError, match="fix_base"):
        make_leap_hand_v1_cfg("right", fix_base=1)


def test_targets_preserve_motor_id_order_and_side_specific_limits() -> None:
    right_positions = [0.0] * 16
    right_positions[12] = 2.0
    right_positions[13] = 2.4
    targets = leap_hand_v1_joint_targets("right", right_positions)
    assert list(targets) == [f"a_{motor_id}" for motor_id in range(16)]
    assert list(targets.values()) == right_positions

    left_positions = [0.0] * 16
    left_positions[12] = -2.0
    left_positions[13] = -2.4
    left_targets = leap_hand_v1_joint_targets("left", left_positions)
    assert left_targets["a_12"] == -2.0
    assert left_targets["a_13"] == -2.4
    for side in ("left", "right"):
        lower = [limits[0] for limits in LEAP_HAND_V1_JOINT_LIMITS[side]]
        upper = [limits[1] for limits in LEAP_HAND_V1_JOINT_LIMITS[side]]
        assert tuple(leap_hand_v1_joint_targets(side, lower).values()) == tuple(lower)
        assert tuple(leap_hand_v1_joint_targets(side, upper).values()) == tuple(upper)


def test_targets_fail_closed_for_invalid_commands() -> None:
    with pytest.raises(ValueError, match="side"):
        leap_hand_v1_joint_targets("LEFT", [0.0] * 16)
    with pytest.raises(TypeError, match="iterable"):
        leap_hand_v1_joint_targets("left", 0.0)
    with pytest.raises(TypeError, match="iterable"):
        leap_hand_v1_joint_targets("left", "0" * 16)
    with pytest.raises(ValueError, match="exactly 16"):
        leap_hand_v1_joint_targets("left", [0.0] * 15)
    invalid_type = [0.0] * 16
    invalid_type[4] = True
    with pytest.raises(TypeError, match="motor 4"):
        leap_hand_v1_joint_targets("left", invalid_type)
    invalid_finite = [0.0] * 16
    invalid_finite[7] = math.nan
    with pytest.raises(ValueError, match="motor 7.*finite"):
        leap_hand_v1_joint_targets("right", invalid_finite)
    invalid_limit = [0.0] * 16
    invalid_limit[12] = 1.0
    with pytest.raises(ValueError, match="motor 12.*left limits"):
        leap_hand_v1_joint_targets("left", invalid_limit)


def _set_translation(prim: Usd.Prim, xyz: tuple[float, float, float]) -> None:
    UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*xyz))


def _add_hand(
    stage: Usd.Stage, side: str, root_path: str, root_offset: tuple[float, float, float]
) -> tuple[str, str]:
    root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    _set_translation(root, root_offset)
    palm_name = "palm_lower_left" if side == "left" else "palm_lower"
    palm = UsdGeom.Xform.Define(stage, f"{root_path}/{palm_name}").GetPrim()
    _set_translation(palm, (0.03, -0.02, 0.04))
    UsdPhysics.RigidBodyAPI.Apply(palm)
    child = UsdGeom.Xform.Define(stage, f"{root_path}/mcp_joint").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(child)
    revolute = UsdPhysics.RevoluteJoint.Define(stage, f"{root_path}/joints/a_1")
    revolute.CreateBody0Rel().SetTargets([palm.GetPath()])
    revolute.CreateBody1Rel().SetTargets([child.GetPath()])
    return root.GetPath().pathString, palm.GetPath().pathString


def _mount_stage() -> tuple[Usd.Stage, str, str, str, str, str]:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    wrists = UsdGeom.Xform.Define(stage, "/World/Wrists").GetPrim()
    _set_translation(wrists, (0.1, 0.2, 0.3))
    left_wrist = UsdGeom.Xform.Define(
        stage, "/World/Wrists/LEFT_GRIPPER_Z_LINK"
    ).GetPrim()
    right_wrist = UsdGeom.Xform.Define(
        stage, "/World/Wrists/RIGHT_GRIPPER_Y_LINK"
    ).GetPrim()
    _set_translation(left_wrist, (0.4, 0.2, 1.0))
    _set_translation(right_wrist, (0.4, -0.2, 1.0))
    UsdPhysics.RigidBodyAPI.Apply(left_wrist)
    UsdPhysics.RigidBodyAPI.Apply(right_wrist)
    UsdGeom.Xform.Define(stage, "/World/Hands")
    left_root, left_palm = _add_hand(
        stage, "left", "/World/Hands/Left", (-0.2, 0.4, 0.7)
    )
    right_root, right_palm = _add_hand(
        stage, "right", "/World/Hands/Right", (-0.2, -0.4, 0.7)
    )
    return (
        stage,
        left_wrist.GetPath().pathString,
        right_wrist.GetPath().pathString,
        left_root,
        right_root,
        left_palm,
    )


def _transform(prim: Usd.Prim) -> Gf.Matrix4d:
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())


def test_mounts_both_hands_idempotently_at_distinct_required_transforms() -> None:
    stage, left_wrist, right_wrist, left_root, right_root, left_palm = _mount_stage()
    left_pos = (0.12, -0.03, 0.05)
    left_angle = 0.2
    left_rot = (0.0, math.sin(left_angle / 2), 0.0, math.cos(left_angle / 2))
    right_pos = (-0.08, 0.04, 0.02)
    right_angle = -0.3
    right_rot = (
        math.sin(right_angle / 2),
        0.0,
        0.0,
        math.cos(right_angle / 2),
    )

    left_joint_path = author_leap_hand_v1_mount(
        stage, left_wrist, left_root, "left", left_pos, left_rot
    )
    right_joint_path = author_leap_hand_v1_mount(
        stage, right_wrist, right_root, "right", right_pos, right_rot
    )
    assert left_joint_path != right_joint_path
    assert (
        author_leap_hand_v1_mount(
            stage, left_wrist, left_root, "left", left_pos, left_rot
        )
        == left_joint_path
    )
    left_joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath(left_joint_path))
    assert left_joint.GetBody0Rel().GetTargets() == [Sdf.Path(left_wrist)]
    assert left_joint.GetBody1Rel().GetTargets() == [Sdf.Path(left_palm)]
    assert left_joint.GetExcludeFromArticulationAttr().Get() is True
    assert tuple(left_joint.GetLocalPos0Attr().Get()) == pytest.approx(left_pos)
    assert tuple(left_joint.GetLocalPos1Attr().Get()) == pytest.approx((0.0, 0.0, 0.0))
    assert tuple(left_joint.GetLocalRot0Attr().Get().GetImaginary()) == pytest.approx(
        left_rot[:3]
    )
    assert float(left_joint.GetLocalRot0Attr().Get().GetReal()) == pytest.approx(
        left_rot[3]
    )

    mount = Gf.Transform()
    mount.SetTranslation(Gf.Vec3d(*left_pos))
    mount.SetRotation(Gf.Rotation(Gf.Quatd(left_rot[3], *left_rot[:3])))
    expected_palm_world = mount.GetMatrix() * _transform(
        stage.GetPrimAtPath(left_wrist)
    )
    actual_palm_world = _transform(stage.GetPrimAtPath(left_palm))
    assert tuple(Gf.Transform(actual_palm_world).GetTranslation()) == pytest.approx(
        tuple(Gf.Transform(expected_palm_world).GetTranslation()), abs=1.0e-8
    )
    assert sum(prim.IsA(UsdPhysics.FixedJoint) for prim in stage.Traverse()) == 2


def test_mount_accepts_standard_non_transformable_world_scope() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Scope.Define(stage, "/World")
    wrist = UsdGeom.Xform.Define(stage, "/World/Wrist").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(wrist)
    root_path, _ = _add_hand(stage, "right", "/World/RightHand", (0.0, 0.0, 0.0))

    path = author_leap_hand_v1_mount(
        stage,
        "/World/Wrist",
        root_path,
        "right",
        (0.0, 0.0, 0.05),
        (0.0, 0.0, 0.0, 1.0),
    )

    assert stage.GetPrimAtPath(path).IsA(UsdPhysics.FixedJoint)


def test_mount_rejects_invalid_pose_side_frames_and_overconstraint() -> None:
    stage, left_wrist, _, left_root, _, left_palm = _mount_stage()
    identity = (0.0, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="USD stage"):
        author_leap_hand_v1_mount(
            None, left_wrist, left_root, "left", (0.0, 0.0, 0.0), identity
        )
    with pytest.raises(ValueError, match="parent rigid body does not exist"):
        author_leap_hand_v1_mount(
            stage, "/World/Missing", left_root, "left", (0.0, 0.0, 0.0), identity
        )
    with pytest.raises(ValueError, match="root prim does not exist"):
        author_leap_hand_v1_mount(
            stage, left_wrist, "/World/Missing", "left", (0.0, 0.0, 0.0), identity
        )
    with pytest.raises(ValueError, match="side"):
        author_leap_hand_v1_mount(
            stage, left_wrist, left_root, "LEFT", (0.0, 0.0, 0.0), identity
        )
    with pytest.raises(TypeError, match="3-element real vector"):
        author_leap_hand_v1_mount(
            stage, left_wrist, left_root, "left", (0.0, 0.0), identity
        )
    with pytest.raises(ValueError, match="finite"):
        author_leap_hand_v1_mount(
            stage, left_wrist, left_root, "left", (0.0, math.inf, 0.0), identity
        )
    with pytest.raises(ValueError, match="normalized XYZW"):
        author_leap_hand_v1_mount(
            stage,
            left_wrist,
            left_root,
            "left",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 2.0),
        )
    with pytest.raises(ValueError, match="incompatible with side"):
        author_leap_hand_v1_mount(
            stage, left_wrist, left_root, "right", (0.0, 0.0, 0.0), identity
        )
    with pytest.raises(ValueError, match="cannot be inside"):
        author_leap_hand_v1_mount(
            stage,
            f"{left_root}/mcp_joint",
            left_root,
            "left",
            (0.0, 0.0, 0.0),
            identity,
        )

    fixed = UsdPhysics.FixedJoint.Define(stage, "/World/PreFixed")
    fixed.CreateBody1Rel().SetTargets([Sdf.Path(left_palm)])
    with pytest.raises(ValueError, match="fix_base=False"):
        author_leap_hand_v1_mount(
            stage, left_wrist, left_root, "left", (0.0, 0.0, 0.0), identity
        )

    stage, left_wrist, _, left_root, _, left_palm = _mount_stage()
    external_joint = UsdPhysics.PrismaticJoint.Define(stage, "/World/ExternalMount")
    external_joint.CreateBody1Rel().SetTargets([Sdf.Path(left_palm)])
    with pytest.raises(ValueError, match="already constrained"):
        author_leap_hand_v1_mount(
            stage, left_wrist, left_root, "left", (0.0, 0.0, 0.0), identity
        )


def test_mount_rejects_incompatible_repeat_or_reserved_prim() -> None:
    stage, left_wrist, _, left_root, _, _ = _mount_stage()
    identity = (0.0, 0.0, 0.0, 1.0)
    author_leap_hand_v1_mount(
        stage, left_wrist, left_root, "left", (0.0, 0.0, 0.0), identity
    )
    with pytest.raises(ValueError, match="existing.*incompatible"):
        author_leap_hand_v1_mount(
            stage, left_wrist, left_root, "left", (0.01, 0.0, 0.0), identity
        )

    second_stage, second_wrist, _, second_root, _, _ = _mount_stage()
    UsdGeom.Xform.Define(second_stage, f"{second_root}/LeapHandV1WristMountJoint")
    with pytest.raises(ValueError, match="not a fixed joint"):
        author_leap_hand_v1_mount(
            second_stage,
            second_wrist,
            second_root,
            "left",
            (0.0, 0.0, 0.0),
            identity,
        )


def test_wsg_profile_is_opt_in_and_selects_variant() -> None:
    default = make_alex_purdue_cfg()
    assert default.spawn.asset_path.endswith("alex_purdue_full_convex.urdf")
    assert set(default.actuators) == {"neck", "arms", "ezgrippers"}

    source = make_alex_purdue_cfg(
        variant="source", end_effector="wsg32_umi_v1", fix_base=False
    )
    convex = make_alex_purdue_cfg(end_effector="wsg32_umi_v1")
    assert source is not convex and source.spawn is not convex.spawn
    assert (
        source.spawn.asset_path
        == (
            ALEX_PURDUE_ASSET_ROOT
            / "urdf"
            / "baseline"
            / "alex_purdue_wsg32_umi_v1.urdf"
        ).as_posix()
    )
    assert convex.spawn.asset_path.endswith("alex_purdue_wsg32_umi_v1_full_convex.urdf")
    assert source.spawn.fix_base is False
    assert source.spawn.self_collision is False
    assert convex.spawn.self_collision is True
    assert set(source.actuators) == {"neck", "arms", "wsg32"}
    assert source.actuators["wsg32"].joint_names_expr == [".*_WSG32_JAW_OPENING"]


def test_wsg_factory_is_collision_ready_and_independent() -> None:
    first = make_wsg32_umi_v1_cfg(fix_base=False)
    second = make_wsg32_umi_v1_cfg()
    assert first is not second and first.spawn is not second.spawn
    assert first.spawn.asset_path.endswith("wsg32_umi_v1.urdf")
    assert first.spawn.fix_base is False and second.spawn.fix_base is True
    assert first.spawn.merge_fixed_joints is True
    assert first.spawn.collision_from_visuals is False
    assert first.spawn.func.__name__ == "spawn_wsg32_umi_v1"
    assert isinstance(first.spawn.rigid_props, RigidBodyBaseCfg)
    assert set(first.actuators) == {"opening"}
    assert first.actuators["opening"].effort_limit == 25.0


def test_wsg_normalized_targets_and_failures() -> None:
    assert alex_purdue_wsg32_targets(0.0, "left") == {"left_WSG32_JAW_OPENING": 0.0}
    assert alex_purdue_wsg32_targets(1.0, "right") == {"right_WSG32_JAW_OPENING": 0.034}
    assert tuple(alex_purdue_wsg32_targets(0.5, "left").values()) == pytest.approx(
        (0.017,)
    )
    for invalid in (-0.01, 1.01, math.nan, math.inf):
        with pytest.raises(ValueError, match=r"within \[0, 1\]"):
            alex_purdue_wsg32_targets(invalid, "left")
    for invalid in (True, "0.5"):
        with pytest.raises(TypeError, match="real scalar"):
            alex_purdue_wsg32_targets(invalid, "left")
    with pytest.raises(ValueError, match="side"):
        alex_purdue_wsg32_targets(0.5, "LEFT")


def test_wsg_factories_reject_invalid_inputs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="qualified WSG32 URDF"):
        make_wsg32_umi_v1_cfg(tmp_path / "missing.urdf")
    with pytest.raises(ValueError, match="unknown Alex Purdue end effector"):
        make_alex_purdue_cfg(end_effector="unknown")
    with pytest.raises(ValueError, match="mutually exclusive"):
        make_alex_purdue_cfg(tmp_path / "missing.urdf", end_effector="wsg32_umi_v1")
