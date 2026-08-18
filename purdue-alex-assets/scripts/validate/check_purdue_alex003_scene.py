#!/usr/bin/env python3
# Copyright (c) 2026, Patrizio Acquadro.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX/TGS CUDA readiness gate for the Purdue Alex003 reference scene."""

from __future__ import annotations

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, device="cuda:0")
simulation_app = app_launcher.app

import omni.usd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab_physx.sim.schemas import (  # noqa: E402
    PhysxArticulationRootPropertiesCfg,
    PhysxCollisionPropertiesCfg,
    PhysxRigidBodyPropertiesCfg,
)

from ihmc_alex_isaaclab.platforms.purdue_alex003_pedestal import (  # noqa: E402
    PURDUE_ALEX003_PEDESTAL_ASSET_ROOT,
    load_purdue_alex003_pedestal_spec,
)
from ihmc_alex_isaaclab.scenes.purdue_alex003 import (  # noqa: E402
    make_purdue_alex003_reference_scene_cfg,
)
from ihmc_alex_isaaclab.robots.purdue_physics import (  # noqa: E402
    ALEX_PURDUE_MIMIC_JOINT_PAIRS,
)
from _physx import (  # noqa: E402
    apply_gravity_compensation,
    assert_finite_joint_state,
    make_readiness_simulation_cfg,
    maximum_contact_force,
    require_readiness_gpu,
    rigid_body_path,
    run_gate,
)

DT_S = 0.005
PROBE_SIZE_M = 0.010
PROBE_Y_M = 0.120


def _step(
    simulation: SimulationContext,
    scene: InteractiveScene,
    contacts: tuple[ContactSensor, ...],
    count: int,
) -> tuple[float, ...]:
    maxima = [0.0] * len(contacts)
    follower_names = {follower for follower, _ in ALEX_PURDUE_MIMIC_JOINT_PAIRS}
    for _ in range(count):
        robot = scene["robot"]
        apply_gravity_compensation(robot, excluded_joint_names=follower_names)
        scene.write_data_to_sim()
        simulation.step()
        scene.update(DT_S)
        for index, contact in enumerate(contacts):
            contact.update(DT_S)
            maxima[index] = max(maxima[index], maximum_contact_force(contact))
    return tuple(maxima)


def _robot_probe_translation(stage, robot_path: str) -> tuple[float, float, float]:
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide],
        useExtentsHint=True,
    )
    bounds = [
        cache.ComputeWorldBound(prim).ComputeAlignedRange()
        for prim in Usd.PrimRange(
            stage.GetPrimAtPath(robot_path), Usd.TraverseInstanceProxies()
        )
        if prim.HasAPI(UsdPhysics.CollisionAPI)
        and "/TORSO_LINK_CONVEX/" in str(prim.GetPath())
    ]
    if not bounds:
        raise RuntimeError("Alex Purdue torso collider was not found")
    minimum = [
        min(float(bound.GetMin()[axis]) for bound in bounds) for axis in range(3)
    ]
    maximum = [
        max(float(bound.GetMax()[axis]) for bound in bounds) for axis in range(3)
    ]
    center = [(minimum[axis] + maximum[axis]) / 2.0 for axis in range(3)]
    center[0] = maximum[0] - 0.020
    return tuple(center)


def _world_matrix(stage, prim_path: str) -> tuple[float, ...]:
    prim = stage.GetPrimAtPath(prim_path)
    assert prim.IsA(UsdGeom.Xform)
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    return tuple(float(matrix[row][column]) for row in range(4) for column in range(4))


def _validate_fixed_base_geometry(stage, pedestal_path: str) -> None:
    spec = load_purdue_alex003_pedestal_spec()
    pedestal = stage.GetPrimAtPath(pedestal_path)
    assert pedestal.IsA(UsdGeom.Xform)
    descendants = tuple(Usd.PrimRange(pedestal))
    rigid_bodies = tuple(
        prim for prim in descendants if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    )
    joints = tuple(prim for prim in descendants if prim.IsA(UsdPhysics.Joint))
    fixed_joints = tuple(prim for prim in joints if prim.IsA(UsdPhysics.FixedJoint))
    link_path = f"{pedestal_path}/Geometry/PURDUE_ALEX003_PEDESTAL"
    link = stage.GetPrimAtPath(link_path)
    assert tuple(str(prim.GetPath()) for prim in rigid_bodies) == (link_path,), (
        "unexpected pedestal rigid bodies: "
        f"{tuple(str(prim.GetPath()) for prim in rigid_bodies)}"
    )
    assert len(joints) == len(fixed_joints) == 1
    assert fixed_joints[0].GetName() == "root_joint"
    fixed_joint = UsdPhysics.FixedJoint(fixed_joints[0])
    assert tuple(str(path) for path in fixed_joint.GetBody0Rel().GetTargets()) == (
        pedestal_path,
    )
    assert tuple(str(path) for path in fixed_joint.GetBody1Rel().GetTargets()) == (
        link_path,
    )
    assert not pedestal.HasAPI(UsdPhysics.RigidBodyAPI)

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.guide],
        False,
        True,
    )
    collision_prims = tuple(
        prim for prim in descendants if prim.HasAPI(UsdPhysics.CollisionAPI)
    )
    assert tuple(prim.GetName() for prim in collision_prims) == tuple(
        part.name for part in spec.parts
    )
    for part in spec.parts:
        collider = stage.GetPrimAtPath(f"{link_path}/{part.name}")
        assert collider.IsA(UsdGeom.Cube)
        assert collider.HasAPI(UsdPhysics.CollisionAPI)
        assert not collider.HasAPI(UsdPhysics.RigidBodyAPI)
        assert (
            UsdGeom.Imageable(collider).GetPurposeAttr().Get() == UsdGeom.Tokens.guide
        )
        aligned_range = bbox_cache.ComputeWorldBound(collider).ComputeAlignedRange()
        minimum = aligned_range.GetMin()
        maximum = aligned_range.GetMax()
        size = tuple(float(maximum[index] - minimum[index]) for index in range(3))
        center = tuple(
            float((maximum[index] + minimum[index]) / 2.0) for index in range(3)
        )
        _assert_close(size, part.size_xyz_m, absolute=1.0e-6)
        _assert_close(center, part.center_xyz_m, absolute=1.0e-6)

    visual_prims = tuple(
        prim
        for prim in Usd.PrimRange(link)
        if prim.IsA(UsdGeom.Gprim) and not prim.HasAPI(UsdPhysics.CollisionAPI)
    )
    assert len(visual_prims) == 39
    assert not any(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in visual_prims)
    assert not any(prim.HasAPI(UsdPhysics.MassAPI) for prim in visual_prims)

    visible_bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_],
    )
    assert not stage.GetPrimAtPath(f"{link_path}/robot_mount_adapter").IsValid()
    assert not stage.GetPrimAtPath(f"{link_path}/robot_mount_collar").IsValid()
    upper_plate = stage.GetPrimAtPath(f"{link_path}/upper_mount_plate")
    assert upper_plate.IsA(UsdGeom.Cube)
    upper_plate_range = visible_bbox_cache.ComputeWorldBound(
        upper_plate
    ).ComputeAlignedRange()
    mount_dimensions = yaml.safe_load(
        (
            PURDUE_ALEX003_PEDESTAL_ASSET_ROOT.parent
            / "purdue_pedestal"
            / "dimensions.yaml"
        ).read_text(encoding="utf-8")
    )
    robot_mount_min_z = spec.alex_root_world_z_m + float(
        mount_dimensions["minimum_xyz"][2]
    )
    _assert_close(
        (float(upper_plate_range.GetMax()[2]),),
        (spec.mounting_plane_world_z_m,),
        absolute=1.0e-6,
    )
    assert abs(robot_mount_min_z - spec.mounting_plane_world_z_m) <= 1.0e-3


def _assert_close(
    actual: tuple[float, ...],
    expected: tuple[float, ...],
    *,
    absolute: float,
) -> None:
    """Assert finite element-wise equality within one absolute tolerance."""

    if not bool(torch.isfinite(torch.tensor(actual)).all()):
        raise AssertionError(f"non-finite values: {actual}")
    if len(actual) != len(expected) or any(
        abs(left - right) > absolute
        for left, right in zip(actual, expected, strict=True)
    ):
        raise AssertionError(
            f"actual={actual}, expected={expected}, tolerance={absolute}"
        )


def main() -> None:
    require_readiness_gpu()
    spec = load_purdue_alex003_pedestal_spec()
    simulation = SimulationContext(make_readiness_simulation_cfg(DT_S))

    scene_cfg = make_purdue_alex003_reference_scene_cfg()
    scene_cfg.robot.spawn.articulation_props = PhysxArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    )
    probe_rest_z = spec.mounting_plane_world_z_m + PROBE_SIZE_M / 2.0
    scene_cfg.probe = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PedestalProbe",
        spawn=sim_utils.CuboidCfg(
            size=(PROBE_SIZE_M, PROBE_SIZE_M, PROBE_SIZE_M),
            rigid_props=PhysxRigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
            collision_props=PhysxCollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.75, 0.12, 0.08),
                roughness=0.45,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, PROBE_Y_M, probe_rest_z + 0.025),
        ),
    )
    scene_cfg.robot_probe = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RobotProbe",
        spawn=sim_utils.CuboidCfg(
            size=(PROBE_SIZE_M, PROBE_SIZE_M, PROBE_SIZE_M),
            rigid_props=PhysxRigidBodyPropertiesCfg(
                disable_gravity=True,
                linear_damping=20.0,
                angular_damping=20.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=PhysxCollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.12, 0.25, 0.8),
                roughness=0.45,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5, 0.0, 1.0)),
    )
    scene = InteractiveScene(scene_cfg)
    robot = scene["robot"]
    probe = scene["probe"]
    robot_probe = scene["robot_probe"]

    stage = omni.usd.get_context().get_stage()
    pedestal_path = "/World/envs/env_0/PurduePedestal"
    pedestal_link_path = f"{pedestal_path}/Geometry/PURDUE_ALEX003_PEDESTAL"
    robot_path = "/World/envs/env_0/Robot"
    torso_path = rigid_body_path(
        stage, robot_path, "PEDESTAL_LINK", traverse_instances=True
    )
    sim_utils.activate_contact_sensors(pedestal_link_path, stage=stage)
    sim_utils.activate_contact_sensors(torso_path, stage=stage)
    contacts = (
        ContactSensor(
            ContactSensorCfg(
                prim_path=pedestal_link_path,
                filter_prim_paths_expr=["/World/envs/env_0/PedestalProbe"],
                update_period=0.0,
                history_length=1,
                debug_vis=False,
            )
        ),
        ContactSensor(
            ContactSensorCfg(
                prim_path=torso_path,
                filter_prim_paths_expr=["/World/envs/env_0/RobotProbe"],
                update_period=0.0,
                history_length=1,
                debug_vis=False,
            )
        ),
    )

    simulation.reset()
    initial_pedestal_matrix = _world_matrix(stage, pedestal_link_path)
    initial_root = robot.data.root_pos_w.torch.clone()
    pedestal_force, initial_robot_force = _step(simulation, scene, contacts, 300)

    _validate_fixed_base_geometry(stage, pedestal_path)
    _assert_close(
        _world_matrix(stage, pedestal_link_path),
        initial_pedestal_matrix,
        absolute=1.0e-9,
    )
    assert torch.isfinite(robot.data.root_pos_w.torch).all()
    assert torch.isfinite(robot.data.body_pos_w.torch).all()
    assert_finite_joint_state(robot, 2.0e-4)
    assert (
        float(torch.max(torch.abs(robot.data.root_pos_w.torch - initial_root)))
        <= 1.0e-6
    )
    assert (
        abs(float(robot.data.root_pos_w.torch[0, 2]) - spec.alex_root_world_z_m)
        <= 1.0e-6
    )

    shoulder_index = robot.data.body_names.index("RIGHT_SHOULDER_Y_LINK")
    shoulder_world_z = float(robot.data.body_pos_w.torch[0, shoulder_index, 2])
    assert abs(shoulder_world_z - spec.right_shoulder_y_world_z_m) <= 1.0e-5

    assert initial_robot_force <= 1.0
    probe_world_z = float(probe.data.root_pos_w.torch[0, 2])
    assert abs(probe_world_z - probe_rest_z) <= 0.003
    assert float(torch.linalg.vector_norm(probe.data.root_lin_vel_w.torch[0])) <= 0.02
    assert pedestal_force > 0.1, pedestal_force

    robot_probe_pose = robot_probe.data.root_pose_w.torch.clone()
    robot_probe_pose[:, :3] = torch.tensor(
        _robot_probe_translation(stage, robot_path),
        device=robot_probe_pose.device,
    )
    robot_probe.write_root_pose_to_sim_index(root_pose=robot_probe_pose)
    robot_probe.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros_like(robot_probe.data.root_vel_w.torch)
    )
    _, robot_force = _step(simulation, scene, contacts, 600)
    robot_probe_actual = robot_probe.data.root_pos_w.torch.clone()
    assert robot_force > 0.1, (robot_probe_pose[:, :3], robot_probe_actual, robot_force)
    assert torch.isfinite(robot_probe.data.root_pos_w.torch).all()
    assert torch.isfinite(robot_probe.data.root_lin_vel_w.torch).all()
    robot_probe_speed = float(
        torch.linalg.vector_norm(robot_probe.data.root_lin_vel_w.torch[0])
    )
    assert robot_probe_speed <= 0.02, robot_probe_speed
    assert all(
        torch.isfinite(contact.data.net_forces_w.torch).all() for contact in contacts
    )
    print(
        "PASS: Purdue Alex003 pedestal/reference scene PhysX/TGS readiness; "
        f"device={torch.cuda.get_device_name(0)!r} "
        f"root_z={float(robot.data.root_pos_w.torch[0, 2]):.8f} "
        f"right_shoulder_y_z={shoulder_world_z:.8f} "
        f"probe_z={probe_world_z:.8f} "
        f"pedestal_contact={pedestal_force:.6f} N "
        f"robot_contact={robot_force:.6f} N "
        f"rigid_bodies=1 dof=0 fixed_joints=1",
        flush=True,
    )


run_gate(main, simulation_app)
