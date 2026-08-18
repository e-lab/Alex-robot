#!/usr/bin/env python3
# Copyright (c) 2026, Patrizio Acquadro.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX/TGS CUDA readiness gate for all maintained Alex V2 profiles."""

from __future__ import annotations

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, device="cuda:0")
simulation_app = app_launcher.app

import omni.usd  # noqa: E402
import torch  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.sim.schemas.schemas_cfg import RigidBodyBaseCfg  # noqa: E402
from isaaclab_physx.sim.schemas import (  # noqa: E402
    PhysxArticulationRootPropertiesCfg,
    PhysxCollisionPropertiesCfg,
)

from ihmc_alex_isaaclab.robots.alex_v2 import make_alex_v2_cfg  # noqa: E402
from ihmc_alex_isaaclab.robots.sensor_frames import (  # noqa: E402
    author_alex_v2_sensor_frames,
)
from _physx import (  # noqa: E402
    apply_gravity_compensation,
    assert_finite_joint_state,
    make_readiness_simulation_cfg,
    maximum_contact_force as contact_force,
    require_readiness_gpu,
    rigid_body_path,
    run_gate,
)

DT_S = 0.005
SWEEP_MARGIN = 0.02
SWEEP_SETTLE_STEPS = 360
POSITION_TOLERANCE = 2.0e-4
TRACKING_TOLERANCE_RAD = 0.05
VARIANTS = ("standard", "forearm_convex", "full_convex")
EXPECTED_JOINT_ORDER = (
    "LEFT_HIP_X",
    "RIGHT_HIP_X",
    "SPINE_Z",
    "LEFT_HIP_Z",
    "RIGHT_HIP_Z",
    "NECK_Z",
    "LEFT_SHOULDER_Y",
    "RIGHT_SHOULDER_Y",
    "LEFT_HIP_Y",
    "RIGHT_HIP_Y",
    "NECK_Y",
    "LEFT_SHOULDER_X",
    "RIGHT_SHOULDER_X",
    "LEFT_KNEE_Y",
    "RIGHT_KNEE_Y",
    "LEFT_SHOULDER_Z",
    "RIGHT_SHOULDER_Z",
    "LEFT_ANKLE_Y",
    "RIGHT_ANKLE_Y",
    "LEFT_ELBOW_Y",
    "RIGHT_ELBOW_Y",
    "LEFT_ANKLE_X",
    "RIGHT_ANKLE_X",
    "LEFT_WRIST_Z",
    "RIGHT_WRIST_Z",
    "LEFT_WRIST_X",
    "RIGHT_WRIST_X",
    "LEFT_GRIPPER_Z",
    "RIGHT_GRIPPER_Z",
)
SWEEP_SUPPORT_POSE_RAD = {
    "LEFT_HIP_X": 0.15,
    "RIGHT_HIP_X": -0.15,
    "LEFT_HIP_Y": -0.35,
    "RIGHT_HIP_Y": -0.35,
    "LEFT_KNEE_Y": 0.70,
    "RIGHT_KNEE_Y": 0.70,
    "LEFT_ANKLE_Y": -0.35,
    "RIGHT_ANKLE_Y": -0.35,
    "LEFT_SHOULDER_X": 1.20,
    "RIGHT_SHOULDER_X": -1.20,
    "LEFT_SHOULDER_Y": -0.80,
    "RIGHT_SHOULDER_Y": -0.80,
    "LEFT_ELBOW_Y": -1.50,
    "RIGHT_ELBOW_Y": -1.50,
}


def _step(
    simulation: SimulationContext,
    robots: tuple[Articulation, ...],
    count: int,
    contact_sensor: ContactSensor | None = None,
) -> float:
    maximum_contact_force = 0.0
    for _ in range(count):
        for robot in robots:
            if robot.is_fixed_base:
                apply_gravity_compensation(robot)
            robot.write_data_to_sim()
        simulation.step()
        for robot in robots:
            robot.update(DT_S)
        if contact_sensor is not None:
            contact_sensor.update(DT_S)
            maximum_contact_force = max(
                maximum_contact_force, contact_force(contact_sensor)
            )
    return maximum_contact_force


def _place_external_probe(stage, root_path: str) -> tuple[str, str]:
    foot_path = rigid_body_path(stage, root_path, "LEFT_FOOT")
    sim_utils.activate_contact_sensors(foot_path, stage=stage)
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide],
        useExtentsHint=True,
    )
    bounds = [
        cache.ComputeWorldBound(prim).ComputeAlignedRange()
        for prim in Usd.PrimRange(
            stage.GetPrimAtPath(foot_path), Usd.TraverseInstanceProxies()
        )
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if not bounds:
        raise RuntimeError(f"LEFT_FOOT has no collision shapes: {foot_path}")
    minimum = [
        min(float(bound.GetMin()[axis]) for bound in bounds) for axis in range(3)
    ]
    maximum = [
        max(float(bound.GetMax()[axis]) for bound in bounds) for axis in range(3)
    ]
    center = [(minimum[axis] + maximum[axis]) / 2.0 for axis in range(3)]
    probe_size = 0.020
    center[1] = maximum[1] + probe_size / 2.0 - 0.001
    probe_path = "/World/AlexV2ExternalProbe"
    probe = sim_utils.CuboidCfg(
        size=(probe_size, probe_size, probe_size),
        rigid_props=RigidBodyBaseCfg(
            disable_gravity=True,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.25),
        collision_props=PhysxCollisionPropertiesCfg(collision_enabled=True),
    )
    probe.func(probe_path, probe, translation=tuple(center))
    return foot_path, probe_path


def _assert_articulation_contract(
    stage, root_path: str, *, fixed: bool, expected_self_collision: bool
) -> None:
    articulation_roots = [
        prim
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path))
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    assert len(articulation_roots) == 1, articulation_roots
    actual_self_collision = (
        articulation_roots[0]
        .GetAttribute("physxArticulation:enabledSelfCollisions")
        .Get()
    )
    assert actual_self_collision is expected_self_collision
    fixed_joints = [
        prim
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path))
        if prim.IsA(UsdPhysics.FixedJoint)
    ]
    assert bool(fixed_joints) is fixed, [str(prim.GetPath()) for prim in fixed_joints]


def _support_pose(robot: Articulation) -> torch.Tensor:
    limits = robot.data.joint_pos_limits.torch[0]
    lower_margin = limits[:, 0] + SWEEP_MARGIN * (limits[:, 1] - limits[:, 0])
    upper_margin = limits[:, 1] - SWEEP_MARGIN * (limits[:, 1] - limits[:, 0])
    home = torch.zeros_like(robot.data.joint_pos.torch)
    by_name = {name: index for index, name in enumerate(robot.data.joint_names)}
    for name, value in SWEEP_SUPPORT_POSE_RAD.items():
        home[:, by_name[name]] = value
    home[0] = torch.clamp(home[0], min=lower_margin, max=upper_margin)
    return home


def _sweep_robot(
    simulation: SimulationContext,
    robots: tuple[Articulation, ...],
    robot: Articulation,
) -> tuple[int, float, float]:
    limits = robot.data.joint_pos_limits.torch[0]
    home = _support_pose(robot)
    velocity = torch.zeros_like(robot.data.joint_vel.torch)
    robot.write_joint_state_to_sim_index(position=home, velocity=velocity)
    robot.reset()
    robot.set_joint_position_target_index(target=home)
    _step(simulation, robots, 20)

    completed = 0
    maximum_tracking_error = 0.0
    minimum_observed_movement = float("inf")
    for joint_index, joint_name in enumerate(robot.data.joint_names):
        robot.write_joint_state_to_sim_index(position=home, velocity=velocity)
        robot.reset()
        robot.set_joint_position_target_index(target=home)
        _step(simulation, robots, 20)
        lower, upper = limits[joint_index]
        span = upper - lower
        assert torch.isfinite(span) and float(span) > 0.0, joint_name
        endpoints = (
            float(lower + SWEEP_MARGIN * span),
            float(upper - SWEEP_MARGIN * span),
        )
        observed_positions: list[float] = []
        for endpoint in endpoints:
            target = home.clone()
            target[:, joint_index] = endpoint
            robot.set_joint_position_target_index(target=target)
            _step(simulation, robots, SWEEP_SETTLE_STEPS)
            actual = float(robot.data.joint_pos.torch[0, joint_index])
            observed_positions.append(actual)
            tracking_error = abs(actual - endpoint)
            maximum_tracking_error = max(maximum_tracking_error, tracking_error)
            assert tracking_error <= TRACKING_TOLERANCE_RAD, (
                joint_name,
                endpoint,
                actual,
                tracking_error,
            )
            assert_finite_joint_state(robot, POSITION_TOLERANCE)
            completed += 1
        movement = max(observed_positions) - min(observed_positions)
        expected_movement = min(0.01, 0.01 * float(span))
        assert movement >= expected_movement, (
            joint_name,
            endpoints,
            observed_positions,
            movement,
        )
        minimum_observed_movement = min(minimum_observed_movement, movement)
    assert completed == 2 * len(EXPECTED_JOINT_ORDER)
    return completed, maximum_tracking_error, minimum_observed_movement


def main() -> None:
    require_readiness_gpu()
    simulation = SimulationContext(make_readiness_simulation_cfg(DT_S))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/Ground", ground)

    fixed_robots: dict[str, tuple[Articulation, str]] = {}
    for index, variant in enumerate(VARIANTS):
        root_path = f"/World/AlexV2_{variant}"
        cfg = make_alex_v2_cfg(fix_base=True, variant=variant).replace(
            prim_path=root_path
        )
        cfg.spawn.articulation_props = PhysxArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        )
        cfg.init_state.pos = (float(index * 3), 0.0, 1.5)
        fixed_robots[variant] = (Articulation(cfg), root_path)

    self_collision_path = "/World/AlexV2_self_collision"
    self_collision_cfg = make_alex_v2_cfg(fix_base=True, variant="full_convex").replace(
        prim_path=self_collision_path
    )
    self_collision_cfg.spawn.articulation_props = PhysxArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    )
    self_collision_cfg.init_state.pos = (12.0, 0.0, 1.5)
    self_collision_robot = Articulation(self_collision_cfg)

    floating_path = "/World/AlexV2_floating"
    floating_cfg = make_alex_v2_cfg(fix_base=False, variant="standard").replace(
        prim_path=floating_path
    )
    floating_cfg.spawn.articulation_props = PhysxArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    )
    floating_cfg.init_state.pos = (9.0, 0.0, 1.2)
    floating_robot = Articulation(floating_cfg)
    robots = tuple(robot for robot, _ in fixed_robots.values()) + (
        self_collision_robot,
        floating_robot,
    )

    stage = omni.usd.get_context().get_stage()
    foot_path, probe_path = _place_external_probe(stage, fixed_robots["full_convex"][1])
    foot_contact = ContactSensor(
        ContactSensorCfg(
            prim_path=foot_path,
            filter_prim_paths_expr=[probe_path],
            update_period=0.0,
            history_length=1,
            debug_vis=False,
        )
    )

    simulation.reset()
    self_collision_home = _support_pose(self_collision_robot)
    self_collision_robot.write_joint_state_to_sim_index(
        position=self_collision_home,
        velocity=torch.zeros_like(self_collision_robot.data.joint_vel.torch),
    )
    self_collision_robot.reset()
    self_collision_robot.set_joint_position_target_index(target=self_collision_home)
    initial_floating_root = floating_robot.data.root_pos_w.torch.clone()
    external_contact_force = _step(simulation, robots, 200, foot_contact)
    assert external_contact_force > 0.1, external_contact_force
    sim_utils.modify_collision_properties(
        probe_path,
        PhysxCollisionPropertiesCfg(collision_enabled=False),
        stage=stage,
    )

    floating_displacement = float(
        torch.linalg.vector_norm(
            floating_robot.data.root_pos_w.torch - initial_floating_root, dim=-1
        ).max()
    )
    assert floating_displacement > 1.0e-3
    assert not floating_robot.is_fixed_base
    _assert_articulation_contract(
        stage, floating_path, fixed=False, expected_self_collision=True
    )
    assert self_collision_robot.is_fixed_base
    assert_finite_joint_state(self_collision_robot, POSITION_TOLERANCE)
    _assert_articulation_contract(
        stage, self_collision_path, fixed=True, expected_self_collision=True
    )

    sweep_metrics: dict[str, tuple[int, float, float]] = {}
    for variant, (robot, root_path) in fixed_robots.items():
        assert robot.is_fixed_base
        assert tuple(robot.data.joint_names) == EXPECTED_JOINT_ORDER
        assert robot.data.joint_pos.torch.shape == (1, 29)
        _assert_articulation_contract(
            stage, root_path, fixed=True, expected_self_collision=False
        )
        authored = author_alex_v2_sensor_frames(stage, root_path, variant=variant)
        assert len(authored) == 19
        assert (
            author_alex_v2_sensor_frames(stage, root_path, variant=variant) == authored
        )
        for frame_path in authored.values():
            prim = stage.GetPrimAtPath(frame_path)
            assert prim.IsA(UsdGeom.Xform)
            assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
            assert not prim.HasAPI(UsdPhysics.CollisionAPI)
            assert not prim.HasAPI(UsdPhysics.MassAPI)
        assert_finite_joint_state(robot, POSITION_TOLERANCE)
        sweep_metrics[variant] = _sweep_robot(simulation, robots, robot)

    print(
        "PASS: Alex V2 PhysX/TGS readiness on "
        f"{torch.cuda.get_device_name(0)} (cuda:0); profiles={len(fixed_robots)}, "
        f"joints=29, sensor_frames=19/profile, external_contact={external_contact_force:.3f} N, "
        f"floating_displacement={floating_displacement:.6f} m, sweeps={sweep_metrics}",
        flush=True,
    )


run_gate(main, simulation_app)
