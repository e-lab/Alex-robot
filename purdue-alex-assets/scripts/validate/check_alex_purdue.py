#!/usr/bin/env python3
# Copyright (c) 2026, Patrizio Acquadro.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX/TGS CUDA readiness gate for Alex Purdue with dual SAKE grippers."""

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

from ihmc_alex_isaaclab.end_effectors.sake_ezgripper import (  # noqa: E402
    alex_purdue_ezgripper_targets,
)
from ihmc_alex_isaaclab.robots.alex_purdue import make_alex_purdue_cfg  # noqa: E402
from ihmc_alex_isaaclab.robots.purdue_frames import (  # noqa: E402
    author_alex_purdue_frames,
)
from ihmc_alex_isaaclab.robots.purdue_physics import (  # noqa: E402
    ALEX_PURDUE_MIMIC_JOINT_PAIRS,
    ALEX_PURDUE_SELF_COLLISION_FILTER_PAIRS,
)
from _physx import (  # noqa: E402
    apply_gravity_compensation,
    assert_finite_joint_state,
    make_readiness_simulation_cfg,
    maximum_contact_force,
    require_readiness_gpu,
    rigid_body_path,
    run_gate,
    set_joint_targets,
)

DT_S = 0.005
TRACKING_TOLERANCE_RAD = 0.05
MIMIC_TOLERANCE_RAD = 1.0e-4
LIMIT_TOLERANCE_RAD = 2.0e-4
SWEEP_MARGIN = 0.02
SWEEP_SETTLE_STEPS = 360
SWEEP_SUPPORT_POSE_RAD = {
    "LEFT_SHOULDER_X": 1.2,
    "LEFT_SHOULDER_Y": -0.8,
    "LEFT_SHOULDER_Z": 0.0,
    "LEFT_ELBOW_Y": -1.5,
    "RIGHT_SHOULDER_X": -1.2,
    "RIGHT_SHOULDER_Y": -0.8,
    "RIGHT_SHOULDER_Z": 0.0,
    "RIGHT_ELBOW_Y": -1.5,
}


def _step(
    simulation: SimulationContext,
    robots: tuple[Articulation, ...],
    sensors: tuple[ContactSensor, ...],
    count: int,
) -> tuple[float, ...]:
    maxima = [0.0] * len(sensors)
    follower_names = {follower for follower, _ in ALEX_PURDUE_MIMIC_JOINT_PAIRS}
    for _ in range(count):
        for robot in robots:
            apply_gravity_compensation(robot, excluded_joint_names=follower_names)
            robot.write_data_to_sim()
        simulation.step()
        for robot in robots:
            robot.update(DT_S)
        for index, sensor in enumerate(sensors):
            sensor.update(DT_S)
            maxima[index] = max(maxima[index], maximum_contact_force(sensor))
    return tuple(maxima)


def _world_translation(stage, prim_path: str) -> tuple[float, float, float]:
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(
        stage.GetPrimAtPath(prim_path)
    )
    value = matrix.ExtractTranslation()
    return tuple(float(value[index]) for index in range(3))


def _probe_translation(
    stage, body_path: str, tcp: tuple[float, float, float]
) -> tuple[float, float, float]:
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide],
        useExtentsHint=True,
    )
    bounds = [
        cache.ComputeWorldBound(prim).ComputeAlignedRange()
        for prim in Usd.PrimRange(
            stage.GetPrimAtPath(body_path), Usd.TraverseInstanceProxies()
        )
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    assert bounds, body_path
    minimum = [
        min(float(bound.GetMin()[axis]) for bound in bounds) for axis in range(3)
    ]
    maximum = [
        max(float(bound.GetMax()[axis]) for bound in bounds) for axis in range(3)
    ]
    center = [(minimum[axis] + maximum[axis]) / 2.0 for axis in range(3)]
    outward_axis = max(range(2), key=lambda axis: abs(center[axis] - tcp[axis]))
    direction = 1.0 if center[outward_axis] >= tcp[outward_axis] else -1.0
    center[outward_axis] = (
        maximum[outward_axis] - 0.004
        if direction > 0
        else minimum[outward_axis] + 0.004
    )
    return tuple(center)


def _spawn_contact_probe(path: str, translation: tuple[float, float, float]) -> None:
    cfg = sim_utils.CuboidCfg(
        size=(0.010, 0.010, 0.010),
        rigid_props=RigidBodyBaseCfg(disable_gravity=True, kinematic_enabled=True),
        collision_props=PhysxCollisionPropertiesCfg(collision_enabled=True),
    )
    cfg.func(path, cfg, translation=translation)


def _mimic_error(robot: Articulation) -> float:
    by_name = {name: index for index, name in enumerate(robot.data.joint_names)}
    assert all(
        follower in by_name and leader in by_name
        for follower, leader in ALEX_PURDUE_MIMIC_JOINT_PAIRS
    )
    return max(
        float(
            torch.abs(
                robot.data.joint_pos.torch[:, by_name[follower]]
                - robot.data.joint_pos.torch[:, by_name[leader]]
            ).max()
        )
        for follower, leader in ALEX_PURDUE_MIMIC_JOINT_PAIRS
    )


def _mimic_errors(robot: Articulation) -> dict[str, float]:
    by_name = {name: index for index, name in enumerate(robot.data.joint_names)}
    return {
        f"{follower}->{leader}": float(
            torch.abs(
                robot.data.joint_pos.torch[:, by_name[follower]]
                - robot.data.joint_pos.torch[:, by_name[leader]]
            ).max()
        )
        for follower, leader in ALEX_PURDUE_MIMIC_JOINT_PAIRS
    }


def _assert_frames(stage, root_path: str, variant: str) -> dict[str, str]:
    authored = author_alex_purdue_frames(stage, root_path, variant=variant)
    assert len(authored) == 15
    assert author_alex_purdue_frames(stage, root_path, variant=variant) == authored
    for frame_path in authored.values():
        prim = stage.GetPrimAtPath(frame_path)
        assert prim.IsA(UsdGeom.Xform)
        assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
        assert not prim.HasAPI(UsdPhysics.CollisionAPI)
    return authored


def _assert_filter_contract(stage, root_path: str) -> int:
    authored: list[tuple[str, str]] = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        relationship = prim.GetRelationship("physics:filteredPairs")
        if relationship:
            authored.extend(
                (str(prim.GetPath()), str(target))
                for target in relationship.GetTargets()
            )
    assert len(ALEX_PURDUE_SELF_COLLISION_FILTER_PAIRS) == 3
    assert len(authored) == 5
    for first, second in ALEX_PURDUE_SELF_COLLISION_FILTER_PAIRS:
        assert any(first in left and second in right for left, right in authored), (
            first,
            second,
            authored,
        )
    return len(authored)


def _tracking_error(robot: Articulation, target: torch.Tensor, index: int) -> float:
    return float(
        torch.abs(robot.data.joint_pos.torch[:, index] - target[:, index]).max()
    )


def main() -> None:
    require_readiness_gpu()
    simulation = SimulationContext(make_readiness_simulation_cfg(DT_S))

    source_cfg = make_alex_purdue_cfg(variant="source").replace(
        prim_path="/World/AlexPurdueSource"
    )
    source_cfg.init_state.pos = (-3.0, 0.0, 0.0)
    source_cfg.spawn.articulation_props = PhysxArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        solver_position_iteration_count=64,
        solver_velocity_iteration_count=8,
    )
    convex_cfg = make_alex_purdue_cfg(variant="full_convex").replace(
        prim_path="/World/AlexPurdueConvex"
    )
    convex_cfg.spawn.articulation_props = PhysxArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=64,
        solver_velocity_iteration_count=8,
    )
    source_robot = Articulation(source_cfg)
    convex_robot = Articulation(convex_cfg)
    robots = (source_robot, convex_robot)

    stage = omni.usd.get_context().get_stage()
    source_frames = _assert_frames(stage, source_cfg.prim_path, "source")
    convex_frames = _assert_frames(stage, convex_cfg.prim_path, "full_convex")
    assert source_frames.keys() == convex_frames.keys()
    filtered_shape_pairs = _assert_filter_contract(stage, convex_cfg.prim_path)

    tcp_positions = {
        side: _world_translation(stage, convex_frames[f"{side}_EZGRIPPER_TCP_FRAME"])
        for side in ("LEFT", "RIGHT")
    }
    palm_positions = {
        side: _world_translation(stage, convex_frames[f"{side}_EZGRIPPER_PALM_FRAME"])
        for side in ("LEFT", "RIGHT")
    }
    palm_tcp = {
        side: sum(
            (tcp_positions[side][axis] - palm_positions[side][axis]) ** 2
            for axis in range(3)
        )
        ** 0.5
        for side in ("LEFT", "RIGHT")
    }
    assert all(abs(distance - 0.160804) <= 1.0e-6 for distance in palm_tcp.values())

    contact_bodies: dict[str, str] = {}
    object_paths: dict[str, str] = {}
    for side in ("left", "right"):
        body = rigid_body_path(
            stage, convex_cfg.prim_path, f"{side}_ezgripper_finger_l2_1"
        )
        object_path = f"/World/{side.title()}SAKEContactObject"
        sim_utils.activate_contact_sensors(body, stage=stage)
        _spawn_contact_probe(
            object_path,
            _probe_translation(stage, body, tcp_positions[side.upper()]),
        )
        contact_bodies[side] = body
        object_paths[side] = object_path

    right_peer = rigid_body_path(
        stage, convex_cfg.prim_path, "right_ezgripper_finger_l2_2"
    )
    sensors = (
        ContactSensor(
            ContactSensorCfg(
                prim_path=contact_bodies["left"],
                filter_prim_paths_expr=[object_paths["left"]],
                update_period=0.0,
            )
        ),
        ContactSensor(
            ContactSensorCfg(
                prim_path=contact_bodies["right"],
                filter_prim_paths_expr=[object_paths["right"]],
                update_period=0.0,
            )
        ),
        ContactSensor(
            ContactSensorCfg(
                prim_path=contact_bodies["right"],
                filter_prim_paths_expr=[right_peer],
                update_period=0.0,
            )
        ),
    )

    simulation.reset()
    for robot in robots:
        assert robot.is_fixed_base
        assert robot.data.joint_pos.torch.shape == (1, 24)
        zero = torch.zeros_like(robot.data.joint_pos.torch)
        robot.write_joint_state_to_sim_index(position=zero, velocity=zero)
        robot.reset()
        robot.set_joint_position_target_index(target=zero)
    initial_roots = [robot.data.root_pos_w.torch.clone() for robot in robots]
    _step(simulation, robots, sensors, 400)

    maximum_hold_error = 0.0
    for robot, initial_root in zip(robots, initial_roots, strict=True):
        assert_finite_joint_state(robot, LIMIT_TOLERANCE_RAD)
        maximum_hold_error = max(
            maximum_hold_error, float(torch.abs(robot.data.joint_pos.torch).max())
        )
        assert float(torch.abs(robot.data.joint_pos.torch).max()) <= 0.02
        assert (
            float(torch.abs(robot.data.root_pos_w.torch - initial_root).max()) <= 1.0e-6
        )

    joint_names = tuple(convex_robot.data.joint_names)
    assert len(joint_names) == 24
    follower_names = {follower for follower, _ in ALEX_PURDUE_MIMIC_JOINT_PAIRS}
    commanded_names = [name for name in joint_names if name not in follower_names]
    assert len(commanded_names) == 20
    by_name = {name: index for index, name in enumerate(joint_names)}
    limits = convex_robot.data.joint_pos_limits.torch[0]
    maximum_tracking_error = 0.0
    maximum_mimic_error = _mimic_error(convex_robot)
    completed = 0
    for name in commanded_names:
        index = by_name[name]
        lower, upper = limits[index]
        span = upper - lower
        assert torch.isfinite(span) and float(span) > 0.0
        for value in (lower + SWEEP_MARGIN * span, upper - SWEEP_MARGIN * span):
            commands = dict(SWEEP_SUPPORT_POSE_RAD)
            commands[name] = float(value)
            target = set_joint_targets(convex_robot, commands)
            _step(simulation, robots, sensors, SWEEP_SETTLE_STEPS)
            error = _tracking_error(convex_robot, target, index)
            maximum_tracking_error = max(maximum_tracking_error, error)
            maximum_mimic_error = max(maximum_mimic_error, _mimic_error(convex_robot))
            assert error <= TRACKING_TOLERANCE_RAD, (name, float(value), error)
            assert maximum_mimic_error <= MIMIC_TOLERANCE_RAD, (
                name,
                float(value),
                maximum_mimic_error,
                _mimic_errors(convex_robot),
            )
            assert_finite_joint_state(convex_robot, LIMIT_TOLERANCE_RAD)
            completed += 1
        convex_robot.set_joint_position_target_index(
            target=torch.zeros_like(convex_robot.data.joint_pos.torch)
        )
        _step(simulation, robots, sensors, 80)
    assert completed == 40

    closed = alex_purdue_ezgripper_targets(0.0, "left") | alex_purdue_ezgripper_targets(
        0.0, "right"
    )
    closed_target = set_joint_targets(convex_robot, closed)
    closed_forces = _step(simulation, robots, sensors, 500)
    closed_error = max(
        _tracking_error(convex_robot, closed_target, by_name[name]) for name in closed
    )
    maximum_mimic_error = max(maximum_mimic_error, _mimic_error(convex_robot))
    assert closed_error <= TRACKING_TOLERANCE_RAD
    assert maximum_mimic_error <= MIMIC_TOLERANCE_RAD
    assert closed_forces[0] > 0.1 and closed_forces[1] > 0.1, closed_forces
    assert closed_forces[2] > 0.1, closed_forces
    for object_path in object_paths.values():
        sim_utils.modify_collision_properties(
            object_path,
            PhysxCollisionPropertiesCfg(collision_enabled=False),
            stage=stage,
        )

    open_commands = alex_purdue_ezgripper_targets(
        1.0, "left"
    ) | alex_purdue_ezgripper_targets(1.0, "right")
    open_target = set_joint_targets(convex_robot, open_commands)
    _step(simulation, robots, sensors, 2000)
    open_error = max(
        _tracking_error(convex_robot, open_target, by_name[name])
        for name in open_commands
    )
    assert open_error <= TRACKING_TOLERANCE_RAD, (
        open_error,
        {
            name: float(convex_robot.data.joint_pos.torch[0, by_name[name]])
            for name in open_commands
        },
    )
    assert_finite_joint_state(convex_robot, LIMIT_TOLERANCE_RAD)

    print(
        "PASS: Alex Purdue SAKE PhysX/TGS readiness on "
        f"{torch.cuda.get_device_name(0)} (cuda:0); profiles=2, joints=24, "
        f"commanded=20, sweep_points={completed}, max_tracking_error={maximum_tracking_error:.6f} rad, "
        f"max_mimic_error={maximum_mimic_error:.8f} rad, gravity_hold={maximum_hold_error:.6f} rad, "
        f"filters=3/{filtered_shape_pairs} shape pairs, bilateral_object_contact={closed_forces[:2]}, "
        f"deliberate_self_contact={closed_forces[2]:.3f} N, tcp={palm_tcp}",
        flush=True,
    )


run_gate(main, simulation_app)
