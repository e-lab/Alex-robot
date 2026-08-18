#!/usr/bin/env python3
# Copyright (c) 2026, Patrizio Acquadro.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX/TGS CUDA readiness gate for Golden Robot and standalone WSG32."""

from __future__ import annotations

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, device="cuda:0")
simulation_app = app_launcher.app

import omni.usd  # noqa: E402
import torch  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.sim.schemas.schemas_cfg import RigidBodyBaseCfg  # noqa: E402
from isaaclab.sim.spawners.materials.physics_materials_cfg import (  # noqa: E402
    RigidBodyMaterialBaseCfg,
)
from isaaclab_physx.sim.schemas import (  # noqa: E402
    PhysxArticulationRootPropertiesCfg,
    PhysxCollisionPropertiesCfg,
    PhysxRigidBodyPropertiesCfg,
)

from ihmc_alex_isaaclab.end_effectors.weiss_wsg32 import (  # noqa: E402
    WSG32_JAW_STROKE_M,
    WSG32_MODEL_REFERENCE_DYNAMIC_FRICTION,
    WSG32_MODEL_REFERENCE_STATIC_FRICTION,
    alex_purdue_wsg32_targets,
    make_wsg32_umi_v1_cfg,
)
from ihmc_alex_isaaclab.robots.alex_purdue import make_alex_purdue_cfg  # noqa: E402
from ihmc_alex_isaaclab.robots.purdue_frames import (  # noqa: E402
    author_alex_purdue_frames,
)
from ihmc_alex_isaaclab.robots.purdue_physics import (  # noqa: E402
    ALEX_PURDUE_WSG32_MIMIC_JOINT_PAIRS,
    ALEX_PURDUE_WSG32_SELF_COLLISION_FILTER_PAIRS,
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
SWEEP_MARGIN = 0.02
SWEEP_SETTLE_STEPS = 360
TRACKING_TOLERANCE_RAD = 0.05
MIMIC_TOLERANCE_M = 1.0e-4
GRASP_PRELOAD_STEPS = 200
GRASP_HOLD_STEPS = 600
GRASP_OBJECT_MASS_KG = 0.25
GRASP_OBJECT_POSITION_M = (0.1305, 0.0, 1.0)
ALEX_SWEEP_JOINTS = (
    "NECK_Z",
    "NECK_Y",
    "LEFT_SHOULDER_Y",
    "LEFT_SHOULDER_X",
    "LEFT_SHOULDER_Z",
    "LEFT_ELBOW_Y",
    "LEFT_WRIST_Z",
    "LEFT_WRIST_X",
    "LEFT_GRIPPER_Y",
    "RIGHT_SHOULDER_Y",
    "RIGHT_SHOULDER_X",
    "RIGHT_SHOULDER_Z",
    "RIGHT_ELBOW_Y",
    "RIGHT_WRIST_Z",
    "RIGHT_WRIST_X",
    "RIGHT_GRIPPER_Y",
)
ALEX_SWEEP_SUPPORT_POSE_RAD = {
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
    articulations: tuple[Articulation, ...],
    rigid_objects: tuple[RigidObject, ...],
    sensors: tuple[ContactSensor, ...],
    count: int,
    *,
    pin_object: RigidObject | None = None,
) -> tuple[float, ...]:
    maxima = [0.0] * len(sensors)
    follower_names = {follower for follower, _ in ALEX_PURDUE_WSG32_MIMIC_JOINT_PAIRS}
    if pin_object is not None:
        pinned_pose = pin_object.data.root_pose_w.torch.clone()
        pinned_velocity = torch.zeros_like(pin_object.data.root_vel_w.torch)
    for _ in range(count):
        if pin_object is not None:
            pin_object.write_root_pose_to_sim_index(root_pose=pinned_pose)
            pin_object.write_root_velocity_to_sim_index(root_velocity=pinned_velocity)
        for articulation in articulations:
            if articulation.num_joints > 2:
                apply_gravity_compensation(
                    articulation, excluded_joint_names=follower_names
                )
            articulation.write_data_to_sim()
        for rigid_object in rigid_objects:
            rigid_object.write_data_to_sim()
        simulation.step()
        for articulation in articulations:
            articulation.update(DT_S)
        for rigid_object in rigid_objects:
            rigid_object.update(DT_S)
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


def _mimic_error(robot: Articulation, pairs: tuple[tuple[str, str], ...]) -> float:
    by_name = {name: index for index, name in enumerate(robot.data.joint_names)}
    assert all(follower in by_name and leader in by_name for follower, leader in pairs)
    return max(
        float(
            torch.abs(
                robot.data.joint_pos.torch[:, by_name[follower]]
                - robot.data.joint_pos.torch[:, by_name[leader]]
            ).max()
        )
        for follower, leader in pairs
    )


def _assert_filter_contract(stage, root_path: str) -> int:
    authored: list[tuple[str, str]] = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        relationship = prim.GetRelationship("physics:filteredPairs")
        if relationship:
            authored.extend(
                (str(prim.GetPath()), str(target))
                for target in relationship.GetTargets()
            )
    assert len(ALEX_PURDUE_WSG32_SELF_COLLISION_FILTER_PAIRS) == 5
    for first, second in ALEX_PURDUE_WSG32_SELF_COLLISION_FILTER_PAIRS:
        assert any(first in left and second in right for left, right in authored), (
            first,
            second,
            authored,
        )
    return len(authored)


def _spawn_golden_probe(path: str, position: tuple[float, float, float]) -> None:
    cfg = sim_utils.SphereCfg(
        radius=0.015,
        rigid_props=RigidBodyBaseCfg(disable_gravity=True, kinematic_enabled=True),
        collision_props=PhysxCollisionPropertiesCfg(collision_enabled=True),
    )
    cfg.func(path, cfg, translation=position)


def _make_sensor(body_path: str, filter_path: str, stage) -> ContactSensor:
    sim_utils.activate_contact_sensors(body_path, stage=stage)
    return ContactSensor(
        ContactSensorCfg(
            prim_path=body_path,
            filter_prim_paths_expr=[filter_path],
            update_period=0.0,
            history_length=1,
            debug_vis=False,
        )
    )


def main() -> None:
    require_readiness_gpu()
    simulation = SimulationContext(make_readiness_simulation_cfg(DT_S))

    source_cfg = make_alex_purdue_cfg(
        variant="source", end_effector="wsg32_umi_v1"
    ).replace(prim_path="/World/AlexPurdueWSGSource")
    source_cfg.init_state.pos = (-2.5, 0.0, 0.0)
    source_cfg.spawn.articulation_props = PhysxArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    )
    convex_cfg = make_alex_purdue_cfg(
        variant="full_convex", end_effector="wsg32_umi_v1"
    ).replace(prim_path="/World/AlexPurdueWSGConvex")
    convex_cfg.init_state.pos = (2.5, 0.0, 0.0)
    convex_cfg.spawn.articulation_props = PhysxArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    )
    source_robot = Articulation(source_cfg)
    convex_robot = Articulation(convex_cfg)

    wsg_cfg = make_wsg32_umi_v1_cfg().replace(prim_path="/World/WSG32Standalone")
    wsg_cfg.init_state.pos = (0.0, 0.0, 1.0)
    wsg_cfg.init_state.joint_pos = {
        "WSG32_JAW_OPENING": WSG32_JAW_STROKE_M,
        "WSG32_JAW_FOLLOWER": WSG32_JAW_STROKE_M,
    }
    wsg_cfg.spawn.articulation_props = PhysxArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    )
    wsg = Articulation(wsg_cfg)

    grasp_object = RigidObject(
        RigidObjectCfg(
            prim_path="/World/GraspObject",
            spawn=sim_utils.CylinderCfg(
                radius=0.015,
                height=0.04,
                rigid_props=PhysxRigidBodyPropertiesCfg(
                    disable_gravity=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_depenetration_velocity=1.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=GRASP_OBJECT_MASS_KG),
                collision_props=PhysxCollisionPropertiesCfg(collision_enabled=True),
                physics_material=RigidBodyMaterialBaseCfg(
                    static_friction=WSG32_MODEL_REFERENCE_STATIC_FRICTION,
                    dynamic_friction=WSG32_MODEL_REFERENCE_DYNAMIC_FRICTION,
                    restitution=0.0,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=GRASP_OBJECT_POSITION_M),
        )
    )

    stage = omni.usd.get_context().get_stage()
    source_frames = author_alex_purdue_frames(
        stage,
        source_cfg.prim_path,
        variant="source",
        end_effector="wsg32_umi_v1",
    )
    convex_frames = author_alex_purdue_frames(
        stage,
        convex_cfg.prim_path,
        variant="full_convex",
        end_effector="wsg32_umi_v1",
    )
    assert len(source_frames) == len(convex_frames) == 15
    base_tcp_distances: dict[str, float] = {}
    golden_probe_paths: dict[str, str] = {}
    for side in ("LEFT", "RIGHT"):
        base = _world_translation(stage, convex_frames[f"{side}_WSG32_BASE_FRAME"])
        tcp = _world_translation(stage, convex_frames[f"{side}_WSG32_TCP_FRAME"])
        distance = sum((tcp[axis] - base[axis]) ** 2 for axis in range(3)) ** 0.5
        assert abs(distance - 0.1305) <= 1.0e-6
        base_tcp_distances[side] = distance
        probe_path = f"/World/{side.title()}GoldenWSGProbe"
        _spawn_golden_probe(probe_path, tcp)
        golden_probe_paths[side] = probe_path

    standalone_negative = rigid_body_path(
        stage, wsg_cfg.prim_path, "WSG32_NEGATIVE_JAW_LINK"
    )
    standalone_positive = rigid_body_path(
        stage, wsg_cfg.prim_path, "WSG32_POSITIVE_JAW_LINK"
    )
    sensors = (
        _make_sensor(standalone_negative, "/World/GraspObject", stage),
        _make_sensor(standalone_positive, "/World/GraspObject", stage),
        _make_sensor(
            rigid_body_path(
                stage, convex_cfg.prim_path, "left_WSG32_NEGATIVE_JAW_LINK"
            ),
            golden_probe_paths["LEFT"],
            stage,
        ),
        _make_sensor(
            rigid_body_path(
                stage, convex_cfg.prim_path, "left_WSG32_POSITIVE_JAW_LINK"
            ),
            golden_probe_paths["LEFT"],
            stage,
        ),
        _make_sensor(
            rigid_body_path(
                stage, convex_cfg.prim_path, "right_WSG32_NEGATIVE_JAW_LINK"
            ),
            golden_probe_paths["RIGHT"],
            stage,
        ),
        _make_sensor(
            rigid_body_path(
                stage, convex_cfg.prim_path, "right_WSG32_POSITIVE_JAW_LINK"
            ),
            golden_probe_paths["RIGHT"],
            stage,
        ),
    )

    simulation.reset()
    articulations = (source_robot, convex_robot, wsg)
    objects = (grasp_object,)
    for robot in (source_robot, convex_robot):
        assert robot.is_fixed_base
    assert wsg.is_fixed_base
    initial_roots = {
        robot.cfg.prim_path: robot.data.root_pos_w.torch.clone()
        for robot in (source_robot, convex_robot, wsg)
    }
    for robot in (source_robot, convex_robot):
        zero = torch.zeros_like(robot.data.joint_pos.torch)
        robot.write_joint_state_to_sim_index(position=zero, velocity=zero)
        robot.reset()
        robot.set_joint_position_target_index(target=zero)
    wsg.reset()
    standalone_open = set_joint_targets(wsg, {"WSG32_JAW_OPENING": WSG32_JAW_STROKE_M})
    _step(simulation, articulations, objects, sensors, 300, pin_object=grasp_object)
    standalone_by_name = {
        name: index for index, name in enumerate(wsg.data.joint_names)
    }
    opening_index = standalone_by_name["WSG32_JAW_OPENING"]
    standalone_open_error = float(
        torch.abs(
            wsg.data.joint_pos.torch[:, opening_index]
            - standalone_open[:, opening_index]
        ).max()
    )
    assert standalone_open_error <= 0.002

    set_joint_targets(wsg, {"WSG32_JAW_OPENING": 0.0})
    grasp_forces = _step(
        simulation, articulations, objects, sensors, 400, pin_object=grasp_object
    )
    standalone_mimic = _mimic_error(wsg, (("WSG32_JAW_FOLLOWER", "WSG32_JAW_OPENING"),))
    assert standalone_mimic <= MIMIC_TOLERANCE_M
    assert grasp_forces[0] > 0.1 and grasp_forces[1] > 0.1, grasp_forces

    _step(simulation, articulations, objects, sensors, GRASP_PRELOAD_STEPS)
    hold_position = grasp_object.data.root_pos_w.torch.clone()
    _step(simulation, articulations, objects, sensors, GRASP_HOLD_STEPS)
    grasp_slip = float(
        torch.linalg.vector_norm(
            grasp_object.data.root_pos_w.torch - hold_position, dim=-1
        ).max()
    )
    assert grasp_slip <= 0.002, grasp_slip

    golden_open = alex_purdue_wsg32_targets(1.0, "left") | alex_purdue_wsg32_targets(
        1.0, "right"
    )
    set_joint_targets(convex_robot, golden_open)
    _step(simulation, articulations, objects, sensors, 400)
    golden_close = alex_purdue_wsg32_targets(0.0, "left") | alex_purdue_wsg32_targets(
        0.0, "right"
    )
    set_joint_targets(convex_robot, golden_close)
    golden_forces = _step(simulation, articulations, objects, sensors, 500)
    assert all(force > 0.1 for force in golden_forces[2:]), golden_forces
    golden_mimic = _mimic_error(convex_robot, ALEX_PURDUE_WSG32_MIMIC_JOINT_PAIRS)
    assert golden_mimic <= MIMIC_TOLERANCE_M

    by_name = {name: index for index, name in enumerate(convex_robot.data.joint_names)}
    limits = convex_robot.data.joint_pos_limits.torch[0]
    maximum_tracking_error = 0.0
    maximum_mimic_error = golden_mimic
    completed = 0
    for name in ALEX_SWEEP_JOINTS:
        index = by_name[name]
        lower, upper = limits[index]
        span = upper - lower
        for value in (lower + SWEEP_MARGIN * span, upper - SWEEP_MARGIN * span):
            commands = dict(ALEX_SWEEP_SUPPORT_POSE_RAD)
            commands[name] = float(value)
            commands.update(
                {
                    "left_WSG32_JAW_OPENING": WSG32_JAW_STROKE_M,
                    "right_WSG32_JAW_OPENING": WSG32_JAW_STROKE_M,
                }
            )
            set_joint_targets(convex_robot, commands)
            _step(simulation, articulations, objects, sensors, SWEEP_SETTLE_STEPS)
            error = float(
                torch.abs(convex_robot.data.joint_pos.torch[:, index] - value).max()
            )
            maximum_tracking_error = max(maximum_tracking_error, error)
            maximum_mimic_error = max(
                maximum_mimic_error,
                _mimic_error(convex_robot, ALEX_PURDUE_WSG32_MIMIC_JOINT_PAIRS),
            )
            assert error <= TRACKING_TOLERANCE_RAD, (name, float(value), error)
            assert maximum_mimic_error <= MIMIC_TOLERANCE_M
            assert_finite_joint_state(convex_robot, 2.0e-4)
            completed += 1
    assert completed == 2 * len(ALEX_SWEEP_JOINTS)

    filtered_shape_pairs = _assert_filter_contract(stage, convex_cfg.prim_path)
    for robot in articulations:
        assert_finite_joint_state(robot, 2.0e-4)
        assert (
            float(
                torch.abs(
                    robot.data.root_pos_w.torch - initial_roots[robot.cfg.prim_path]
                ).max()
            )
            <= 1.0e-6
        )

    print(
        "PASS: Golden Robot WSG32 + UMI and standalone WSG32 PhysX/TGS readiness on "
        f"{torch.cuda.get_device_name(0)} (cuda:0); profiles=2, alex_sweep_points={completed}, "
        f"max_tracking_error={maximum_tracking_error:.6f} rad, golden_mimic={maximum_mimic_error:.8f} m, "
        f"standalone_open_error={standalone_open_error:.8f} m, standalone_mimic={standalone_mimic:.8f} m, "
        f"grasp_mass={GRASP_OBJECT_MASS_KG:.2f} kg, hold={GRASP_HOLD_STEPS * DT_S:.3f} s, "
        f"grasp_slip={grasp_slip:.6f} m, bilateral_grasp={grasp_forces[:2]}, "
        f"golden_bilateral_contacts={golden_forces[2:]}, tcp={base_tcp_distances}, "
        f"filters=5/{filtered_shape_pairs} shape pairs",
        flush=True,
    )


run_gate(main, simulation_app)
