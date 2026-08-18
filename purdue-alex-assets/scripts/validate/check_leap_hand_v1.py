#!/usr/bin/env python3
# Copyright (c) 2026, Patrizio Acquadro.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX/TGS CUDA gate for mounted LEAP Hand V1 left and right."""

from __future__ import annotations

import math
from pathlib import Path

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, device="cuda:0")
simulation_app = app_launcher.app

import omni.usd  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab_physx.sim.schemas import (  # noqa: E402
    PhysxCollisionPropertiesCfg,
    PhysxRigidBodyPropertiesCfg,
)

from ihmc_alex_isaaclab.end_effectors.leap_hand_v1 import (  # noqa: E402
    LEAP_HAND_V1_MASS_KG,
    LEAP_HAND_V1_PALM_LINK,
    author_leap_hand_v1_mount,
    make_leap_hand_v1_cfg,
)
from _physx import (  # noqa: E402
    assert_finite_joint_state,
    make_readiness_simulation_cfg,
    maximum_contact_force,
    require_readiness_gpu,
    rigid_body_path,
    run_gate,
    set_joint_targets,
)

DT_S = 1.0 / 120.0
HOME_STEPS = 1000
CONTACT_STEPS = 240
SWEEP_SETTLE_STEPS = 240
SWEEP_FRACTION = 0.02
TRACKING_TOLERANCE_RAD = 0.05
MOUNT_TOLERANCE = 1.0e-4
CONTACT_GAP_M = 0.02
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

FIXTURE_POSITIONS = {
    "left": (-0.4, 0.0, 1.0),
    "right": (0.4, 0.0, 1.0),
}
MOUNT_TRANSFORMS = {
    "left": ((0.0, 0.0, 0.04), (0.0, 0.0, 0.0, 1.0)),
    "right": ((0.01, -0.005, 0.05), (0.0, 0.0, 0.0, 1.0)),
}
HOME_POSITIONS = {
    "left": {
        f"a_{index}": (-1.4 if index == 12 else -1.6 if index == 13 else 0.0)
        for index in range(16)
    },
    "right": {f"a_{index}": (1.4 if index == 12 else 0.0) for index in range(16)},
}


def _colliders(stage: Usd.Stage, root_path: str) -> list[Usd.Prim]:
    return [
        prim
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path))
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]


def _spawn_fixture(path: str, position: tuple[float, float, float]) -> None:
    cfg = sim_utils.CuboidCfg(
        size=(0.06, 0.06, 0.06),
        rigid_props=PhysxRigidBodyPropertiesCfg(
            disable_gravity=True,
            kinematic_enabled=True,
        ),
        collision_props=PhysxCollisionPropertiesCfg(collision_enabled=False),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    )
    cfg.func(path, cfg, translation=position)


def _fingertip_probe_position(stage: Usd.Stage, body_path: str) -> tuple[float, ...]:
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
            UsdGeom.Tokens.guide,
        ],
    )
    bounds = cache.ComputeWorldBound(
        stage.GetPrimAtPath(body_path)
    ).ComputeAlignedRange()
    minimum = bounds.GetMin()
    maximum = bounds.GetMax()
    radius = 0.01
    return (
        0.5 * float(minimum[0] + maximum[0]),
        0.5 * float(minimum[1] + maximum[1]),
        float(maximum[2]) + radius + CONTACT_GAP_M,
    )


def _spawn_contact_probe(path: str, position: tuple[float, ...]) -> RigidObject:
    return RigidObject(
        RigidObjectCfg(
            prim_path=path,
            spawn=sim_utils.SphereCfg(
                radius=0.01,
                rigid_props=PhysxRigidBodyPropertiesCfg(
                    disable_gravity=True,
                    kinematic_enabled=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_depenetration_velocity=1.0,
                ),
                collision_props=PhysxCollisionPropertiesCfg(collision_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=position),
        )
    )


def _make_sensor(
    root_path: str,
    body_path: str,
    filter_path: str,
    stage: Usd.Stage,
) -> ContactSensor:
    sim_utils.activate_contact_sensors(body_path, stage=stage)
    return ContactSensor(
        ContactSensorCfg(
            prim_path=f"{root_path}/fingertip",
            filter_prim_paths_expr=[filter_path],
            update_period=0.0,
            history_length=1,
            debug_vis=False,
        )
    )


def _quaternion_error(actual_xyzw: torch.Tensor, expected_xyzw: torch.Tensor) -> float:
    actual = actual_xyzw.to(torch.float64)
    expected = expected_xyzw.to(torch.float64)
    actual = actual / torch.linalg.vector_norm(actual)
    expected = expected / torch.linalg.vector_norm(expected)
    chord = min(
        float(torch.linalg.vector_norm(actual - expected)),
        float(torch.linalg.vector_norm(actual + expected)),
    )
    return 4.0 * math.asin(min(1.0, 0.5 * chord))


def _mount_error(
    hand: Articulation,
    expected_position: torch.Tensor,
    expected_rotation: torch.Tensor,
) -> tuple[float, float]:
    pose = hand.data.root_pose_w.torch[0]
    return (
        float(torch.linalg.vector_norm(pose[:3] - expected_position)),
        _quaternion_error(pose[3:], expected_rotation),
    )


def _step(
    simulation: SimulationContext,
    hands: tuple[Articulation, Articulation],
    sensors: tuple[ContactSensor, ContactSensor],
    expected_mounts: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    count: int,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    maximum_force = [0.0, 0.0]
    final_force = [0.0, 0.0]
    final_mount_error = [0.0, 0.0]
    for _ in range(count):
        for hand in hands:
            hand.write_data_to_sim()
        simulation.step()
        for index, hand in enumerate(hands):
            hand.update(DT_S)
            assert_finite_joint_state(hand, 0.01)
            position_error, rotation_error = _mount_error(hand, *expected_mounts[index])
            final_mount_error[index] = max(position_error, rotation_error)
        for index, sensor in enumerate(sensors):
            sensor.update(DT_S)
            force = maximum_contact_force(sensor)
            final_force[index] = force
            maximum_force[index] = max(maximum_force[index], force)
    return tuple(maximum_force), tuple(final_mount_error), tuple(final_force)


def _assert_stage_contract(stage: Usd.Stage, root_path: str) -> None:
    colliders = _colliders(stage, root_path)
    assert len(colliders) == 17, len(colliders)
    for collider in colliders:
        contact_offset = float(
            collider.GetAttribute("physxCollision:contactOffset").Get()
        )
        rest_offset = float(collider.GetAttribute("physxCollision:restOffset").Get())
        assert abs(contact_offset - 0.002) <= 1.0e-9, contact_offset
        assert abs(rest_offset) <= 1.0e-9, rest_offset
    rigid_bodies = [
        prim
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path))
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    assert len(rigid_bodies) == 17
    assert all(
        body.GetAttribute("physxRigidBody:disableGravity").Get() is False
        for body in rigid_bodies
    )
    assert all(
        "PhysxContactReportAPI" in body.GetAppliedSchemas() for body in rigid_bodies
    )
    filtered_pairs = [
        target
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path))
        for target in prim.GetRelationship("physics:filteredPairs").GetTargets()
        if prim.GetRelationship("physics:filteredPairs")
    ]
    assert not filtered_pairs
    palm = stage.GetPrimAtPath(
        rigid_body_path(
            stage,
            root_path,
            LEAP_HAND_V1_PALM_LINK[root_path.rsplit("/", 1)[-1].lower()],
        )
    )
    assert palm.GetAttribute("physxArticulation:enabledSelfCollisions").Get() is True
    assert (
        palm.GetAttribute("physxArticulation:solverPositionIterationCount").Get() == 8
    )
    assert (
        palm.GetAttribute("physxArticulation:solverVelocityIterationCount").Get() == 4
    )
    material_path = f"{root_path}/material"
    material = stage.GetPrimAtPath(material_path)
    assert material.HasAPI(UsdPhysics.MaterialAPI)
    material_api = UsdPhysics.MaterialAPI(material)
    assert float(material_api.GetStaticFrictionAttr().Get()) == 1.0
    assert float(material_api.GetDynamicFrictionAttr().Get()) == 1.0
    assert float(material_api.GetRestitutionAttr().Get()) == 0.0
    assert all(
        collider.GetRelationship("material:binding:physics").GetTargets()
        == [material.GetPath()]
        for collider in colliders
    )


def _assert_runtime_contract(hand: Articulation, side: str) -> None:
    assert hand.num_joints == 16 and hand.num_bodies == 17
    assert set(hand.data.joint_names) == {f"a_{index}" for index in range(16)}
    assert LEAP_HAND_V1_PALM_LINK[side] in hand.data.body_names
    masses = wp.to_torch(hand.root_view.get_masses())
    assert abs(float(masses.sum()) - LEAP_HAND_V1_MASS_KG[side]) <= 1.0e-6


def main() -> None:
    require_readiness_gpu()
    simulation = SimulationContext(make_readiness_simulation_cfg(DT_S))

    for side in ("left", "right"):
        _spawn_fixture(f"/World/{side.title()}Wrist", FIXTURE_POSITIONS[side])
    hands = []
    for side in ("left", "right"):
        cfg = make_leap_hand_v1_cfg(
            side,
            f"/World/{side.title()}",
            fix_base=False,
            usd_dir=(
                REPOSITORY_ROOT
                / "build"
                / "isaac"
                / "leap_hand_v1"
                / "validation"
                / side
            ),
        )
        cfg.spawn.force_usd_conversion = True
        cfg.init_state.joint_pos = HOME_POSITIONS[side]
        hands.append(Articulation(cfg))
    hand_pair = (hands[0], hands[1])

    stage = omni.usd.get_context().get_stage()
    sensors = []
    probes = []
    for side in ("left", "right"):
        root_path = f"/World/{side.title()}"
        wrist_path = f"/World/{side.title()}Wrist"
        position, rotation = MOUNT_TRANSFORMS[side]
        joint_path = author_leap_hand_v1_mount(
            stage,
            wrist_path,
            root_path,
            side,
            position,
            rotation,
        )
        joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath(joint_path))
        assert joint.GetExcludeFromArticulationAttr().Get() is True
        _assert_stage_contract(stage, root_path)
        fingertip = rigid_body_path(stage, root_path, "fingertip")
        probe_path = f"/World/{side.title()}ContactProbe"
        probes.append(
            _spawn_contact_probe(
                probe_path,
                _fingertip_probe_position(stage, fingertip),
            )
        )
        sensors.append(_make_sensor(root_path, fingertip, probe_path, stage))
    sensor_pair = (sensors[0], sensors[1])

    simulation.reset()
    expected_mounts = []
    for side in ("left", "right"):
        fixture = FIXTURE_POSITIONS[side]
        mount_position, mount_rotation = MOUNT_TRANSFORMS[side]
        expected_mounts.append(
            (
                torch.tensor(
                    [fixture[index] + mount_position[index] for index in range(3)],
                    device="cuda:0",
                ),
                torch.tensor(mount_rotation, device="cuda:0"),
            )
        )
    expected_mount_pair = (expected_mounts[0], expected_mounts[1])

    home_targets = []
    for side, hand in zip(("left", "right"), hand_pair, strict=True):
        _assert_runtime_contract(hand, side)
        assert not hand.is_fixed_base
        home = set_joint_targets(hand, HOME_POSITIONS[side])
        hand.write_joint_state_to_sim_index(
            position=home,
            velocity=torch.zeros_like(home),
        )
        hand.reset()
        home_targets.append(set_joint_targets(hand, HOME_POSITIONS[side]))

    home_forces, home_mount_errors, _ = _step(
        simulation,
        hand_pair,
        sensor_pair,
        expected_mount_pair,
        HOME_STEPS,
    )
    assert all(force <= 1.0e-4 for force in home_forces), home_forces
    assert all(error <= MOUNT_TOLERANCE for error in home_mount_errors), (
        home_mount_errors
    )
    home_errors = [
        float(torch.abs(hand.data.joint_pos.torch - target).max())
        for hand, target in zip(hand_pair, home_targets, strict=True)
    ]
    assert all(error <= TRACKING_TOLERANCE_RAD for error in home_errors), home_errors

    downward_velocity = torch.tensor([[0.0, 0.0, -0.5, 0.0, 0.0, 0.0]], device="cuda:0")
    for probe in probes:
        probe.write_root_velocity_to_sim_index(root_velocity=downward_velocity)
    contact_forces, _, final_contact_forces = _step(
        simulation,
        hand_pair,
        sensor_pair,
        expected_mount_pair,
        CONTACT_STEPS,
    )
    assert all(force > 1.0e-4 for force in contact_forces), (
        contact_forces,
        final_contact_forces,
    )
    assert all(force < 1.0e3 for force in contact_forces), contact_forces
    assert all(force <= 1.0e-4 for force in final_contact_forces), final_contact_forces
    _, settled_mount_errors, _ = _step(
        simulation,
        hand_pair,
        sensor_pair,
        expected_mount_pair,
        CONTACT_STEPS,
    )
    assert all(error <= MOUNT_TOLERANCE for error in settled_mount_errors), (
        settled_mount_errors
    )

    maximum_tracking_error = 0.0
    completed_sweeps = 0
    for hand_index, (side, hand) in enumerate(
        zip(("left", "right"), hand_pair, strict=True)
    ):
        by_name = {name: index for index, name in enumerate(hand.data.joint_names)}
        limits = hand.data.joint_pos_limits.torch[0]
        for motor_id in range(16):
            joint_index = by_name[f"a_{motor_id}"]
            lower, upper = limits[joint_index]
            span = upper - lower
            home_value = HOME_POSITIONS[side][f"a_{motor_id}"]
            for value in (
                max(float(lower), home_value - SWEEP_FRACTION * float(span)),
                min(float(upper), home_value + SWEEP_FRACTION * float(span)),
            ):
                values = {f"a_{motor_id}": float(value)}
                target = set_joint_targets(hand, HOME_POSITIONS[side] | values)
                other_side = ("left", "right")[1 - hand_index]
                set_joint_targets(hand_pair[1 - hand_index], HOME_POSITIONS[other_side])
                _step(
                    simulation,
                    hand_pair,
                    sensor_pair,
                    expected_mount_pair,
                    SWEEP_SETTLE_STEPS,
                )
                error = float(
                    torch.abs(
                        hand.data.joint_pos.torch[:, joint_index]
                        - target[:, joint_index]
                    ).max()
                )
                maximum_tracking_error = max(maximum_tracking_error, error)
                assert error <= TRACKING_TOLERANCE_RAD, (
                    side,
                    motor_id,
                    value,
                    error,
                )
                completed_sweeps += 1

    left_target = set_joint_targets(
        hand_pair[0], HOME_POSITIONS["left"] | {"a_15": 0.6}
    )
    right_target = set_joint_targets(
        hand_pair[1], HOME_POSITIONS["right"] | {"a_15": -0.6}
    )
    _step(
        simulation,
        hand_pair,
        sensor_pair,
        expected_mount_pair,
        360,
    )
    left_index = hand_pair[0].data.joint_names.index("a_15")
    right_index = hand_pair[1].data.joint_names.index("a_15")
    left_position = float(hand_pair[0].data.joint_pos.torch[:, left_index].item())
    right_position = float(hand_pair[1].data.joint_pos.torch[:, right_index].item())
    assert left_position > 0.0 and right_position < 0.0
    assert (
        abs(left_position - float(left_target[:, left_index])) <= TRACKING_TOLERANCE_RAD
    )
    assert (
        abs(right_position - float(right_target[:, right_index]))
        <= TRACKING_TOLERANCE_RAD
    )
    right_before = hand_pair[1].data.joint_pos.torch.clone()
    set_joint_targets(hand_pair[0], HOME_POSITIONS["left"] | {"a_15": -0.6})
    _step(
        simulation,
        hand_pair,
        sensor_pair,
        expected_mount_pair,
        360,
    )
    cross_talk = float(
        torch.abs(hand_pair[1].data.joint_pos.torch - right_before).max()
    )
    assert cross_talk <= 0.01, cross_talk
    for side, hand in zip(("left", "right"), hand_pair, strict=True):
        set_joint_targets(hand, HOME_POSITIONS[side])
    _, final_mount_errors, _ = _step(
        simulation,
        hand_pair,
        sensor_pair,
        expected_mount_pair,
        SWEEP_SETTLE_STEPS,
    )
    assert all(error <= MOUNT_TOLERANCE for error in final_mount_errors), (
        final_mount_errors
    )
    maximum_mount_error = max(
        *home_mount_errors,
        *settled_mount_errors,
        *final_mount_errors,
    )

    print(
        "PASS: LEAP Hand V1 left/right PhysX/TGS smoke passed on RTX 4090 "
        f"cuda:0 at dt={DT_S:.8f}; bodies=17/17; dofs=16/16; "
        f"sweeps={completed_sweeps}; max_tracking_error={maximum_tracking_error:.6f} rad; "
        f"max_mount_error={maximum_mount_error:.3e}; "
        f"contact_forces={tuple(round(value, 6) for value in contact_forces)} N; "
        f"resolved_contact_forces={tuple(round(value, 6) for value in final_contact_forces)} N; "
        f"initial_contact_gap={CONTACT_GAP_M:.3f} m; "
        f"cross_talk={cross_talk:.3e} rad; self_collision=enabled",
        flush=True,
    )


run_gate(main, simulation_app)
