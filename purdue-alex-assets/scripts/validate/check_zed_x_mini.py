#!/usr/bin/env python3
# Copyright (c) 2026, Patrizio Acquadro.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX/TGS CUDA Golden Robot gate for the official ZED X Mini Wide."""

from __future__ import annotations

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, device="cuda:0", enable_cameras=True)
simulation_app = app_launcher.app

import omni.usd  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402
from pxr import UsdPhysics  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import Camera  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils.configclass import configclass  # noqa: E402
from isaaclab.utils.math import quat_error_magnitude, subtract_frame_transforms  # noqa: E402
from isaaclab_physx.physics import PhysxManager  # noqa: E402
from isaaclab_physx.sim.schemas import (  # noqa: E402
    PhysxArticulationRootPropertiesCfg,
)

from ihmc_alex_isaaclab.robots.alex_purdue import make_alex_purdue_cfg  # noqa: E402
from ihmc_alex_isaaclab.robots.purdue_physics import (  # noqa: E402
    ALEX_PURDUE_WSG32_MIMIC_JOINT_PAIRS,
)
from ihmc_alex_isaaclab.sensors.zed_x_mini import (  # noqa: E402
    ZED_X_MINI_DIAGONAL_INERTIA_KG_M2,
    ZED_X_MINI_LENS,
    ZED_X_MINI_MASS_KG,
    ZED_X_MINI_MODEL,
    author_alex_purdue_zed_x_mini_mount,
    make_zed_x_mini_cfgs,
)
from ihmc_alex_isaaclab.sensors.zed_x_mini_dependency import (  # noqa: E402
    load_zed_x_mini_manifest,
)
from _physx import (  # noqa: E402
    apply_gravity_compensation,
    assert_finite_joint_state,
    make_readiness_simulation_cfg,
    make_static_box_cfg,
    require_readiness_gpu,
    run_gate,
)

DT_S = 0.005
WARMUP_STEPS = 120
MOVE_STEPS = 20
POSE_TOLERANCE_M = 1.0e-4
POSE_TOLERANCE_RAD = 1.0e-4
EXPECTED_BASELINE_M = float(load_zed_x_mini_manifest()["asset"]["baseline_m"])


@configclass
class ZedXMiniGoldenSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=1800.0, color=(0.75, 0.80, 0.90)),
    )
    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.DistantLightCfg(intensity=3200.0, color=(1.0, 0.93, 0.82)),
        init_state=AssetBaseCfg.InitialStateCfg(
            rot=(0.2209424, -0.2209424, 0.2209424, 0.9238795)
        ),
    )
    backdrop = make_static_box_cfg(
        "{ENV_REGEX_NS}/Backdrop",
        size=(0.05, 20.0, 20.0),
        position=(3.0, 0.0, 2.0),
        color=(0.10, 0.18, 0.62),
    )
    rear_wall = make_static_box_cfg(
        "{ENV_REGEX_NS}/RearWall",
        size=(0.05, 20.0, 20.0),
        position=(-3.0, 0.0, 2.0),
        color=(0.18, 0.22, 0.28),
    )
    left_wall = make_static_box_cfg(
        "{ENV_REGEX_NS}/LeftWall",
        size=(20.0, 0.05, 20.0),
        position=(0.0, 3.0, 2.0),
        color=(0.28, 0.18, 0.16),
    )
    right_wall = make_static_box_cfg(
        "{ENV_REGEX_NS}/RightWall",
        size=(20.0, 0.05, 20.0),
        position=(0.0, -3.0, 2.0),
        color=(0.16, 0.28, 0.18),
    )
    ceiling = make_static_box_cfg(
        "{ENV_REGEX_NS}/Ceiling",
        size=(20.0, 20.0, 0.05),
        position=(0.0, 0.0, 5.0),
        color=(0.32, 0.32, 0.32),
    )
    red_target = make_static_box_cfg(
        "{ENV_REGEX_NS}/RedTarget",
        size=(0.30, 0.45, 0.60),
        position=(0.9, -0.38, 0.30),
        color=(0.90, 0.08, 0.04),
    )
    green_target = make_static_box_cfg(
        "{ENV_REGEX_NS}/GreenTarget",
        size=(0.40, 0.32, 0.38),
        position=(1.25, 0.32, 0.19),
        color=(0.04, 0.72, 0.15),
    )
    golden_robot = make_alex_purdue_cfg(
        fix_base=True,
        variant="full_convex",
        end_effector="wsg32_umi_v1",
    )
    golden_robot.prim_path = "{ENV_REGEX_NS}/Robot"
    zed_model, zed_left, zed_right = make_zed_x_mini_cfgs(
        "{ENV_REGEX_NS}/ZED_X_Mini",
        resolution="SVGA",
        spawn_init_pos=(0.1, 0.0, 0.55),
    )


def _step(
    simulation: SimulationContext,
    scene: InteractiveScene,
    cameras: tuple[Camera, Camera],
    count: int,
) -> None:
    follower_names = {follower for follower, _ in ALEX_PURDUE_WSG32_MIMIC_JOINT_PAIRS}
    for _ in range(count):
        robot = scene["golden_robot"]
        apply_gravity_compensation(robot, excluded_joint_names=follower_names)
        scene.write_data_to_sim()
        simulation.step()
        scene.update(DT_S)
        for camera in cameras:
            camera.update(DT_S)


def _relative_zed_pose(
    scene: InteractiveScene, zed_body_view: object
) -> tuple[torch.Tensor, torch.Tensor]:
    robot = scene["golden_robot"]
    head_index = robot.data.body_names.index("HEAD_LINK")
    head_pose = robot.data.body_pose_w.torch[:, head_index]
    zed_pose = wp.to_torch(zed_body_view.get_transforms())
    assert tuple(zed_pose.shape) == (1, 7)
    return subtract_frame_transforms(
        head_pose[:, :3],
        head_pose[:, 3:7],
        zed_pose[:, :3],
        zed_pose[:, 3:7],
    )


def _camera_tensors(
    cameras: tuple[Camera, Camera],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    left = cameras[0].data.output
    right = cameras[1].data.output
    if left is None or right is None:
        raise RuntimeError("ZED camera outputs were not allocated")
    try:
        return (
            left["rgb"].torch,
            right["rgb"].torch,
            left["distance_to_image_plane"].torch,
        )
    except KeyError as error:
        raise RuntimeError(f"ZED camera output is missing: {error}") from error


def _assert_images(
    left_rgb: torch.Tensor, right_rgb: torch.Tensor, depth: torch.Tensor
) -> float:
    assert left_rgb.is_cuda and right_rgb.is_cuda and depth.is_cuda
    assert left_rgb.device.index == right_rgb.device.index == depth.device.index == 0
    assert tuple(left_rgb.shape[1:3]) == (600, 960)
    assert tuple(right_rgb.shape[1:3]) == (600, 960)
    assert left_rgb.shape[-1] >= 3 and right_rgb.shape[-1] >= 3
    assert torch.isfinite(left_rgb.to(dtype=torch.float32)).all()
    assert torch.isfinite(right_rgb.to(dtype=torch.float32)).all()
    assert float(left_rgb.to(dtype=torch.float32).std()) > 8.0
    assert float(right_rgb.to(dtype=torch.float32).std()) > 8.0
    stereo_difference = float(
        torch.mean(
            torch.abs(
                left_rgb[..., :3].to(dtype=torch.float32)
                - right_rgb[..., :3].to(dtype=torch.float32)
            )
        )
    )
    assert stereo_difference > 0.1, stereo_difference
    assert depth.dtype == torch.float32
    valid = torch.isfinite(depth) & (depth > 0.0)
    valid_fraction = float(valid.to(dtype=torch.float32).mean())
    assert valid_fraction == 1.0, valid_fraction
    return valid_fraction


def main() -> None:
    require_readiness_gpu()
    simulation = SimulationContext(make_readiness_simulation_cfg(DT_S))
    scene_cfg = ZedXMiniGoldenSceneCfg(
        num_envs=1,
        env_spacing=3.0,
        replicate_physics=False,
    )
    scene_cfg.golden_robot.spawn.articulation_props = (
        PhysxArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        )
    )
    scene = InteractiveScene(scene_cfg)
    stage = omni.usd.get_context().get_stage()
    robot_path = "/World/envs/env_0/Robot"
    zed_path = "/World/envs/env_0/ZED_X_Mini"
    cameras = (scene["zed_left"], scene["zed_right"])
    joint_path = author_alex_purdue_zed_x_mini_mount(stage, robot_path, zed_path)
    simulation.reset()

    physics_sim_view = PhysxManager.get_physics_sim_view()
    zed_body_view = physics_sim_view.create_rigid_body_view(zed_path)
    assert zed_body_view is not None and zed_body_view.count == 1

    robot = scene["golden_robot"]
    assert set(("NECK_Z", "NECK_Y")).issubset(robot.data.joint_names)
    zero = torch.zeros_like(robot.data.joint_pos.torch)
    robot.write_joint_state_to_sim_index(
        position=zero,
        velocity=torch.zeros_like(robot.data.joint_vel.torch),
    )
    robot.reset()
    robot.set_joint_position_target_index(target=zero)
    _step(simulation, scene, cameras, WARMUP_STEPS)

    model_prim = stage.GetPrimAtPath(f"{zed_path}/base_link/{ZED_X_MINI_MODEL}")
    assert (
        model_prim.GetVariantSets().GetVariantSet("lens").GetVariantSelection()
        == ZED_X_MINI_LENS
    )
    mass = float(UsdPhysics.MassAPI(stage.GetPrimAtPath(zed_path)).GetMassAttr().Get())
    assert abs(mass - ZED_X_MINI_MASS_KG) <= 1.0e-6
    inertia = tuple(
        float(value)
        for value in UsdPhysics.MassAPI(stage.GetPrimAtPath(zed_path))
        .GetDiagonalInertiaAttr()
        .Get()
    )
    assert all(
        abs(actual - expected) <= 1.0e-9
        for actual, expected in zip(
            inertia, ZED_X_MINI_DIAGONAL_INERTIA_KG_M2, strict=True
        )
    )
    left_position = cameras[0].data.pos_w.torch
    right_position = cameras[1].data.pos_w.torch
    baseline = float(
        torch.linalg.vector_norm(
            left_position - right_position,
            dim=-1,
        ).max()
    )
    assert abs(baseline - EXPECTED_BASELINE_M) <= 1.0e-4, baseline

    initial_relative_position, initial_relative_quaternion = _relative_zed_pose(
        scene, zed_body_view
    )
    joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath(joint_path))
    joint_position = torch.tensor(
        tuple(joint.GetLocalPos0Attr().Get()), device="cuda:0", dtype=torch.float32
    ).unsqueeze(0)
    joint_rotation = joint.GetLocalRot0Attr().Get()
    joint_quaternion = torch.tensor(
        (*tuple(joint_rotation.GetImaginary()), float(joint_rotation.GetReal())),
        device="cuda:0",
        dtype=torch.float32,
    ).unsqueeze(0)
    initial_position_error = float(
        torch.linalg.vector_norm(
            initial_relative_position - joint_position, dim=-1
        ).max()
    )
    initial_rotation_error = float(
        quat_error_magnitude(initial_relative_quaternion, joint_quaternion).max()
    )
    assert initial_position_error <= POSE_TOLERANCE_M, initial_position_error
    assert initial_rotation_error <= POSE_TOLERANCE_RAD, initial_rotation_error

    initial_left, initial_right, initial_depth = _camera_tensors(cameras)
    initial_left = initial_left.clone()
    initial_right = initial_right.clone()
    initial_depth = initial_depth.clone()
    initial_valid_fraction = _assert_images(initial_left, initial_right, initial_depth)

    target = zero.clone()
    by_name = {name: index for index, name in enumerate(robot.data.joint_names)}
    initial_neck = robot.data.joint_pos.torch[
        :, [by_name["NECK_Z"], by_name["NECK_Y"]]
    ].clone()
    home_error = float(torch.abs(initial_neck).max())
    assert home_error <= 0.05, home_error
    target[:, by_name["NECK_Z"]] = 0.08
    target[:, by_name["NECK_Y"]] = 0.05
    maximum_position_error = 0.0
    maximum_rotation_error = 0.0
    for index in range(MOVE_STEPS):
        fraction = float(index + 1) / MOVE_STEPS
        commanded = zero + fraction * (target - zero)
        robot.write_joint_state_to_sim_index(
            position=commanded,
            velocity=torch.zeros_like(robot.data.joint_vel.torch),
        )
        robot.set_joint_position_target_index(target=commanded)
        _step(simulation, scene, cameras, 1)
        assert_finite_joint_state(robot)
        relative_position, relative_quaternion = _relative_zed_pose(
            scene, zed_body_view
        )
        if (
            not torch.isfinite(relative_position).all()
            or not torch.isfinite(relative_quaternion).all()
        ):
            raise RuntimeError(
                f"non-finite ZED fixed-joint pose at neck-motion step {index}"
            )
        maximum_position_error = max(
            maximum_position_error,
            float(
                torch.linalg.vector_norm(
                    relative_position - initial_relative_position, dim=-1
                ).max()
            ),
        )
        maximum_rotation_error = max(
            maximum_rotation_error,
            float(
                quat_error_magnitude(
                    relative_quaternion, initial_relative_quaternion
                ).max()
            ),
        )
    assert maximum_position_error <= POSE_TOLERANCE_M, maximum_position_error
    assert maximum_rotation_error <= POSE_TOLERANCE_RAD, maximum_rotation_error
    neck_error = float(
        torch.max(
            torch.abs(
                robot.data.joint_pos.torch[:, [by_name["NECK_Z"], by_name["NECK_Y"]]]
                - target[:, [by_name["NECK_Z"], by_name["NECK_Y"]]]
            )
        )
    )
    assert neck_error <= 0.05, (
        neck_error,
        robot.data.joint_pos.torch[:, [by_name["NECK_Z"], by_name["NECK_Y"]]],
    )
    neck_movement = float(
        torch.abs(
            robot.data.joint_pos.torch[:, [by_name["NECK_Z"], by_name["NECK_Y"]]]
            - initial_neck
        ).max()
    )
    assert neck_movement >= 0.02, neck_movement

    moved_left, moved_right, moved_depth = _camera_tensors(cameras)
    moved_valid_fraction = _assert_images(moved_left, moved_right, moved_depth)
    image_change = float(
        torch.mean(
            torch.abs(
                moved_left[..., :3].to(dtype=torch.float32)
                - initial_left[..., :3].to(dtype=torch.float32)
            )
        )
    )
    assert image_change > 0.5, image_change
    print(
        "PASS: official ZED X Mini Wide validated on Golden Robot with "
        f"PhysX/TGS and {torch.cuda.get_device_name(0)} (cuda:0); "
        f"model={ZED_X_MINI_MODEL}, lens={ZED_X_MINI_LENS}, "
        f"resolution=960x600, baseline={baseline:.6f} m, mass={mass:.6f} kg, "
        f"depth_valid=[{initial_valid_fraction:.3%}, {moved_valid_fraction:.3%}], "
        f"rigid_error=[{maximum_position_error:.8f} m, "
        f"{maximum_rotation_error:.8f} rad], neck_error={neck_error:.6f} rad, "
        f"neck_movement={neck_movement:.6f} rad, "
        f"home_error={home_error:.6f} rad, "
        f"image_change={image_change:.3f}",
        flush=True,
    )


run_gate(main, simulation_app)
