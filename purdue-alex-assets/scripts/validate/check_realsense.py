#!/usr/bin/env python3
# Copyright (c) 2026, Patrizio Acquadro.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX/TGS CUDA RGB-D and modular-mount gate for D405 and D435."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run_sequential_workers() -> int:
    """Run each camera in a fresh Isaac process to bound GPU and stage state."""

    summaries: list[str] = []
    for model in ("d405", "d435"):
        completed = subprocess.run(
            [sys.executable, Path(__file__).resolve().as_posix(), "--model", model],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=False,
        )
        output = completed.stdout + completed.stderr
        print(output, end="", flush=True)
        marker = f"PASS: {model.upper()} "
        pass_lines = [line for line in output.splitlines() if line.startswith(marker)]
        if (
            completed.returncode != 0
            or len(pass_lines) != 1
            or "Traceback (most recent call last)" in output
        ):
            print(
                f"NO-GO: sequential RealSense worker failed for {model.upper()} "
                f"(exit={completed.returncode})",
                flush=True,
            )
            return 1
        summaries.append(pass_lines[0].removeprefix("PASS: "))
    print(
        "PASS: sequential official RealSense D405 and D435 workers completed; "
        + "; ".join(summaries),
        flush=True,
    )
    return 0


if __name__ == "__main__" and "--model" not in sys.argv:
    raise SystemExit(_run_sequential_workers())

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", choices=("d405", "d435"), required=True)
worker_args, remaining_args = parser.parse_known_args()
WORKER_MODEL = worker_args.model
sys.argv = [sys.argv[0], *remaining_args]

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=True, device="cuda:0", enable_cameras=True)
simulation_app = app_launcher.app

import omni.usd  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

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
from ihmc_alex_isaaclab.sensors.realsense import (  # noqa: E402
    D405_RESOLUTION,
    D435_RESOLUTION,
    author_realsense_mount,
    make_realsense_d405_cfgs,
    make_realsense_d435_cfgs,
)
from _physx import (  # noqa: E402
    apply_gravity_compensation,
    assert_finite_joint_state,
    make_readiness_simulation_cfg,
    make_static_box_cfg,
    require_readiness_gpu,
    run_gate,
)

MODEL_SPECIFICATIONS = {
    "d405": (make_realsense_d405_cfgs, D405_RESOLUTION, (0.18, -0.09, 0.02)),
    "d435": (make_realsense_d435_cfgs, D435_RESOLUTION, (0.18, 0.09, 0.02)),
}
MODEL_FACTORY, MODEL_RESOLUTION, MOUNT_POSITION = MODEL_SPECIFICATIONS[WORKER_MODEL]
MODEL_NAME = WORKER_MODEL.upper()

DT_S = 0.005
WARMUP_STEPS = 90
MOVE_STEPS = 20
POSE_TOLERANCE_M = 1.0e-4
POSE_TOLERANCE_RAD = 1.0e-4
MODEL_SUBPATH = "Geometry/base_link/camera_bottom_screw_frame/camera_link"
COLOR_OPTICAL_SUBPATH = f"{MODEL_SUBPATH}/camera_color_frame/camera_color_optical_frame"


@configclass
class RealSenseGoldenSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=1800.0, color=(0.78, 0.82, 0.90)),
    )
    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.DistantLightCfg(intensity=3200.0, color=(1.0, 0.94, 0.84)),
        init_state=AssetBaseCfg.InitialStateCfg(
            rot=(0.2209424, -0.2209424, 0.2209424, 0.9238795)
        ),
    )
    backdrop = make_static_box_cfg(
        "{ENV_REGEX_NS}/Backdrop",
        size=(0.02, 4.0, 4.0),
        position=(0.82, 0.0, 1.45),
        color=(0.06, 0.16, 0.72),
    )
    alignment_target = make_static_box_cfg(
        "{ENV_REGEX_NS}/AlignmentTarget",
        size=(0.02, 0.20, 0.20),
        position=(0.63, 0.0, 0.64),
        color=(0.92, 0.04, 0.02),
    )
    golden_robot = make_alex_purdue_cfg(
        fix_base=True,
        variant="full_convex",
        end_effector="wsg32_umi_v1",
    )
    golden_robot.prim_path = "{ENV_REGEX_NS}/Robot"
    sensor_model, sensor_camera = MODEL_FACTORY(
        "{ENV_REGEX_NS}/RealSense",
        spawn_init_pos=(0.1, 0.0, 0.55),
    )
    sensor_model.spawn.force_usd_conversion = True


def _step(
    simulation: SimulationContext,
    scene: InteractiveScene,
    camera: Camera,
    count: int,
) -> None:
    follower_names = {follower for follower, _ in ALEX_PURDUE_WSG32_MIMIC_JOINT_PAIRS}
    for _ in range(count):
        robot = scene["golden_robot"]
        apply_gravity_compensation(robot, excluded_joint_names=follower_names)
        scene.write_data_to_sim()
        simulation.step()
        scene.update(DT_S)
        camera.update(DT_S)


def _relative_pose(
    scene: InteractiveScene, body_view: object
) -> tuple[torch.Tensor, torch.Tensor]:
    robot = scene["golden_robot"]
    head_index = robot.data.body_names.index("HEAD_LINK")
    head_pose = robot.data.body_pose_w.torch[:, head_index]
    sensor_pose = wp.to_torch(body_view.get_transforms())
    if tuple(sensor_pose.shape) != (1, 7):
        raise RuntimeError(
            f"unexpected RealSense rigid pose shape: {sensor_pose.shape}"
        )
    return subtract_frame_transforms(
        head_pose[:, :3],
        head_pose[:, 3:7],
        sensor_pose[:, :3],
        sensor_pose[:, 3:7],
    )


def _bounding_box(mask: torch.Tensor) -> tuple[int, int, int, int]:
    rows, columns = torch.where(mask)
    if rows.numel() == 0:
        raise RuntimeError("alignment mask is empty")
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()),
        int(rows.max()),
    )


def _assert_aligned_images(
    model: str,
    camera: Camera,
    resolution: tuple[int, int],
    *,
    require_edge_alignment: bool = True,
) -> tuple[float, float, tuple[int, int, int, int], tuple[int, int, int, int]]:
    output = camera.data.output
    if output is None:
        raise RuntimeError(f"{model} camera outputs were not allocated")
    try:
        rgb = output["rgb"].torch
        depth = output["distance_to_image_plane"].torch
    except KeyError as error:
        raise RuntimeError(f"{model} camera output is missing: {error}") from error

    width, height = resolution
    assert rgb.is_cuda and depth.is_cuda
    assert rgb.device.index == depth.device.index == 0
    assert tuple(rgb.shape) == (1, height, width, 3), tuple(rgb.shape)
    assert tuple(depth.shape) == (1, height, width, 1), tuple(depth.shape)
    rgb_float = rgb.to(dtype=torch.float32)
    assert torch.isfinite(rgb_float).all()
    rgb_std = float(rgb_float.std())
    assert rgb_std > 8.0, rgb_std

    depth_image = depth[0, ..., 0]
    valid = torch.isfinite(depth_image) & (depth_image > 0.0)
    valid_fraction = float(valid.to(dtype=torch.float32).mean())
    assert valid_fraction > 0.50, valid_fraction

    color = rgb_float[0]
    red_mask = (
        (color[..., 0] > 80.0)
        & (color[..., 0] > 1.6 * color[..., 1])
        & (color[..., 0] > 1.6 * color[..., 2])
        & valid
    )
    red_pixels = int(red_mask.sum())
    if red_pixels <= 1000:
        raise AssertionError(
            f"{model} alignment target is missing: red_pixels={red_pixels}, "
            f"camera_pos={tuple(camera.data.pos_w.torch[0])}"
        )
    target_depth = torch.median(depth_image[red_mask])
    depth_mask = valid & (torch.abs(depth_image - target_depth) < 0.025)
    color_box = _bounding_box(red_mask)
    depth_box = _bounding_box(depth_mask)
    edge_error_px = max(
        abs(color_edge - depth_edge)
        for color_edge, depth_edge in zip(color_box, depth_box, strict=True)
    )
    if require_edge_alignment:
        assert edge_error_px <= 1, (color_box, depth_box, edge_error_px)
    return valid_fraction, rgb_std, color_box, depth_box


def _assert_converted_asset(
    stage: Usd.Stage, model: str, sensor_path: str, usd_path: Path
) -> None:
    assert usd_path.is_file(), usd_path
    root = stage.GetPrimAtPath(sensor_path)
    assert root and root.IsValid()
    required = (
        "Geometry/base_link/camera_bottom_screw_frame",
        MODEL_SUBPATH,
        f"{MODEL_SUBPATH}/camera_depth_frame/camera_depth_optical_frame",
        f"{MODEL_SUBPATH}/camera_infra1_frame/camera_infra1_optical_frame",
        f"{MODEL_SUBPATH}/camera_infra2_frame/camera_infra2_optical_frame",
        COLOR_OPTICAL_SUBPATH,
    )
    for relative_path in required:
        prim = stage.GetPrimAtPath(f"{sensor_path}/{relative_path}")
        assert prim and prim.IsValid(), f"{model}: missing {relative_path}"
    camera = stage.GetPrimAtPath(f"{sensor_path}/{COLOR_OPTICAL_SUBPATH}/Camera")
    assert camera and camera.IsA(UsdGeom.Camera)
    meshes = [
        prim
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
        if prim.IsA(UsdGeom.Mesh)
    ]
    assert meshes, f"{model}: converted official mesh is missing"


def main() -> None:
    require_readiness_gpu()
    simulation = SimulationContext(make_readiness_simulation_cfg(DT_S))
    scene_cfg = RealSenseGoldenSceneCfg(
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
    robot = stage.GetPrimAtPath(robot_path)
    head_paths = [
        prim.GetPath().pathString
        for prim in Usd.PrimRange(robot)
        if prim.GetName() == "HEAD_LINK"
    ]
    assert len(head_paths) == 1, head_paths
    head_path = head_paths[0]
    sensor_path = "/World/envs/env_0/RealSense"
    body_path = f"{sensor_path}/{MODEL_SUBPATH}"
    joint_path = author_realsense_mount(
        stage,
        head_path,
        sensor_path,
        pos=MOUNT_POSITION,
    )
    simulation.reset()

    physics_sim_view = PhysxManager.get_physics_sim_view()
    body_view = physics_sim_view.create_rigid_body_view(body_path)
    assert body_view is not None and body_view.count == 1
    camera = scene["sensor_camera"]
    robot_asset = scene["golden_robot"]
    zero = torch.zeros_like(robot_asset.data.joint_pos.torch)
    robot_asset.write_joint_state_to_sim_index(
        position=zero,
        velocity=torch.zeros_like(robot_asset.data.joint_vel.torch),
    )
    robot_asset.reset()
    robot_asset.set_joint_position_target_index(target=zero)
    _step(simulation, scene, camera, WARMUP_STEPS)

    initial_relative = _relative_pose(scene, body_view)
    initial_rgb = camera.data.output["rgb"].torch.clone()
    model_cfg = scene_cfg.sensor_model
    usd_path = Path(model_cfg.spawn.usd_dir) / model_cfg.spawn.usd_file_name
    _assert_converted_asset(stage, MODEL_NAME, sensor_path, usd_path)
    initial_image_metrics = _assert_aligned_images(MODEL_NAME, camera, MODEL_RESOLUTION)

    by_name = {name: index for index, name in enumerate(robot_asset.data.joint_names)}
    assert set(("NECK_Z", "NECK_Y")).issubset(by_name)
    target = zero.clone()
    target[:, by_name["NECK_Z"]] = 0.08
    target[:, by_name["NECK_Y"]] = 0.05
    maximum_errors = [0.0, 0.0]
    for step in range(MOVE_STEPS):
        fraction = float(step + 1) / MOVE_STEPS
        commanded = zero + fraction * (target - zero)
        robot_asset.write_joint_state_to_sim_index(
            position=commanded,
            velocity=torch.zeros_like(robot_asset.data.joint_vel.torch),
        )
        robot_asset.set_joint_position_target_index(target=commanded)
        _step(simulation, scene, camera, 1)
        position, rotation = _relative_pose(scene, body_view)
        initial_position, initial_rotation = initial_relative
        position_error = float(
            torch.linalg.vector_norm(position - initial_position, dim=-1).max()
        )
        rotation_error = float(quat_error_magnitude(rotation, initial_rotation).max())
        maximum_errors[0] = max(maximum_errors[0], position_error)
        maximum_errors[1] = max(maximum_errors[1], rotation_error)

    position_error, rotation_error = maximum_errors
    assert position_error <= POSE_TOLERANCE_M, (MODEL_NAME, position_error)
    assert rotation_error <= POSE_TOLERANCE_RAD, (MODEL_NAME, rotation_error)
    valid_fraction, rgb_std, _, _ = _assert_aligned_images(
        MODEL_NAME,
        camera,
        MODEL_RESOLUTION,
        require_edge_alignment=False,
    )
    color_box, depth_box = initial_image_metrics[2:]
    moved_rgb = camera.data.output["rgb"].torch
    image_change = float(
        torch.mean(
            torch.abs(
                moved_rgb.to(dtype=torch.float32) - initial_rgb.to(dtype=torch.float32)
            )
        )
    )
    assert image_change > 0.5, (MODEL_NAME, image_change)
    joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath(joint_path))
    assert joint.GetBody0Rel().GetTargets() == [
        stage.GetPrimAtPath(head_path).GetPath()
    ]
    assert joint.GetBody1Rel().GetTargets() == [
        stage.GetPrimAtPath(body_path).GetPath()
    ]

    assert_finite_joint_state(robot_asset)
    print(
        f"PASS: {MODEL_NAME} official URDF->USD asset validated with aligned RGB-D "
        f"and modular mount on PhysX/TGS {torch.cuda.get_device_name(0)} (cuda:0); "
        f"resolution={MODEL_RESOLUTION[0]}x{MODEL_RESOLUTION[1]},"
        f"depth_valid={valid_fraction:.3%},rgb_std={rgb_std:.3f},"
        f"boxes={color_box}/{depth_box},image_change={image_change:.3f},"
        f"rigid_error={position_error:.8f}m/{rotation_error:.8f}rad",
        flush=True,
    )


run_gate(main, simulation_app)
