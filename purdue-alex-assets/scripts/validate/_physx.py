# Copyright (c) 2026, Patrizio Acquadro.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared PhysX readiness contract."""

from __future__ import annotations

from typing import Any


def require_readiness_gpu() -> str:
    import torch

    if not torch.cuda.is_available() or torch.cuda.current_device() != 0:
        raise RuntimeError("PhysX readiness requires the visible CUDA device cuda:0")
    device_name = torch.cuda.get_device_name(0)
    if "4090" not in device_name:
        raise RuntimeError(f"expected RTX 4090 on cuda:0, got {device_name!r}")
    return device_name


def make_readiness_simulation_cfg(dt_s: float) -> Any:
    from isaaclab.sim import SimulationCfg
    from isaaclab_physx.physics import PhysxCfg

    return SimulationCfg(
        dt=dt_s,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=PhysxCfg(
            solver_type=1,
            solve_articulation_contact_last=True,
            enable_enhanced_determinism=True,
            enable_external_forces_every_iteration=True,
            min_position_iteration_count=8,
            min_velocity_iteration_count=4,
        ),
    )


def apply_gravity_compensation(
    articulation: Any, *, excluded_joint_names: set[str] | frozenset[str] = frozenset()
) -> None:
    import torch

    controlled_ids = [
        index
        for index, name in enumerate(articulation.data.joint_names)
        if name not in excluded_joint_names
    ]
    gravity = articulation.data.gravity_compensation_forces.torch[
        :, articulation.num_base_dofs :
    ][:, controlled_ids]
    assert torch.isfinite(gravity).all()
    articulation.set_joint_effort_target_index(target=gravity, joint_ids=controlled_ids)


def rigid_body_path(
    stage: Any,
    root_path: str,
    body_name: str,
    *,
    traverse_instances: bool = False,
) -> str:
    from pxr import Usd, UsdPhysics

    root = stage.GetPrimAtPath(root_path)
    iterator = (
        Usd.PrimRange(root, Usd.TraverseInstanceProxies())
        if traverse_instances
        else Usd.PrimRange(root)
    )
    matches = [
        prim
        for prim in iterator
        if prim.GetName() == body_name and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    assert len(matches) == 1, (root_path, body_name, matches)
    return str(matches[0].GetPath())


def set_joint_targets(articulation: Any, values: dict[str, float]) -> Any:
    import torch

    target = torch.zeros_like(articulation.data.joint_pos.torch)
    by_name = {name: index for index, name in enumerate(articulation.data.joint_names)}
    missing = sorted(set(values) - set(by_name))
    assert not missing, missing
    for name, value in values.items():
        target[:, by_name[name]] = value
    articulation.set_joint_position_target_index(target=target)
    return target


def assert_finite_joint_state(
    articulation: Any, tolerance: float | None = None
) -> None:
    import torch

    position = articulation.data.joint_pos.torch
    velocity = articulation.data.joint_vel.torch
    limits = articulation.data.joint_pos_limits.torch
    assert torch.isfinite(position).all() and torch.isfinite(velocity).all()
    if tolerance is None:
        return
    assert torch.all(position >= limits[..., 0] - tolerance)
    assert torch.all(position <= limits[..., 1] + tolerance)


def maximum_contact_force(sensor: Any) -> float:
    import torch

    force = torch.linalg.vector_norm(sensor.data.net_forces_w.torch, dim=-1).max()
    matrix = sensor.data.force_matrix_w
    if matrix is not None:
        force = torch.maximum(
            force, torch.linalg.vector_norm(matrix.torch, dim=-1).max()
        )
    assert torch.isfinite(force)
    return float(force)


def make_static_box_cfg(
    prim_path: str,
    *,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
    color: tuple[float, float, float],
) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=size,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                roughness=0.45,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=position),
    )


def run_gate(main: Any, simulation_app: Any) -> None:
    import traceback

    failed = False
    try:
        main()
    except BaseException:
        traceback.print_exc()
        failed = True
    finally:
        simulation_app.close()
    if failed:
        raise SystemExit(1)
