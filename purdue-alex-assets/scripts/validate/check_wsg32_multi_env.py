#!/usr/bin/env python3
# Copyright (c) 2026, Patrizio Acquadro.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX/TGS CUDA 128-environment, 1000-step WSG32 + UMI soak gate."""

from __future__ import annotations

from pathlib import Path

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, device="cuda:0")
simulation_app = app_launcher.app

import torch  # noqa: E402

from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils.configclass import configclass  # noqa: E402

from ihmc_alex_isaaclab.end_effectors.weiss_wsg32 import (  # noqa: E402
    WSG32_JAW_STROKE_M,
    make_wsg32_umi_v1_cfg,
)
from _physx import (  # noqa: E402
    assert_finite_joint_state,
    make_readiness_simulation_cfg,
    require_readiness_gpu,
    run_gate,
    set_joint_targets,
)

DT_S = 0.005
NUM_ENVS = 128
SOAK_STEPS = 1000
WARMUP_STEPS = 100
OUTPUT_DIR = (
    Path(__file__).resolve().parents[2] / "build" / "validation" / "wsg32_multi_env"
)


@configclass
class WSG32SoakSceneCfg(InteractiveSceneCfg):
    wsg = make_wsg32_umi_v1_cfg()


def main() -> None:
    require_readiness_gpu()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    simulation = SimulationContext(make_readiness_simulation_cfg(DT_S))
    scene_cfg = WSG32SoakSceneCfg(num_envs=NUM_ENVS, env_spacing=0.5)
    scene_cfg.wsg.spawn.usd_dir = (OUTPUT_DIR / "usd").as_posix()
    scene = InteractiveScene(scene_cfg)
    wsg = scene["wsg"]
    simulation.reset()

    assert wsg.data.joint_pos.torch.shape[0] == NUM_ENVS
    by_name = {name: index for index, name in enumerate(wsg.data.joint_names)}
    leader = by_name["WSG32_JAW_OPENING"]
    follower = by_name["WSG32_JAW_FOLLOWER"]
    open_state = torch.full_like(wsg.data.joint_pos.torch, WSG32_JAW_STROKE_M)
    zero_velocity = torch.zeros_like(wsg.data.joint_vel.torch)
    wsg.write_joint_state_to_sim_index(
        position=open_state,
        velocity=zero_velocity,
    )
    wsg.reset()
    set_joint_targets(wsg, {"WSG32_JAW_OPENING": WSG32_JAW_STROKE_M})
    for _ in range(WARMUP_STEPS):
        scene.write_data_to_sim()
        simulation.step()
        scene.update(DT_S)
    assert_finite_joint_state(wsg, 2.0e-4)

    minimum = float(wsg.data.joint_pos.torch[:, leader].min())
    maximum = float(wsg.data.joint_pos.torch[:, leader].max())
    max_mimic_error = 0.0
    for step in range(SOAK_STEPS):
        phase = (step // 250) % 2
        set_joint_targets(
            wsg, {"WSG32_JAW_OPENING": WSG32_JAW_STROKE_M if phase else 0.0}
        )
        scene.write_data_to_sim()
        simulation.step()
        scene.update(DT_S)

        position = wsg.data.joint_pos.torch
        assert_finite_joint_state(wsg, 2.0e-4)
        opening = position[:, leader]
        minimum = min(minimum, float(opening.min()))
        maximum = max(maximum, float(opening.max()))
        max_mimic_error = max(
            max_mimic_error,
            float(torch.max(torch.abs(position[:, follower] - opening))),
        )

    assert minimum >= -2.0e-4, minimum
    assert maximum <= WSG32_JAW_STROKE_M + 2.0e-4, maximum
    assert minimum <= 0.002, minimum
    assert maximum >= WSG32_JAW_STROKE_M - 0.002, maximum
    assert maximum - minimum >= 0.032
    assert max_mimic_error <= 1.0e-4, max_mimic_error
    print(
        "PASS: WSG32 + UMI v1 PhysX/TGS multi-environment soak on "
        f"{torch.cuda.get_device_name(0)} (cuda:0); envs={NUM_ENVS}, "
        f"steps={SOAK_STEPS}, sweep=[{minimum:.6f}, {maximum:.6f}] m/jaw, "
        f"max_mimic_error={max_mimic_error:.8f} m, output={OUTPUT_DIR}",
        flush=True,
    )


run_gate(main, simulation_app)
