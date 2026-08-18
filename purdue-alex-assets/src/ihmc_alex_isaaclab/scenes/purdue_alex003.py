# Copyright (c) 2026, Patrizio Acquadro.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generic measured Purdue Alex003 reference composition for Isaac Lab."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from ..platforms.purdue_alex003_pedestal import (
    load_purdue_alex003_pedestal_spec,
    make_purdue_alex003_pedestal_cfg,
)
from ..robots.alex_purdue import make_alex_purdue_cfg
from ..robots.purdue_frames import AlexPurdueVariant


@configclass
class _PurdueAlex003ReferenceSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    pedestal = make_purdue_alex003_pedestal_cfg()
    robot = make_alex_purdue_cfg()
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1800.0,
            color=(0.75, 0.78, 0.82),
        ),
    )


def make_purdue_alex003_reference_scene_cfg(
    *,
    num_envs: int = 1,
    env_spacing: float = 2.0,
    robot_variant: AlexPurdueVariant = "full_convex",
) -> InteractiveSceneCfg:
    """Return an independent floor, pedestal, and fixed-base Alex composition."""

    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
        raise ValueError("num_envs must be a positive integer")
    if (
        isinstance(env_spacing, bool)
        or not isinstance(env_spacing, (int, float))
        or not math.isfinite(float(env_spacing))
        or float(env_spacing) <= 0.0
    ):
        raise ValueError("env_spacing must be a positive finite number")

    spec = load_purdue_alex003_pedestal_spec()
    cfg = _PurdueAlex003ReferenceSceneCfg(
        num_envs=num_envs,
        env_spacing=float(env_spacing),
    )
    cfg.pedestal = make_purdue_alex003_pedestal_cfg()
    cfg.robot = make_alex_purdue_cfg(fix_base=True, variant=robot_variant)
    cfg.robot.prim_path = "{ENV_REGEX_NS}/Robot"
    cfg.robot.init_state.pos = (0.0, 0.0, spec.alex_root_world_z_m)
    cfg.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)
    return cfg
