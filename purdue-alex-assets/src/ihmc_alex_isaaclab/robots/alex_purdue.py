# Copyright (c) 2026, Patrizio Acquadro.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generic Isaac Lab configuration for the fixed-base IHMC Alex Purdue robot."""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyBaseCfg

from ..end_effectors.weiss_wsg32 import (
    WSG32_MAX_FINGER_SPEED_M_S,
    WSG32_MAX_FORCE_PER_JAW_N,
)
from .purdue_frames import (
    ALEX_PURDUE_ASSET_ROOT,
    AlexPurdueEndEffector,
    AlexPurdueVariant,
    _resolve_alex_purdue_urdf,
)
from .purdue_physics import (
    spawn_alex_purdue_full_convex,
    spawn_alex_purdue_wsg32_full_convex,
    spawn_alex_purdue_wsg32_source,
)

_EFFORT_LIMITS = {
    "neck": 20.86,
    "shoulder_y": 160.7,
    "shoulder_x": 160.7,
    "shoulder_z": 71.28,
    "elbow_y": 71.28,
    "wrist": 20.86,
    "terminal_gripper": 20.86,
    "ezgripper": 1.0,
    "wsg32": WSG32_MAX_FORCE_PER_JAW_N,
}

_VELOCITY_LIMITS = {
    "neck": 17.3,
    "shoulder_y": 10.38,
    "shoulder_x": 10.38,
    "shoulder_z": 4.47,
    "elbow_y": 4.47,
    "wrist": 17.3,
    "terminal_gripper": 17.3,
    "ezgripper": 3.67,
    "wsg32": WSG32_MAX_FINGER_SPEED_M_S,
}


def _pd(
    joint_names_expr: list[str],
    *,
    stiffness: dict[str, float] | float,
    damping: dict[str, float] | float,
    velocity_limit_sim: dict[str, float] | float,
    effort_limit_sim: dict[str, float] | float,
    armature: float,
) -> IdealPDActuatorCfg:
    return IdealPDActuatorCfg(
        joint_names_expr=joint_names_expr,
        stiffness=stiffness,
        damping=damping,
        velocity_limit=velocity_limit_sim,
        effort_limit=effort_limit_sim,
        velocity_limit_sim=velocity_limit_sim,
        effort_limit_sim=effort_limit_sim,
        armature=armature,
    )


_ALEX_PURDUE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=(
            ALEX_PURDUE_ASSET_ROOT
            / "urdf"
            / "baseline"
            / "alex_purdue_full_convex.urdf"
        ).as_posix(),
        fix_base=True,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        collision_from_visuals=False,
        collision_type="Convex Hull",
        self_collision=True,
        make_instanceable=False,
        func=spawn_alex_purdue_full_convex,
        activate_contact_sensors=True,
        rigid_props=RigidBodyBaseCfg(
            disable_gravity=False,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0, damping=0.0
            ),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "neck": _pd(
            ["NECK_Z", "NECK_Y"],
            stiffness=5.0,
            damping=1.0,
            velocity_limit_sim=_VELOCITY_LIMITS["neck"],
            effort_limit_sim=_EFFORT_LIMITS["neck"],
            armature=0.01,
        ),
        "arms": _pd(
            [
                ".*SHOULDER_Y",
                ".*SHOULDER_X",
                ".*SHOULDER_Z",
                ".*ELBOW_Y",
                ".*WRIST_Z",
                ".*WRIST_X",
                ".*GRIPPER_Y",
            ],
            stiffness={
                ".*SHOULDER_Y": 26.78,
                ".*SHOULDER_X": 26.78,
                ".*SHOULDER_Z": 23.76,
                ".*ELBOW_Y": 23.76,
                ".*WRIST_Z": 5.0,
                ".*WRIST_X": 5.0,
                ".*GRIPPER_Y": 2.0,
            },
            damping={
                ".*SHOULDER_Y": 8.0,
                ".*SHOULDER_X": 8.0,
                ".*SHOULDER_Z": 4.0,
                ".*ELBOW_Y": 4.0,
                ".*WRIST_Z": 1.0,
                ".*WRIST_X": 1.0,
                ".*GRIPPER_Y": 0.5,
            },
            velocity_limit_sim={
                ".*SHOULDER_Y": _VELOCITY_LIMITS["shoulder_y"],
                ".*SHOULDER_X": _VELOCITY_LIMITS["shoulder_x"],
                ".*SHOULDER_Z": _VELOCITY_LIMITS["shoulder_z"],
                ".*ELBOW_Y": _VELOCITY_LIMITS["elbow_y"],
                ".*WRIST_Z": _VELOCITY_LIMITS["wrist"],
                ".*WRIST_X": _VELOCITY_LIMITS["wrist"],
                ".*GRIPPER_Y": _VELOCITY_LIMITS["terminal_gripper"],
            },
            effort_limit_sim={
                ".*SHOULDER_Y": _EFFORT_LIMITS["shoulder_y"],
                ".*SHOULDER_X": _EFFORT_LIMITS["shoulder_x"],
                ".*SHOULDER_Z": _EFFORT_LIMITS["shoulder_z"],
                ".*ELBOW_Y": _EFFORT_LIMITS["elbow_y"],
                ".*WRIST_Z": _EFFORT_LIMITS["wrist"],
                ".*WRIST_X": _EFFORT_LIMITS["wrist"],
                ".*GRIPPER_Y": _EFFORT_LIMITS["terminal_gripper"],
            },
            armature=0.01,
        ),
        "ezgrippers": _pd(
            [".*ezgripper_knuckle_palm_l1_1", ".*ezgripper_knuckle_l1_l2_1"],
            stiffness=2.0,
            damping=0.1,
            velocity_limit_sim=_VELOCITY_LIMITS["ezgripper"],
            effort_limit_sim=_EFFORT_LIMITS["ezgripper"],
            armature=0.01,
        ),
    },
)


def make_alex_purdue_cfg(
    asset_path: str | os.PathLike[str] | None = None,
    *,
    fix_base: bool = True,
    variant: AlexPurdueVariant = "full_convex",
    end_effector: AlexPurdueEndEffector = "sake_ezgripper",
) -> ArticulationCfg:
    """Return an independent Alex Purdue configuration bound to one URDF profile."""

    path = _resolve_alex_purdue_urdf(asset_path, variant, end_effector)
    cfg = _ALEX_PURDUE_CFG.copy()
    cfg.spawn.asset_path = path.as_posix()
    cfg.spawn.fix_base = fix_base
    if end_effector == "wsg32_umi_v1":
        cfg.actuators.pop("ezgrippers")
        cfg.actuators["wsg32"] = _pd(
            [".*_WSG32_JAW_OPENING"],
            stiffness=2000.0,
            damping=60.0,
            velocity_limit_sim=_VELOCITY_LIMITS["wsg32"],
            effort_limit_sim=_EFFORT_LIMITS["wsg32"],
            armature=0.001,
        )
        cfg.spawn.func = (
            spawn_alex_purdue_wsg32_source
            if variant == "source"
            else spawn_alex_purdue_wsg32_full_convex
        )
    if variant == "source":
        cfg.spawn.self_collision = False
        cfg.spawn.make_instanceable = False
        if end_effector == "sake_ezgripper":
            cfg.spawn.func = sim_utils.spawn_from_urdf
    return cfg


__all__ = [
    "AlexPurdueEndEffector",
    "make_alex_purdue_cfg",
]
