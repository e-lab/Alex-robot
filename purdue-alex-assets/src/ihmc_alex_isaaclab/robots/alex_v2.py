# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generic Isaac Lab configuration for the IHMC Alex V2 robot."""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyBaseCfg

from .sensor_frames import ALEX_V2_ASSET_ROOT, AlexV2Variant, _resolve_alex_v2_urdf

_EFFORT_LIMITS = {
    "hip_x": 160.7,
    "hip_z": 70.5,
    "hip_y": 217.2,
    "knee_y": 217.2,
    "ankle_y": 193.6,
    "ankle_x": 145.2,
    "spine_z": 160.7,
    "neck": 20.86,
    "shoulder_y": 160.7,
    "shoulder_x": 160.7,
    "shoulder_z": 71.28,
    "elbow_y": 71.28,
    "wrist": 20.86,
    "gripper": 20.86,
}

_VELOCITY_LIMITS = {
    "hip_x": 10.38,
    "hip_z": 10.59,
    "hip_y": 9.3,
    "knee_y": 9.3,
    "ankle_y": 9.72,
    "ankle_x": 9.72,
    "spine_z": 10.38,
    "neck": 17.3,
    "shoulder_y": 10.38,
    "shoulder_x": 10.38,
    "shoulder_z": 4.47,
    "elbow_y": 4.47,
    "wrist": 17.3,
    "gripper": 17.3,
}


def _implicit(
    joint_names_expr: list[str],
    *,
    stiffness: dict[str, float],
    damping: dict[str, float],
    velocity_limit_sim: dict[str, float],
    effort_limit_sim: dict[str, float],
) -> ImplicitActuatorCfg:
    return ImplicitActuatorCfg(
        joint_names_expr=joint_names_expr,
        stiffness=stiffness,
        damping=damping,
        velocity_limit_sim=velocity_limit_sim,
        effort_limit_sim=effort_limit_sim,
        armature={expression: 0.01 for expression in joint_names_expr},
    )


_ALEX_V2_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=(ALEX_V2_ASSET_ROOT / "urdf" / "alex_v2.urdf").as_posix(),
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        collision_from_visuals=False,
        self_collision=True,
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
        pos=(0.0, 0.0, 1.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "legs": _implicit(
            [".*HIP_X", ".*HIP_Z", ".*HIP_Y", ".*KNEE_Y", ".*ANKLE_Y", ".*ANKLE_X"],
            stiffness={
                ".*HIP_X": 80.35,
                ".*HIP_Z": 70.5,
                ".*HIP_Y": 108.6,
                ".*KNEE_Y": 108.6,
                ".*ANKLE_Y": 96.8,
                ".*ANKLE_X": 72.6,
            },
            damping={
                ".*HIP_X": 8.035,
                ".*HIP_Z": 7.05,
                ".*HIP_Y": 10.86,
                ".*KNEE_Y": 10.86,
                ".*ANKLE_Y": 9.68,
                ".*ANKLE_X": 7.26,
            },
            velocity_limit_sim={
                ".*HIP_X": _VELOCITY_LIMITS["hip_x"],
                ".*HIP_Z": _VELOCITY_LIMITS["hip_z"],
                ".*HIP_Y": _VELOCITY_LIMITS["hip_y"],
                ".*KNEE_Y": _VELOCITY_LIMITS["knee_y"],
                ".*ANKLE_Y": _VELOCITY_LIMITS["ankle_y"],
                ".*ANKLE_X": _VELOCITY_LIMITS["ankle_x"],
            },
            effort_limit_sim={
                ".*HIP_X": _EFFORT_LIMITS["hip_x"],
                ".*HIP_Z": _EFFORT_LIMITS["hip_z"],
                ".*HIP_Y": _EFFORT_LIMITS["hip_y"],
                ".*KNEE_Y": _EFFORT_LIMITS["knee_y"],
                ".*ANKLE_Y": _EFFORT_LIMITS["ankle_y"],
                ".*ANKLE_X": _EFFORT_LIMITS["ankle_x"],
            },
        ),
        "torso_head": _implicit(
            ["SPINE_Z", "NECK_Z", "NECK_Y"],
            stiffness={"SPINE_Z": 80.35, "NECK_Z": 5.0, "NECK_Y": 5.0},
            damping={"SPINE_Z": 8.035, "NECK_Z": 1.0, "NECK_Y": 1.0},
            velocity_limit_sim={
                "SPINE_Z": _VELOCITY_LIMITS["spine_z"],
                "NECK_Z": _VELOCITY_LIMITS["neck"],
                "NECK_Y": _VELOCITY_LIMITS["neck"],
            },
            effort_limit_sim={
                "SPINE_Z": _EFFORT_LIMITS["spine_z"],
                "NECK_Z": _EFFORT_LIMITS["neck"],
                "NECK_Y": _EFFORT_LIMITS["neck"],
            },
        ),
        "arms": _implicit(
            [
                ".*SHOULDER_Y",
                ".*SHOULDER_X",
                ".*SHOULDER_Z",
                ".*ELBOW_Y",
                ".*WRIST_Z",
                ".*WRIST_X",
                ".*GRIPPER_Z",
            ],
            stiffness={
                ".*SHOULDER_Y": 26.78,
                ".*SHOULDER_X": 26.78,
                ".*SHOULDER_Z": 23.76,
                ".*ELBOW_Y": 23.76,
                ".*WRIST_Z": 5.0,
                ".*WRIST_X": 5.0,
                ".*GRIPPER_Z": 2.0,
            },
            damping={
                ".*SHOULDER_Y": 8.0,
                ".*SHOULDER_X": 8.0,
                ".*SHOULDER_Z": 4.0,
                ".*ELBOW_Y": 4.0,
                ".*WRIST_Z": 1.0,
                ".*WRIST_X": 1.0,
                ".*GRIPPER_Z": 0.5,
            },
            velocity_limit_sim={
                ".*SHOULDER_Y": _VELOCITY_LIMITS["shoulder_y"],
                ".*SHOULDER_X": _VELOCITY_LIMITS["shoulder_x"],
                ".*SHOULDER_Z": _VELOCITY_LIMITS["shoulder_z"],
                ".*ELBOW_Y": _VELOCITY_LIMITS["elbow_y"],
                ".*WRIST_Z": _VELOCITY_LIMITS["wrist"],
                ".*WRIST_X": _VELOCITY_LIMITS["wrist"],
                ".*GRIPPER_Z": _VELOCITY_LIMITS["gripper"],
            },
            effort_limit_sim={
                ".*SHOULDER_Y": _EFFORT_LIMITS["shoulder_y"],
                ".*SHOULDER_X": _EFFORT_LIMITS["shoulder_x"],
                ".*SHOULDER_Z": _EFFORT_LIMITS["shoulder_z"],
                ".*ELBOW_Y": _EFFORT_LIMITS["elbow_y"],
                ".*WRIST_Z": _EFFORT_LIMITS["wrist"],
                ".*WRIST_X": _EFFORT_LIMITS["wrist"],
                ".*GRIPPER_Z": _EFFORT_LIMITS["gripper"],
            },
        ),
    },
)


def make_alex_v2_cfg(
    asset_path: str | os.PathLike[str] | None = None,
    *,
    fix_base: bool = False,
    variant: AlexV2Variant = "standard",
) -> ArticulationCfg:
    """Return an independent Alex V2 configuration bound to one URDF profile."""

    path = _resolve_alex_v2_urdf(asset_path, variant)
    cfg = _ALEX_V2_CFG.copy()
    cfg.spawn.asset_path = path.as_posix()
    cfg.spawn.fix_base = fix_base
    return cfg


__all__ = ["make_alex_v2_cfg"]
