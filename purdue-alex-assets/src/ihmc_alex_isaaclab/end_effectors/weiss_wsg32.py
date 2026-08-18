# Copyright (c) 2026, Patrizio Acquadro.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command and physics constants for the qualified WSG32 + UMI v1 profile."""

from __future__ import annotations

import math
import os
from numbers import Real
from pathlib import Path
from typing import Literal

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyBaseCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import (
    RigidBodyMaterialBaseCfg,
)

from .._paths import REPOSITORY_ROOT

WSG32Side = Literal["left", "right"]
WSG32_JAW_STROKE_M = 0.034
WSG32_MAX_FINGER_SPEED_M_S = 0.4
WSG32_MAX_FORCE_PER_JAW_N = 25.0
WSG32_MODEL_REFERENCE_STATIC_FRICTION = 1.5
WSG32_MODEL_REFERENCE_DYNAMIC_FRICTION = 1.2
_DEFAULT_ASSET_ROOT = REPOSITORY_ROOT / "assets" / "end_effectors" / "weiss_wsg32"
WEISS_WSG32_ASSET_ROOT = (
    Path(os.environ.get("WEISS_WSG32_ASSET_ROOT", _DEFAULT_ASSET_ROOT))
    .expanduser()
    .resolve()
)


def alex_purdue_wsg32_targets(position: Real, side: WSG32Side) -> dict[str, float]:
    """Map ``0=closed`` and ``1=open`` to one WSG leader-joint target."""

    if isinstance(position, bool) or not isinstance(position, Real):
        raise TypeError("WSG32 position must be a real scalar")
    normalized = float(position)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError("WSG32 position must be finite and within [0, 1]")
    if side not in ("left", "right"):
        raise ValueError("WSG32 side must be 'left' or 'right'")
    return {f"{side}_WSG32_JAW_OPENING": WSG32_JAW_STROKE_M * normalized}


def _resolve_wsg32_urdf(
    asset_path: str | os.PathLike[str] | None,
) -> Path:
    path = (
        WEISS_WSG32_ASSET_ROOT / "urdf" / "wsg32_umi_v1.urdf"
        if asset_path is None
        else Path(asset_path).expanduser().resolve()
    )
    if not path.is_file():
        raise FileNotFoundError(f"qualified WSG32 URDF does not exist: {path}")
    return path


def author_wsg32_finger_physics_material(
    stage: object, robot_prim_path: str
) -> tuple[str, ...]:
    """Bind the simulation-ready model-reference material to UMI colliders."""

    from pxr import Usd, UsdPhysics  # type: ignore

    root = stage.GetPrimAtPath(robot_prim_path)
    if not root or not root.IsValid():
        raise ValueError(f"WSG32 root prim does not exist: {robot_prim_path}")
    instance_roots = {}
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if "UMI_V1_" not in str(prim.GetPath()) or not prim.HasAPI(
            UsdPhysics.CollisionAPI
        ):
            continue
        parent = prim
        while parent and parent.IsValid() and parent != root:
            if parent.IsInstance():
                instance_roots[str(parent.GetPath())] = parent
                break
            parent = parent.GetParent()
    for instance_root in instance_roots.values():
        instance_root.SetInstanceable(False)

    colliders = [
        prim
        for prim in Usd.PrimRange(root)
        if "UMI_V1_" in str(prim.GetPath()) and prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if not colliders:
        raise ValueError(f"WSG32 UMI colliders are missing below {robot_prim_path}")
    material_path = f"{robot_prim_path}/WSG32UMIV1ModelReferenceMaterial"
    material = RigidBodyMaterialBaseCfg(
        static_friction=WSG32_MODEL_REFERENCE_STATIC_FRICTION,
        dynamic_friction=WSG32_MODEL_REFERENCE_DYNAMIC_FRICTION,
        restitution=0.0,
    )
    material.func(material_path, material)
    for collider in colliders:
        sim_utils.bind_physics_material(collider.GetPath(), material_path, stage=stage)
    return tuple(str(collider.GetPath()) for collider in colliders)


def spawn_wsg32_umi_v1(
    prim_path: str,
    cfg: sim_utils.UrdfFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: object,
) -> object:
    """Spawn standalone WSG instances and author their UMI contact material."""

    prim = sim_utils.spawn_from_urdf(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    stage = sim_utils.get_current_stage()
    roots = sim_utils.find_matching_prims(prim_path, stage=stage)
    if not roots:
        raise RuntimeError(
            f"standalone WSG32 spawn produced no prims for {prim_path!r}"
        )
    for root in roots:
        author_wsg32_finger_physics_material(stage, str(root.GetPath()))
    return prim


def make_wsg32_umi_v1_cfg(
    asset_path: str | os.PathLike[str] | None = None,
    *,
    fix_base: bool = True,
) -> ArticulationCfg:
    """Return a standalone Isaac Lab configuration for WSG32 + UMI v1."""

    path = _resolve_wsg32_urdf(asset_path)
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/WSG32",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=path.as_posix(),
            fix_base=fix_base,
            merge_fixed_joints=True,
            replace_cylinders_with_capsules=False,
            collision_from_visuals=False,
            collision_type="Convex Hull",
            self_collision=False,
            make_instanceable=False,
            activate_contact_sensors=True,
            func=spawn_wsg32_umi_v1,
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
            "opening": IdealPDActuatorCfg(
                joint_names_expr=["WSG32_JAW_OPENING"],
                stiffness=2000.0,
                damping=60.0,
                velocity_limit=WSG32_MAX_FINGER_SPEED_M_S,
                effort_limit=WSG32_MAX_FORCE_PER_JAW_N,
                velocity_limit_sim=WSG32_MAX_FINGER_SPEED_M_S,
                effort_limit_sim=WSG32_MAX_FORCE_PER_JAW_N,
                armature=0.001,
            )
        },
    )


__all__ = [
    "WSG32Side",
    "alex_purdue_wsg32_targets",
    "make_wsg32_umi_v1_cfg",
]
