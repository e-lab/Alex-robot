# Copyright (c) 2026, Patrizio Acquadro.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Official LEAP Hand V1 articulations, commands, and wrist mount contract."""

from __future__ import annotations

import math
import os
from collections.abc import Iterable
from numbers import Real
from pathlib import Path
from typing import Any, Literal

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.schemas.schemas_cfg import (
    ArticulationRootPropertiesCfg,
    CollisionPropertiesCfg,
    RigidBodyPropertiesCfg,
)
from isaaclab.sim.spawners.materials.physics_materials_cfg import (
    RigidBodyMaterialBaseCfg,
)

from .._usd import (
    define_fixed_joint,
    existing_fixed_joint,
    finite_vector,
    matrix_pose,
    normalized_xyzw,
)
from .._paths import REPOSITORY_ROOT

LeapHandV1Side = Literal["left", "right"]

LEAP_HAND_V1_ASSET_ROOT = (
    REPOSITORY_ROOT / "assets" / "end_effectors" / "leap_hand_v1"
).resolve()
LEAP_HAND_V1_DEFAULT_USD_ROOT = (
    REPOSITORY_ROOT / "build" / "isaac" / "leap_hand_v1"
).resolve()

LEAP_HAND_V1_MASS_KG = {"left": 0.744, "right": 0.746}
LEAP_HAND_V1_PALM_LINK = {"left": "palm_lower_left", "right": "palm_lower"}
LEAP_HAND_V1_JOINT_LIMITS = {
    "left": (
        (-1.047, 1.047),
        (-0.314, 2.23),
        (-0.506, 1.885),
        (-0.366, 2.042),
        (-1.047, 1.047),
        (-0.314, 2.23),
        (-0.506, 1.885),
        (-0.366, 2.042),
        (-1.047, 1.047),
        (-0.314, 2.23),
        (-0.506, 1.885),
        (-0.366, 2.042),
        (-2.094, 0.349),
        (-2.443, 0.47),
        (-1.20, 1.90),
        (-1.34, 1.88),
    ),
    "right": (
        (-1.047, 1.047),
        (-0.314, 2.23),
        (-0.506, 1.885),
        (-0.366, 2.042),
        (-1.047, 1.047),
        (-0.314, 2.23),
        (-0.506, 1.885),
        (-0.366, 2.042),
        (-1.047, 1.047),
        (-0.314, 2.23),
        (-0.506, 1.885),
        (-0.366, 2.042),
        (-0.349, 2.094),
        (-0.47, 2.443),
        (-1.20, 1.90),
        (-1.34, 1.88),
    ),
}

_DEFAULT_PRIM_PATH = {
    "left": "{ENV_REGEX_NS}/LeapHandV1Left",
    "right": "{ENV_REGEX_NS}/LeapHandV1Right",
}
_MOUNT_JOINT_NAME = "LeapHandV1WristMountJoint"


def _validated_side(side: str) -> LeapHandV1Side:
    if side not in ("left", "right"):
        raise ValueError("LEAP Hand V1 side must be 'left' or 'right'")
    return side


def leap_hand_v1_joint_targets(
    side: LeapHandV1Side, positions: Iterable[Real]
) -> dict[str, float]:
    """Map positions in official motor-ID order ``0..15`` to Isaac joint names."""

    validated_side = _validated_side(side)
    if isinstance(positions, (str, bytes, dict)):
        raise TypeError("positions must be an iterable of 16 real joint positions")
    try:
        values = tuple(positions)
    except TypeError as error:
        raise TypeError(
            "positions must be an iterable of 16 real joint positions"
        ) from error
    if len(values) != 16:
        raise ValueError("positions must contain exactly 16 values in motor-ID order")

    targets = {}
    for motor_id, (value, limits) in enumerate(
        zip(values, LEAP_HAND_V1_JOINT_LIMITS[validated_side], strict=True)
    ):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"motor {motor_id} position must be a real scalar")
        position = float(value)
        if not math.isfinite(position):
            raise ValueError(f"motor {motor_id} position must be finite")
        lower, upper = limits
        if not lower <= position <= upper:
            raise ValueError(
                f"motor {motor_id} position {position} is outside {validated_side} "
                f"limits [{lower}, {upper}] rad"
            )
        targets[f"a_{motor_id}"] = position
    return targets


def _normalize_generated_joint_names(usd_path: str | os.PathLike[str]) -> None:
    """Adapt Isaac 6's numeric-name sanitization in the ignored USD product."""

    from pxr import Sdf, Usd

    stage = Usd.Stage.Open(Path(usd_path).as_posix())
    if stage is None or not stage.GetDefaultPrim():
        raise RuntimeError(f"failed to open generated LEAP Hand V1 USD: {usd_path}")
    root = stage.GetDefaultPrim().GetPath()
    old_paths = [root.AppendPath(f"Physics/tn__{motor_id}_") for motor_id in range(16)]
    new_paths = [root.AppendPath(f"Physics/a_{motor_id}") for motor_id in range(16)]
    old_present = [stage.GetPrimAtPath(path).IsValid() for path in old_paths]
    new_present = [stage.GetPrimAtPath(path).IsValid() for path in new_paths]
    if all(new_present) and not any(old_present):
        return
    if not all(old_present) or any(new_present):
        raise RuntimeError(
            "generated LEAP Hand V1 USD has an incompatible joint-name layout"
        )
    edit = Sdf.BatchNamespaceEdit()
    for old_path, new_path in zip(old_paths, new_paths, strict=True):
        edit.Add(old_path, new_path)
    if not stage.GetRootLayer().Apply(edit):
        raise RuntimeError("failed to normalize generated LEAP Hand V1 joint names")
    stage.GetRootLayer().Save()


@sim_utils.clone
def _spawn_leap_hand_v1(
    prim_path: str,
    cfg: sim_utils.UrdfFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: object,
) -> object:
    """Generate the flat USD, normalize Isaac joint names, and spawn the hand."""

    from isaaclab.sim.converters import UrdfConverter
    from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file

    converter = UrdfConverter(cfg)
    _normalize_generated_joint_names(converter.usd_path)
    root = _spawn_from_usd_file(
        prim_path,
        converter.usd_path,
        cfg,
        translation,
        orientation,
        **kwargs,
    )
    if cfg.activate_contact_sensors:
        from pxr import Usd, UsdPhysics

        rigid_body_paths = [
            str(prim.GetPath())
            for prim in Usd.PrimRange(root)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        if len(rigid_body_paths) != 17:
            raise RuntimeError(
                "generated LEAP Hand V1 must contain exactly 17 rigid bodies; "
                f"found {len(rigid_body_paths)}"
            )
        for rigid_body_path in rigid_body_paths:
            sim_utils.activate_contact_sensors(rigid_body_path)
    return root


def make_leap_hand_v1_cfg(
    side: LeapHandV1Side,
    prim_path: str | None = None,
    fix_base: bool = True,
    usd_dir: str | os.PathLike[str] | None = None,
) -> ArticulationCfg:
    """Return one independent URDF-first LEAP V1 articulation configuration."""

    validated_side = _validated_side(side)
    selected_prim_path = (
        _DEFAULT_PRIM_PATH[validated_side] if prim_path is None else prim_path
    )
    if (
        not isinstance(selected_prim_path, str)
        or not selected_prim_path.strip()
        or selected_prim_path == "/"
    ):
        raise ValueError("prim_path must be a non-empty, non-root string")
    if not isinstance(fix_base, bool):
        raise TypeError("fix_base must be a bool")

    urdf_path = LEAP_HAND_V1_ASSET_ROOT / "urdf" / f"leap_hand_v1_{validated_side}.urdf"
    if not urdf_path.is_file():
        raise FileNotFoundError(f"LEAP Hand V1 URDF does not exist: {urdf_path}")
    output_root = (
        LEAP_HAND_V1_DEFAULT_USD_ROOT / validated_side
        if usd_dir is None
        else Path(usd_dir).expanduser().resolve()
    )

    return ArticulationCfg(
        prim_path=selected_prim_path.rstrip("/"),
        spawn=sim_utils.UrdfFileCfg(
            asset_path=urdf_path.as_posix(),
            usd_dir=output_root.as_posix(),
            usd_file_name=f"leap_hand_v1_{validated_side}.usd",
            fix_base=fix_base,
            merge_fixed_joints=False,
            replace_cylinders_with_capsules=False,
            collision_from_visuals=False,
            collision_type="Convex Decomposition",
            self_collision=True,
            make_instanceable=False,
            robot_type="End Effector",
            run_asset_transformer=False,
            run_multi_physics_conversion=False,
            activate_contact_sensors=True,
            func=_spawn_leap_hand_v1,
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=False,
                angular_damping=0.01,
            ),
            collision_props=CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            articulation_props=ArticulationRootPropertiesCfg(
                articulation_enabled=True,
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
                fix_root_link=fix_base,
            ),
            physics_material=RigidBodyMaterialBaseCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                target_type="position",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0,
                    damping=0.0,
                ),
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={"a_.*": 0.0},
            joint_vel={"a_.*": 0.0},
        ),
        soft_joint_pos_limit_factor=1.0,
        actuators={
            "fingers": IdealPDActuatorCfg(
                joint_names_expr=["a_.*"],
                stiffness=3.0,
                damping=0.1,
                effort_limit=0.5,
                effort_limit_sim=0.95,
                velocity_limit=8.48,
                velocity_limit_sim=8.48,
                armature=0.001,
                friction=0.01,
            )
        },
    )


def _find_palm_body(stage: Any, root: Any, side: LeapHandV1Side) -> Any:
    from pxr import Usd, UsdPhysics

    rigid_bodies = [
        prim for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    child_body_paths = {
        target
        for prim in Usd.PrimRange(root)
        if prim.IsA(UsdPhysics.RevoluteJoint)
        for target in UsdPhysics.RevoluteJoint(prim).GetBody1Rel().GetTargets()
    }
    candidates = [
        prim for prim in rigid_bodies if prim.GetPath() not in child_body_paths
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"LEAP Hand V1 must contain exactly one palm root body; found {len(candidates)}"
        )
    palm = candidates[0]
    expected_name = LEAP_HAND_V1_PALM_LINK[side]
    has_side_marker = any(
        prim.GetName() == expected_name for prim in Usd.PrimRange(palm)
    )
    if palm.GetName() != expected_name and not has_side_marker:
        raise ValueError(
            f"LEAP Hand V1 root is incompatible with side {side!r}: "
            f"missing {expected_name!r}"
        )
    return palm


def author_leap_hand_v1_mount(
    stage: Any,
    parent_body_prim_path: str,
    hand_prim_path: str,
    side: LeapHandV1Side,
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float],
) -> str:
    """Fix the official palm root to a parent-local metre/XYZW pose.

    The hand must have been imported with ``fix_base=False``. The operation is
    idempotent for an identical mount and fails closed for a fixed-base,
    differently mounted, wrong-handed, or otherwise overconstrained hand.
    """

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    if stage is None or not hasattr(stage, "GetPrimAtPath"):
        raise ValueError("USD stage is required to author a LEAP Hand V1 mount")
    validated_side = _validated_side(side)
    position = finite_vector(pos, length=3, name="pos")
    quaternion_xyzw = normalized_xyzw(rot, name="rot")
    parent = stage.GetPrimAtPath(parent_body_prim_path)
    if not parent or not parent.IsValid():
        raise ValueError(f"parent rigid body does not exist: {parent_body_prim_path}")
    if not parent.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError(f"parent prim is not a rigid body: {parent_body_prim_path}")

    root_path = hand_prim_path.rstrip("/")
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        raise ValueError(f"LEAP Hand V1 root prim does not exist: {hand_prim_path}")
    if parent.GetPath().HasPrefix(root.GetPath()):
        raise ValueError("parent rigid body cannot be inside the LEAP hand")
    palm = _find_palm_body(stage, root, validated_side)
    for prim, label in (
        (parent, "parent rigid body"),
        (root, "LEAP Hand V1 root"),
        (palm, "LEAP Hand V1 palm body"),
    ):
        if not UsdGeom.Xformable(prim):
            raise ValueError(f"{label} is not transformable: {prim.GetPath()}")

    joint_path = Sdf.Path(f"{root_path}/{_MOUNT_JOINT_NAME}")
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint) or prim.GetPath() == joint_path:
            continue
        joint = UsdPhysics.Joint(prim)
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()
        if palm.GetPath() not in body0 + body1:
            continue
        is_internal_finger_joint = (
            prim.IsA(UsdPhysics.RevoluteJoint)
            and prim.GetPath().HasPrefix(root.GetPath())
            and body0 == [palm.GetPath()]
            and len(body1) == 1
            and body1[0].HasPrefix(root.GetPath())
        )
        if not is_internal_finger_joint:
            raise ValueError(
                "LEAP Hand V1 palm is already constrained; import with "
                f"fix_base=False and remove incompatible mounts: {prim.GetPath()}"
            )

    parent_rotation = Gf.Quatf(
        quaternion_xyzw[3],
        quaternion_xyzw[0],
        quaternion_xyzw[1],
        quaternion_xyzw[2],
    )
    identity = Gf.Quatf(1.0, 0.0, 0.0, 0.0)
    existing_joint_path = existing_fixed_joint(
        stage,
        joint_path,
        parent,
        palm,
        position,
        parent_rotation,
        (0.0, 0.0, 0.0),
        identity,
        label="LEAP Hand V1 mount",
        exclude_from_articulation=True,
    )

    time = Usd.TimeCode.Default()
    parent_world = UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(time)
    root_world = UsdGeom.Xformable(root).ComputeLocalToWorldTransform(time)
    palm_world = UsdGeom.Xformable(palm).ComputeLocalToWorldTransform(time)
    palm_in_root = palm_world * root_world.GetInverse()
    mount = Gf.Transform()
    mount.SetTranslation(Gf.Vec3d(*position))
    mount.SetRotation(Gf.Rotation(Gf.Quatd(quaternion_xyzw[3], *quaternion_xyzw[:3])))
    desired_palm_world = mount.GetMatrix() * parent_world
    desired_root_world = palm_in_root.GetInverse() * desired_palm_world
    root_ancestor = root.GetParent()
    while (
        root_ancestor
        and root_ancestor.IsValid()
        and not UsdGeom.Xformable(root_ancestor)
    ):
        root_ancestor = root_ancestor.GetParent()
    root_parent_world = (
        UsdGeom.Xformable(root_ancestor).ComputeLocalToWorldTransform(time)
        if root_ancestor and root_ancestor.IsValid()
        else Gf.Matrix4d(1.0)
    )
    desired_root_local = desired_root_world * root_parent_world.GetInverse()
    root_position, root_rotation = matrix_pose(desired_root_local)
    if not sim_utils.standardize_xform_ops(
        root,
        translation=root_position,
        orientation=(
            *tuple(root_rotation.GetImaginary()),
            float(root_rotation.GetReal()),
        ),
    ):
        raise ValueError(f"LEAP Hand V1 root is not transformable: {hand_prim_path}")

    if existing_joint_path is not None:
        return existing_joint_path
    return define_fixed_joint(
        stage,
        joint_path,
        parent,
        palm,
        position,
        parent_rotation,
        (0.0, 0.0, 0.0),
        identity,
        exclude_from_articulation=True,
    )


__all__ = [
    "LeapHandV1Side",
    "author_leap_hand_v1_mount",
    "leap_hand_v1_joint_targets",
    "make_leap_hand_v1_cfg",
]
