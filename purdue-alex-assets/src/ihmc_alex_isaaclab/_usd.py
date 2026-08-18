# Copyright (c) 2026, Patrizio Acquadro.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared USD pose and fixed-joint helpers."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any


def finite_vector(value: object, *, length: int, name: str) -> tuple[float, ...]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != length
        or any(isinstance(item, bool) or not isinstance(item, Real) for item in value)
    ):
        raise TypeError(f"{name} must be a {length}-element real vector")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def normalized_xyzw(value: object, *, name: str) -> tuple[float, ...]:
    quaternion = finite_vector(value, length=4, name=name)
    norm = math.sqrt(sum(component * component for component in quaternion))
    if abs(norm - 1.0) > 1.0e-6:
        raise ValueError(f"{name} must be a normalized XYZW quaternion")
    return quaternion


def matrix_pose(matrix: Any) -> tuple[tuple[float, ...], Any]:
    from pxr import Gf

    transform = Gf.Transform(matrix)
    quaternion = transform.GetRotation().GetQuat()
    return (
        tuple(float(value) for value in transform.GetTranslation()),
        Gf.Quatf(
            float(quaternion.GetReal()),
            *tuple(float(value) for value in quaternion.GetImaginary()),
        ),
    )


def quaternion_close(actual: Any, expected: Any, tolerance: float = 1.0e-6) -> bool:
    actual_xyzw = (
        *tuple(float(value) for value in actual.GetImaginary()),
        float(actual.GetReal()),
    )
    expected_xyzw = (
        *tuple(float(value) for value in expected.GetImaginary()),
        float(expected.GetReal()),
    )
    direct = math.sqrt(
        sum((left - right) ** 2 for left, right in zip(actual_xyzw, expected_xyzw))
    )
    negated = math.sqrt(
        sum((left + right) ** 2 for left, right in zip(actual_xyzw, expected_xyzw))
    )
    return min(direct, negated) <= tolerance


def joint_pose_matches(joint: Any, position: tuple[float, ...], rotation: Any) -> bool:
    actual_position = joint.GetLocalPos0Attr().Get()
    actual_rotation = joint.GetLocalRot0Attr().Get()
    return (
        actual_position is not None
        and actual_rotation is not None
        and all(
            abs(float(actual_position[index]) - position[index]) <= 1.0e-6
            for index in range(3)
        )
        and quaternion_close(actual_rotation, rotation)
    )


def existing_fixed_joint(
    stage: Any,
    joint_path: Any,
    body0: Any,
    body1: Any,
    position0: tuple[float, ...],
    rotation0: Any,
    position1: tuple[float, ...],
    rotation1: Any,
    *,
    label: str,
    exclude_from_articulation: bool | None = None,
) -> str | None:
    """Accept an identical existing fixed joint or return ``None``."""

    from pxr import UsdPhysics

    existing = stage.GetPrimAtPath(joint_path)
    if not existing or not existing.IsValid():
        return None
    if not existing.IsA(UsdPhysics.FixedJoint):
        raise ValueError(f"{label} path is not a fixed joint: {joint_path}")
    joint = UsdPhysics.FixedJoint(existing)
    actual_position1 = joint.GetLocalPos1Attr().Get()
    actual_rotation1 = joint.GetLocalRot1Attr().Get()
    compatible = (
        joint.GetBody0Rel().GetTargets() == [body0.GetPath()]
        and joint.GetBody1Rel().GetTargets() == [body1.GetPath()]
        and joint_pose_matches(joint, position0, rotation0)
        and actual_position1 is not None
        and all(
            abs(float(actual_position1[index]) - position1[index]) <= 1.0e-6
            for index in range(3)
        )
        and actual_rotation1 is not None
        and quaternion_close(actual_rotation1, rotation1)
    )
    if exclude_from_articulation is not None:
        compatible = compatible and (
            joint.GetExcludeFromArticulationAttr().Get() is exclude_from_articulation
        )
    if not compatible:
        raise ValueError(f"existing {label} is incompatible: {joint_path}")
    return joint_path.pathString


def define_fixed_joint(
    stage: Any,
    joint_path: Any,
    body0: Any,
    body1: Any,
    position0: tuple[float, ...],
    rotation0: Any,
    position1: tuple[float, ...],
    rotation1: Any,
    *,
    exclude_from_articulation: bool | None = None,
) -> str:
    """Define a fixed joint with explicit local poses."""

    from pxr import Gf, UsdPhysics

    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([body0.GetPath()])
    joint.CreateBody1Rel().SetTargets([body1.GetPath()])
    if exclude_from_articulation is not None:
        joint.CreateExcludeFromArticulationAttr().Set(exclude_from_articulation)
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*position0))
    joint.CreateLocalRot0Attr().Set(rotation0)
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*position1))
    joint.CreateLocalRot1Attr().Set(rotation1)
    return joint_path.pathString
