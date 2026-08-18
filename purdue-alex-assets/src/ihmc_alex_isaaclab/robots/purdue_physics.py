# Copyright (c) 2026, Patrizio Acquadro.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD physics authoring required by the Alex Purdue full-convex profile."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import isaaclab.sim as sim_utils

from ..end_effectors.weiss_wsg32 import author_wsg32_finger_physics_material

if TYPE_CHECKING:
    from isaaclab.sim.spawners.from_files import UrdfFileCfg


ALEX_PURDUE_SELF_COLLISION_FILTER_PAIRS = (
    ("HEAD_LINK", "TORSO_LINK"),
    ("LEFT_GRIPPER_Y_LINK", "LEFT_WRIST_Z_LINK"),
    ("RIGHT_GRIPPER_Y_LINK", "RIGHT_WRIST_Z_LINK"),
)
"""The three measured false-positive self-collision pairs in the convex profile."""

ALEX_PURDUE_WSG32_SELF_COLLISION_FILTER_PAIRS = (
    *ALEX_PURDUE_SELF_COLLISION_FILTER_PAIRS,
    ("left_WSG32_NEGATIVE_JAW_LINK", "left_WSG32_POSITIVE_JAW_LINK"),
    ("right_WSG32_NEGATIVE_JAW_LINK", "right_WSG32_POSITIVE_JAW_LINK"),
)
"""Alex filters plus the two adjacent WSG jaw pairs."""

ALEX_PURDUE_MIMIC_JOINT_PAIRS = (
    ("left_ezgripper_knuckle_palm_l1_2", "left_ezgripper_knuckle_palm_l1_1"),
    ("left_ezgripper_knuckle_l1_l2_2", "left_ezgripper_knuckle_l1_l2_1"),
    ("right_ezgripper_knuckle_palm_l1_2", "right_ezgripper_knuckle_palm_l1_1"),
    ("right_ezgripper_knuckle_l1_l2_2", "right_ezgripper_knuckle_l1_l2_1"),
)
"""Follower/leader pairs retained from the SAKE URDF mimic joints."""

ALEX_PURDUE_WSG32_MIMIC_JOINT_PAIRS = (
    ("left_WSG32_JAW_FOLLOWER", "left_WSG32_JAW_OPENING"),
    ("right_WSG32_JAW_FOLLOWER", "right_WSG32_JAW_OPENING"),
)
"""Follower/leader pairs for the two qualified WSG32 grippers."""


def author_alex_purdue_self_collision_filters(
    stage: Any,
    robot_prim_path: str,
    filter_pairs: tuple[tuple[str, str], ...] = ALEX_PURDUE_SELF_COLLISION_FILTER_PAIRS,
) -> tuple[tuple[str, str], ...]:
    """Author exactly the three full-convex collision-shape pair filters."""

    from pxr import Usd, UsdPhysics  # type: ignore

    robot = stage.GetPrimAtPath(robot_prim_path)
    if not robot or not robot.IsValid():
        raise ValueError(f"Alex Purdue robot prim does not exist: {robot_prim_path}")

    filtered_links = {link for pair in filter_pairs for link in pair}

    def owner_link_name(prim: Any) -> str | None:
        parent = prim.GetParent()
        while parent and parent.IsValid() and parent != robot:
            for link in filtered_links:
                if parent.GetName() == f"{link}_CONVEX":
                    return link
            if parent.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_name = parent.GetName()
                return rigid_name if rigid_name in filtered_links else None
            parent = parent.GetParent()
        return None

    # De-instance only filtered collision containers so unrelated meshes stay
    # shared while their shapes can receive local USD opinions.
    instance_roots: dict[str, Any] = {}
    for prim in Usd.PrimRange(robot, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if owner_link_name(prim) not in filtered_links:
            continue
        parent = prim
        while parent and parent.IsValid() and parent != robot:
            if parent.IsInstance():
                instance_roots[str(parent.GetPath())] = parent
                break
            parent = parent.GetParent()
    for prim in instance_roots.values():
        prim.SetInstanceable(False)

    colliders: dict[str, list[Any]] = {link: [] for link in filtered_links}
    existing_filters: list[str] = []
    for prim in Usd.PrimRange(robot):
        if prim.HasRelationship("physics:filteredPairs"):
            targets = prim.GetRelationship("physics:filteredPairs").GetTargets()
            if targets:
                existing_filters.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            link = owner_link_name(prim)
            if link in colliders:
                colliders[link].append(prim)

    if existing_filters:
        raise ValueError(
            "Alex Purdue source already contains collision pair filters: "
            + ", ".join(existing_filters)
        )
    missing = sorted(link for link, shapes in colliders.items() if not shapes)
    if missing:
        raise ValueError(
            f"Alex Purdue full-convex links have no colliders: {', '.join(missing)}"
        )

    authored: list[tuple[str, str]] = []
    for first_link, second_link in filter_pairs:
        for first in colliders[first_link]:
            api = UsdPhysics.FilteredPairsAPI.Apply(first)
            relationship = api.CreateFilteredPairsRel()
            for second in colliders[second_link]:
                relationship.AddTarget(second.GetPath())
                authored.append((str(first.GetPath()), str(second.GetPath())))
    return tuple(authored)


def spawn_alex_purdue_full_convex(
    prim_path: str,
    cfg: UrdfFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: object,
) -> Any:
    """Spawn all requested instances, then author instance-local pair filters."""

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
        raise RuntimeError(f"Alex Purdue spawn produced no prims for {prim_path!r}")
    for root in roots:
        author_alex_purdue_self_collision_filters(stage, str(root.GetPath()))
    return prim


def spawn_alex_purdue_wsg32_full_convex(
    prim_path: str,
    cfg: UrdfFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: object,
) -> Any:
    """Spawn the WSG full-convex profile and author common USD physics data."""

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
        raise RuntimeError(f"Alex Purdue spawn produced no prims for {prim_path!r}")
    for root in roots:
        robot_path = str(root.GetPath())
        author_alex_purdue_self_collision_filters(
            stage, robot_path, ALEX_PURDUE_WSG32_SELF_COLLISION_FILTER_PAIRS
        )
        author_wsg32_finger_physics_material(stage, robot_path)
    return prim


def spawn_alex_purdue_wsg32_source(
    prim_path: str,
    cfg: UrdfFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: object,
) -> Any:
    """Spawn the WSG source profile and author its UMI contact material."""

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
        raise RuntimeError(f"Alex Purdue spawn produced no prims for {prim_path!r}")
    for root in roots:
        robot_path = str(root.GetPath())
        author_wsg32_finger_physics_material(stage, robot_path)
    return prim
