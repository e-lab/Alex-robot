# Copyright (c) 2026, Patrizio Acquadro.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot configuration and frame contracts."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from isaaclab.sim.schemas.schemas_cfg import RigidBodyBaseCfg

from ihmc_alex_isaaclab.end_effectors.sake_ezgripper import (
    alex_purdue_ezgripper_targets,
)
from ihmc_alex_isaaclab.robots.alex_purdue import make_alex_purdue_cfg
from ihmc_alex_isaaclab.robots.alex_v2 import make_alex_v2_cfg
from ihmc_alex_isaaclab.robots.purdue_frames import (
    ALEX_PURDUE_ASSET_ROOT,
    alex_purdue_frame_specs,
)
from ihmc_alex_isaaclab.robots.purdue_physics import (
    ALEX_PURDUE_MIMIC_JOINT_PAIRS,
    ALEX_PURDUE_SELF_COLLISION_FILTER_PAIRS,
)
from ihmc_alex_isaaclab.robots.sensor_frames import (
    ALEX_V2_ASSET_ROOT,
    alex_v2_sensor_frame_specs,
)


def test_purdue_factory_returns_independent_fixed_base_configs() -> None:
    first = make_alex_purdue_cfg(fix_base=False)
    second = make_alex_purdue_cfg()
    assert first is not second and first.spawn is not second.spawn
    assert first.spawn.fix_base is False
    assert second.spawn.fix_base is True
    assert second.spawn.merge_fixed_joints is True
    assert second.spawn.collision_from_visuals is False
    assert second.spawn.collision_type == "Convex Hull"
    assert second.spawn.self_collision is True
    assert second.spawn.make_instanceable is False
    assert isinstance(second.spawn.rigid_props, RigidBodyBaseCfg)
    assert second.spawn.rigid_props.disable_gravity is False
    assert second.spawn.articulation_props is None
    assert second.spawn.joint_drive_props is None
    assert set(second.actuators) == {"neck", "arms", "ezgrippers"}
    assert second.actuators["neck"].stiffness == 5.0
    assert second.actuators["neck"].damping == 1.0
    assert second.actuators["arms"].stiffness[".*SHOULDER_Y"] == 26.78
    assert second.actuators["arms"].damping[".*SHOULDER_Y"] == 8.0
    assert second.actuators["arms"].stiffness[".*GRIPPER_Y"] == 2.0
    assert second.actuators["ezgrippers"].stiffness == 2.0
    assert second.actuators["ezgrippers"].damping == 0.1
    assert second.actuators["ezgrippers"].effort_limit == 1.0
    assert second.actuators["ezgrippers"].velocity_limit == 3.67
    first.init_state.pos = (1.0, 2.0, 3.0)
    assert second.init_state.pos == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("variant", "filename"),
    [
        ("source", "alex_purdue_full.urdf"),
        ("full_convex", "alex_purdue_full_convex.urdf"),
    ],
)
def test_purdue_factory_selects_variant(variant: str, filename: str) -> None:
    cfg = make_alex_purdue_cfg(variant=variant)
    assert (
        cfg.spawn.asset_path
        == (ALEX_PURDUE_ASSET_ROOT / "urdf" / "baseline" / filename).as_posix()
    )
    expected = variant == "full_convex"
    assert cfg.spawn.self_collision is expected
    assert cfg.spawn.make_instanceable is False


def test_purdue_full_convex_filters_only_three_false_positive_pairs() -> None:
    assert ALEX_PURDUE_SELF_COLLISION_FILTER_PAIRS == (
        ("HEAD_LINK", "TORSO_LINK"),
        ("LEFT_GRIPPER_Y_LINK", "LEFT_WRIST_Z_LINK"),
        ("RIGHT_GRIPPER_Y_LINK", "RIGHT_WRIST_Z_LINK"),
    )
    assert len(ALEX_PURDUE_MIMIC_JOINT_PAIRS) == 4


def test_purdue_factory_rejects_invalid_inputs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Alex Purdue URDF does not exist"):
        make_alex_purdue_cfg(tmp_path / "missing.urdf")
    with pytest.raises(ValueError, match="unknown Alex Purdue variant"):
        make_alex_purdue_cfg(variant="unknown")
    with pytest.raises(ValueError, match="mutually exclusive"):
        make_alex_purdue_cfg(
            ALEX_PURDUE_ASSET_ROOT / "urdf" / "baseline" / "alex_purdue_full.urdf",
            variant="source",
        )


def test_ezgripper_normalized_targets_and_failures() -> None:
    assert alex_purdue_ezgripper_targets(0.0, "left") == {
        "left_ezgripper_knuckle_palm_l1_1": 1.3,
        "left_ezgripper_knuckle_l1_l2_1": 1.3,
    }
    assert alex_purdue_ezgripper_targets(1.0, "right") == {
        "right_ezgripper_knuckle_palm_l1_1": 0.0,
        "right_ezgripper_knuckle_l1_l2_1": 0.0,
    }
    assert tuple(alex_purdue_ezgripper_targets(0.5, "left").values()) == pytest.approx(
        (0.65, 0.65)
    )
    for invalid in (-0.01, 1.01, math.nan, math.inf):
        with pytest.raises(ValueError, match=r"within \[0, 1\]"):
            alex_purdue_ezgripper_targets(invalid, "left")
    with pytest.raises(TypeError, match="real scalar"):
        alex_purdue_ezgripper_targets(True, "left")
    with pytest.raises(TypeError, match="real scalar"):
        alex_purdue_ezgripper_targets("0.5", "left")
    with pytest.raises(ValueError, match="side"):
        alex_purdue_ezgripper_targets(0.5, "LEFT")


def test_alex_v2_factory_returns_independent_configs() -> None:
    first = make_alex_v2_cfg(fix_base=True)
    second = make_alex_v2_cfg()
    assert first is not second and first.spawn is not second.spawn
    assert first.spawn.fix_base is True
    assert second.spawn.fix_base is False
    assert second.spawn.merge_fixed_joints is True
    assert second.spawn.collision_from_visuals is False
    assert second.spawn.self_collision is True
    assert isinstance(second.spawn.rigid_props, RigidBodyBaseCfg)
    assert second.spawn.articulation_props is None
    first.init_state.pos = (1.0, 2.0, 3.0)
    assert second.init_state.pos == (0.0, 0.0, 1.0)


@pytest.mark.parametrize(
    ("variant", "filename"),
    [
        ("standard", "alex_v2.urdf"),
        ("forearm_convex", "alex_v2_forearm_convex.urdf"),
        ("full_convex", "alex_v2_full_convex.urdf"),
    ],
)
def test_alex_v2_factory_selects_variant(variant: str, filename: str) -> None:
    cfg = make_alex_v2_cfg(variant=variant)
    assert cfg.spawn.asset_path == (ALEX_V2_ASSET_ROOT / "urdf" / filename).as_posix()


def test_alex_v2_factory_rejects_invalid_inputs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Alex V2 URDF does not exist"):
        make_alex_v2_cfg(tmp_path / "missing.urdf")
    with pytest.raises(ValueError, match="unknown Alex V2 variant"):
        make_alex_v2_cfg(variant="unknown")
    with pytest.raises(ValueError, match="mutually exclusive"):
        make_alex_v2_cfg(
            ALEX_V2_ASSET_ROOT / "urdf" / "alex_v2.urdf",
            variant="forearm_convex",
        )


def test_alex_v2_frame_contract_and_failures(tmp_path: Path) -> None:
    by_variant = {
        variant: alex_v2_sensor_frame_specs(variant=variant)
        for variant in ("standard", "forearm_convex", "full_convex")
    }
    assert by_variant["forearm_convex"] == by_variant["standard"]
    assert by_variant["full_convex"] == by_variant["standard"]

    specs = by_variant["standard"]
    assert len(specs) == 19
    assert sum(spec["kind"] == "imu" for spec in specs) == 17
    assert sum(spec["kind"] == "camera" for spec in specs) == 2
    assert len({spec["frame"] for spec in specs}) == len(specs)
    assert all(
        math.isfinite(value)
        for spec in specs
        for key in ("xyz_m", "rpy_rad")
        for value in spec[key]
    )

    head = {spec["frame"]: spec for spec in specs}["HEAD_ZED_X_MINI_FRAME"]
    assert head["parent_link"] == "HEAD_LINK"
    assert head["xyz_m"] == pytest.approx((0.11603, 0.009965, -0.02983))
    assert head["rpy_rad"] == pytest.approx((0.0, 0.3633, 0.0))

    with pytest.raises(FileNotFoundError, match="Alex V2 URDF does not exist"):
        alex_v2_sensor_frame_specs(tmp_path / "missing.urdf")
    malformed = tmp_path / "malformed_v2.urdf"
    malformed.write_text('<robot name="NotAlex"/>\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected Alex V2 URDF root"):
        alex_v2_sensor_frame_specs(malformed)
    with pytest.raises(ValueError, match="unknown Alex V2 variant"):
        alex_v2_sensor_frame_specs(variant="unknown")


def test_purdue_frame_contract_and_failures(tmp_path: Path) -> None:
    source = alex_purdue_frame_specs(variant="source")
    assert source == alex_purdue_frame_specs(variant="full_convex")
    assert len(source) == 15
    assert {
        kind: sum(spec["kind"] == kind for spec in source)
        for kind in ("imu", "camera", "palm", "tcp")
    } == {"imu": 10, "camera": 1, "palm": 2, "tcp": 2}
    assert all(
        math.isfinite(value)
        for spec in source
        for key in ("xyz_m", "rpy_rad")
        for value in spec[key]
    )

    head = {spec["frame"]: spec for spec in source}["HEAD_ZED_X_MINI_FRAME"]
    assert head["parent_link"] == "HEAD_LINK"
    assert head["xyz_m"] == pytest.approx((0.11603, 0.009965, -0.02983))
    assert head["rpy_rad"] == pytest.approx((0.0, 0.3633, 0.0))

    with pytest.raises(FileNotFoundError, match="Alex Purdue URDF does not exist"):
        alex_purdue_frame_specs(tmp_path / "missing.urdf")
    malformed = tmp_path / "malformed_purdue.urdf"
    malformed.write_text('<robot name="NotAlex"/>\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected Alex Purdue URDF root"):
        alex_purdue_frame_specs(malformed)
    with pytest.raises(ValueError, match="unknown Alex Purdue variant"):
        alex_purdue_frame_specs(variant="unknown")
