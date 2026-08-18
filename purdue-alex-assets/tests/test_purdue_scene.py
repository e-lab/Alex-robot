# Copyright (c) 2026, Patrizio Acquadro.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Purdue measurements, pedestal, and reference-scene contracts."""

from __future__ import annotations

import inspect
from copy import deepcopy
from pathlib import Path

import isaaclab.sim as sim_utils
import pytest
import yaml
from isaaclab.sim.schemas.schemas_cfg import RigidBodyBaseCfg

import ihmc_alex_isaaclab.platforms.purdue_alex003_pedestal as pedestal_module
from ihmc_alex_isaaclab.platforms.purdue_alex003_pedestal import (
    PURDUE_ALEX003_MEASUREMENTS_PATH,
    PURDUE_ALEX003_PEDESTAL_ASSET_ROOT,
    PURDUE_ALEX003_PEDESTAL_URDF_PATH,
    load_purdue_alex003_pedestal_spec,
    make_purdue_alex003_pedestal_cfg,
)
from ihmc_alex_isaaclab.scenes.purdue_alex003 import (
    make_purdue_alex003_reference_scene_cfg,
)


def _measurement_record() -> dict[str, object]:
    loaded = yaml.safe_load(
        PURDUE_ALEX003_MEASUREMENTS_PATH.read_text(encoding="utf-8")
    )
    assert isinstance(loaded, dict)
    return loaded


def _write_measurement(path: Path, record: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(record, sort_keys=False), encoding="utf-8")
    return path


def test_loader_derives_measurements_and_alignment_from_yaml() -> None:
    record = _measurement_record()
    spec = load_purdue_alex003_pedestal_spec()
    pedestal = record["pedestal"]
    robot = record["robot"]

    assert spec.source_path == PURDUE_ALEX003_MEASUREMENTS_PATH.resolve()
    assert spec.model_id == pedestal["model_id"]
    assert tuple(part.name for part in spec.parts) == (
        "lower_base",
        "column",
        "upper_mount",
    )
    minimum_z = 0.0
    for part in spec.parts:
        source = pedestal["parts"][part.name]["size_m"]
        assert part.size_xyz_m == pytest.approx((source["x"], source["y"], source["z"]))
        maximum_z = minimum_z + source["z"]
        assert part.center_xyz_m == pytest.approx(
            (0.0, 0.0, (minimum_z + maximum_z) / 2.0)
        )
        assert part.z_bounds_m == pytest.approx((minimum_z, maximum_z))
        minimum_z = maximum_z

    assert spec.mounting_plane_world_z_m == pytest.approx(
        pedestal["mounting_plane_world_z_m"]["value"]
    )
    assert spec.alex_root_world_z_m == pytest.approx(0.88385597)
    assert spec.right_shoulder_y_world_z_m == pytest.approx(
        robot["right_shoulder_y_world_z_m"]["value"]
    )


def test_loader_rejects_invalid_geometry_and_alignment(tmp_path: Path) -> None:
    record = _measurement_record()

    invalid_convention = deepcopy(record)
    invalid_convention["setup"]["coordinate_convention"]["y"] = "right"
    with pytest.raises(ValueError, match="coordinate convention"):
        load_purdue_alex003_pedestal_spec(
            _write_measurement(tmp_path / "invalid_convention.yaml", invalid_convention)
        )

    broken_stack = deepcopy(record)
    broken_stack["pedestal"]["parts"]["column"]["size_m"]["z"] += 0.01
    with pytest.raises(ValueError, match="does not terminate"):
        load_purdue_alex003_pedestal_spec(
            _write_measurement(tmp_path / "broken_stack.yaml", broken_stack)
        )

    broken_alignment = deepcopy(record)
    broken_alignment["robot"]["right_shoulder_y_world_z_m"]["value"] = "unknown"
    with pytest.raises(ValueError, match="finite number"):
        load_purdue_alex003_pedestal_spec(
            _write_measurement(tmp_path / "broken_alignment.yaml", broken_alignment)
        )


def test_pedestal_factory_returns_independent_fixed_base_urdf_configs(
    tmp_path: Path,
) -> None:
    first = make_purdue_alex003_pedestal_cfg()
    second = make_purdue_alex003_pedestal_cfg()
    custom_urdf = tmp_path / "custom_pedestal.urdf"
    custom_urdf.write_text(
        PURDUE_ALEX003_PEDESTAL_URDF_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    custom = make_purdue_alex003_pedestal_cfg(asset_path=custom_urdf)

    assert PURDUE_ALEX003_PEDESTAL_ASSET_ROOT.is_dir()
    assert first is not second and first.spawn is not second.spawn
    assert first.class_type is None
    assert isinstance(first.spawn, sim_utils.UrdfFileCfg)
    assert first.spawn.asset_path == PURDUE_ALEX003_PEDESTAL_URDF_PATH.as_posix()
    assert first.spawn.fix_base is True
    assert first.spawn.link_density == 1000.0
    assert first.spawn.joint_drive is None
    assert first.spawn.collision_from_visuals is False
    assert isinstance(first.spawn.rigid_props, RigidBodyBaseCfg)
    assert first.spawn.rigid_props.disable_gravity is False
    assert first.spawn.articulation_props is None
    assert first.spawn.run_asset_transformer is False
    assert first.spawn.run_multi_physics_conversion is False
    assert custom.spawn.asset_path == custom_urdf.resolve().as_posix()
    assert first.init_state.pos == (0.0, 0.0, 0.0)
    assert first.init_state.rot == (0.0, 0.0, 0.0, 1.0)

    first.init_state.pos = (1.0, 2.0, 3.0)
    assert second.init_state.pos == (0.0, 0.0, 0.0)

    assert not hasattr(pedestal_module, "PurdueAlex003PedestalSpawnerCfg")
    assert not hasattr(pedestal_module, "spawn_purdue_alex003_pedestal")
    assert (
        "measurement_path"
        not in inspect.signature(make_purdue_alex003_pedestal_cfg).parameters
    )


def test_pedestal_factory_rejects_missing_or_non_urdf_assets(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        make_purdue_alex003_pedestal_cfg(asset_path=tmp_path / "missing.urdf")

    wrong_type = tmp_path / "pedestal.usda"
    wrong_type.write_text("#usda 1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a URDF"):
        make_purdue_alex003_pedestal_cfg(asset_path=wrong_type)


@pytest.mark.parametrize(
    ("variant", "filename"),
    [
        ("source", "alex_purdue_full.urdf"),
        ("full_convex", "alex_purdue_full_convex.urdf"),
    ],
)
def test_reference_scene_keeps_robot_and_pedestal_separate(
    variant: str, filename: str
) -> None:
    spec = load_purdue_alex003_pedestal_spec()
    first = make_purdue_alex003_reference_scene_cfg(robot_variant=variant)
    second = make_purdue_alex003_reference_scene_cfg(robot_variant=variant)

    assert first is not second
    assert first.pedestal is not first.robot
    assert first.pedestal.prim_path == "{ENV_REGEX_NS}/PurduePedestal"
    assert isinstance(first.pedestal.spawn, sim_utils.UrdfFileCfg)
    assert first.pedestal.spawn.fix_base is True
    assert first.robot.prim_path == "{ENV_REGEX_NS}/Robot"
    assert first.robot.spawn.fix_base is True
    assert first.robot.spawn.asset_path.endswith(filename)
    assert first.robot.init_state.pos == pytest.approx(
        (0.0, 0.0, spec.alex_root_world_z_m)
    )
    assert first.robot.init_state.rot == (0.0, 0.0, 0.0, 1.0)
    assert first.ground.prim_path == "/World/defaultGroundPlane"

    first.robot.init_state.pos = (1.0, 2.0, 3.0)
    assert second.robot.init_state.pos == pytest.approx(
        (0.0, 0.0, spec.alex_root_world_z_m)
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_envs": 0}, "num_envs"),
        ({"num_envs": True}, "num_envs"),
        ({"env_spacing": 0.0}, "env_spacing"),
        ({"env_spacing": float("nan")}, "env_spacing"),
    ],
)
def test_reference_scene_rejects_invalid_layout_inputs(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        make_purdue_alex003_reference_scene_cfg(**kwargs)
