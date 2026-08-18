#!/usr/bin/env python3
# Copyright (c) 2026, Patrizio Acquadro.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate the complete RealSense D405 and D435 URDFs from pinned Xacro."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import shutil
import subprocess
import sys
import tempfile
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = REPOSITORY_ROOT / "assets" / "sensors" / "realsense"
DESCRIPTION_ROOT = ASSET_ROOT / "source" / "realsense2_description"
DEFAULT_XACRO_ROOT = REPOSITORY_ROOT / "build" / "dependencies" / "xacro"

XACRO_REPOSITORY = "https://github.com/ros/xacro.git"
XACRO_TAG = "2.1.1"
XACRO_COMMIT = "390772abfe1e068f54aed674ce43873229a7db4e"

EXPECTED_LINKS = {
    "base_link",
    "camera_bottom_screw_frame",
    "camera_link",
    "camera_depth_frame",
    "camera_depth_optical_frame",
    "camera_infra1_frame",
    "camera_infra1_optical_frame",
    "camera_infra2_frame",
    "camera_infra2_optical_frame",
    "camera_color_frame",
    "camera_color_optical_frame",
}

MODELS = {
    "d405": {
        "source": DESCRIPTION_ROOT / "urdf" / "test_d405_camera.urdf.xacro",
        "destination": ASSET_ROOT / "d405" / "urdf" / "realsense_d405.urdf",
        "mappings": {"use_nominal_extrinsics": "true"},
        "mesh_source": "package://realsense2_description/meshes/d405.stl",
        "mesh_destination": "../meshes/d405.stl",
    },
    "d435": {
        "source": DESCRIPTION_ROOT / "urdf" / "test_d435_camera.urdf.xacro",
        "destination": ASSET_ROOT / "d435" / "urdf" / "realsense_d435.urdf",
        "mappings": {
            "use_nominal_extrinsics": "true",
            "add_plug": "false",
            "use_mesh": "true",
        },
        "mesh_source": "package://realsense2_description/meshes/d435.dae",
        "mesh_destination": "../meshes/d435_isaac.dae",
    },
}


def _run(command: list[str], *, cwd: Path) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_xacro_root(root: Path) -> Path:
    """Validate a source checkout of the generation-only Xacro dependency."""

    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"xacro source root does not exist: {root}; rerun with --prepare-xacro"
        )
    actual_commit = _run(["git", "rev-parse", "HEAD"], cwd=root)
    if actual_commit != XACRO_COMMIT:
        raise ValueError(
            f"xacro checkout must be {XACRO_TAG} ({XACRO_COMMIT}), got {actual_commit}"
        )
    package = root / "xacro" / "__init__.py"
    if not package.is_file():
        raise FileNotFoundError(f"xacro Python package is missing: {package}")
    return root


def prepare_xacro(root: Path) -> Path:
    """Create or validate the pinned generation-only Xacro checkout."""

    root = root.expanduser().resolve()
    if root.exists():
        return validate_xacro_root(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".xacro-", dir=root.parent))
    try:
        shutil.rmtree(temporary)
        subprocess.run(
            [
                "git",
                "clone",
                "--branch",
                XACRO_TAG,
                "--depth",
                "1",
                XACRO_REPOSITORY,
                temporary.as_posix(),
            ],
            cwd=root.parent,
            check=True,
        )
        validate_xacro_root(temporary)
        temporary.replace(root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return root


def _load_xacro(root: Path):
    root_text = root.as_posix()
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    try:
        xacro = importlib.import_module("xacro")
        substitution_args = importlib.import_module("xacro.substitution_args")
    except ModuleNotFoundError as error:
        raise RuntimeError(f"failed to import xacro from {root}") from error
    return xacro, substitution_args


def _urdf_bytes(model: str, xacro_root: Path) -> bytes:
    specification = MODELS[model]
    source = specification["source"]
    if not source.is_file():
        raise FileNotFoundError(f"official RealSense Xacro is missing: {source}")

    xacro, substitution_args = _load_xacro(xacro_root)
    original_find = substitution_args._eval_find

    def find_description(package: str) -> str:
        if package != "realsense2_description":
            raise ValueError(f"unexpected Xacro package lookup: {package}")
        return DESCRIPTION_ROOT.as_posix()

    substitution_args._eval_find = find_description
    try:
        document = xacro.process_file(
            source.as_posix(), mappings=specification["mappings"]
        )
    finally:
        substitution_args._eval_find = original_find

    content = document.toprettyxml(indent="  ")
    stable_source = (
        f"assets/sensors/realsense/source/realsense2_description/urdf/{source.name}"
    )
    content = content.replace(source.as_posix(), stable_source)
    content = content.replace(
        specification["mesh_source"], specification["mesh_destination"]
    )
    if "package://" in content:
        raise ValueError(f"generated {model} URDF contains a non-portable package URI")
    return content.encode("utf-8")


def _write_or_check(path: Path, content: bytes, *, check: bool) -> None:
    if check:
        if not path.is_file() or path.read_bytes() != content:
            raise ValueError(f"generated artifact is stale: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _joint_origin(robot: ET.Element, name: str) -> tuple[float, float, float]:
    joint = robot.find(f"joint[@name='{name}']")
    if joint is None or joint.get("type") != "fixed":
        raise ValueError(f"generated URDF is missing fixed joint {name!r}")
    origin = joint.find("origin")
    if origin is None or origin.get("xyz") is None:
        raise ValueError(f"generated URDF joint {name!r} has no origin")
    return tuple(float(value) for value in origin.get("xyz").split())


def _validate_frozen_contracts() -> None:
    manifest = tomllib.loads(
        (ASSET_ROOT / "dependency.toml").read_text(encoding="utf-8")
    )
    for relative_path, expected_hash in manifest["source_sha256"].items():
        path = ASSET_ROOT / relative_path
        if not path.is_file() or _sha256(path) != expected_hash:
            raise ValueError(f"official source hash mismatch: {path}")

    expected_origins = {
        "d405": {
            "camera_link_joint": (0.01085, 0.009, 0.021),
            "camera_color_joint": (0.0, 0.0, 0.0),
            "camera_infra2_joint": (0.0, -0.018, 0.0),
        },
        "d435": {
            "camera_link_joint": (0.0106, 0.0175, 0.0125),
            "camera_color_joint": (0.0, 0.015, 0.0),
            "camera_infra2_joint": (0.0, -0.050, 0.0),
        },
    }
    for model, specification in MODELS.items():
        product = specification["destination"]
        frozen = manifest["models"][model]
        if _sha256(product) != frozen["urdf_sha256"]:
            raise ValueError(f"generated URDF hash mismatch: {product}")
        mesh = ASSET_ROOT / frozen["mesh"]
        if _sha256(mesh) != frozen["stored_mesh_sha256"]:
            raise ValueError(f"stored mesh hash mismatch: {mesh}")

        robot = ET.parse(product).getroot()
        links = {link.get("name") for link in robot.findall("link")}
        if links != EXPECTED_LINKS:
            raise ValueError(f"generated {model} URDF link topology changed")
        joints = robot.findall("joint")
        if len(joints) != 10 or {joint.get("type") for joint in joints} != {"fixed"}:
            raise ValueError(f"generated {model} URDF fixed-joint topology changed")
        for joint_name, expected_origin in expected_origins[model].items():
            actual_origin = _joint_origin(robot, joint_name)
            if any(
                abs(actual - expected) > 1.0e-12
                for actual, expected in zip(actual_origin, expected_origin, strict=True)
            ):
                raise ValueError(
                    f"generated {model} nominal frame changed: {joint_name}"
                )
        mesh_element = robot.find("link[@name='camera_link']/visual/geometry/mesh")
        if (
            mesh_element is None
            or mesh_element.get("filename") != specification["mesh_destination"]
        ):
            raise ValueError(f"generated {model} portable mesh reference changed")


def generate(*, check: bool, xacro_root: Path) -> None:
    xacro_root = validate_xacro_root(xacro_root)
    for model, specification in MODELS.items():
        _write_or_check(
            specification["destination"],
            _urdf_bytes(model, xacro_root),
            check=check,
        )
    _validate_frozen_contracts()
    mode = "validated" if check else "generated"
    print(
        f"PASS: {mode} complete D405/D435 URDFs with xacro {XACRO_TAG} ({XACRO_COMMIT})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="fail if generated URDFs are stale"
    )
    parser.add_argument(
        "--prepare-xacro",
        action="store_true",
        help="prepare the pinned Xacro source checkout before generation",
    )
    parser.add_argument(
        "--xacro-root",
        type=Path,
        default=DEFAULT_XACRO_ROOT,
        help="pinned xacro source checkout (default: ignored build dependency)",
    )
    args = parser.parse_args()
    try:
        if args.prepare_xacro:
            prepare_xacro(args.xacro_root)
        generate(check=args.check, xacro_root=args.xacro_root)
    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        ValueError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
