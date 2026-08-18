# Copyright (c) 2026, Patrizio Acquadro.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pinned external Stereolabs ZED Isaac Sim dependency resolution."""

from __future__ import annotations

import hashlib
import os
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .._paths import REPOSITORY_ROOT

ZED_X_MINI_MANIFEST_PATH = (
    REPOSITORY_ROOT / "assets" / "sensors" / "zed_x_mini" / "dependency.toml"
)
DEFAULT_ZED_ISAAC_SIM_ROOT = (
    REPOSITORY_ROOT / "build" / "dependencies" / "zed-isaac-sim"
)


@dataclass(frozen=True)
class ZedIsaacSimDependency:
    """Validated paths for the pinned external dependency."""

    root: Path
    extension_root: Path
    usd_path: Path


def load_zed_x_mini_manifest() -> dict[str, object]:
    """Load the tracked dependency and physical-model contract."""

    if not ZED_X_MINI_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"ZED X Mini dependency manifest does not exist: {ZED_X_MINI_MANIFEST_PATH}"
        )
    with ZED_X_MINI_MANIFEST_PATH.open("rb") as stream:
        return tomllib.load(stream)


def _manifest_contract() -> tuple[dict[str, object], dict[str, object]]:
    manifest = load_zed_x_mini_manifest()
    try:
        upstream = manifest["upstream"]
        asset = manifest["asset"]
    except KeyError as error:
        raise ValueError("ZED X Mini dependency manifest is incomplete") from error
    if not isinstance(upstream, dict) or not isinstance(asset, dict):
        raise ValueError("ZED X Mini dependency manifest sections must be tables")
    return upstream, asset


def _git_head(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", root.as_posix(), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise ValueError(
            f"ZED Isaac Sim root is not a readable Git checkout: {root}"
        ) from error
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_zed_isaac_sim_root(
    root: str | os.PathLike[str],
) -> ZedIsaacSimDependency:
    """Validate commit, extension compatibility, and official USD bytes."""

    resolved = Path(root).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"ZED Isaac Sim root does not exist: {resolved}")

    upstream, asset = _manifest_contract()
    expected_commit = str(upstream.get("commit", ""))
    expected_extension_version = str(upstream.get("extension_version", ""))
    expected_kit_version = str(upstream.get("kit_version", ""))
    expected_hash = str(asset.get("sha256", ""))
    extension_path_value = str(upstream.get("extension_path", ""))
    usd_path_value = str(asset.get("path", ""))
    if not all(
        (
            expected_commit,
            expected_extension_version,
            expected_kit_version,
            expected_hash,
            extension_path_value,
            usd_path_value,
        )
    ):
        raise ValueError("ZED X Mini dependency manifest has empty required values")
    extension_relative_path = Path(extension_path_value)
    usd_relative_path = Path(usd_path_value)

    actual_commit = _git_head(resolved)
    if actual_commit != expected_commit:
        raise ValueError(
            "incompatible ZED Isaac Sim commit: "
            f"expected {expected_commit}, got {actual_commit} at {resolved}"
        )

    extension_root = resolved / extension_relative_path
    config_path = extension_root / "config" / "extension.toml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"ZED Isaac Sim extension metadata does not exist: {config_path}"
        )
    with config_path.open("rb") as stream:
        extension = tomllib.load(stream)
    actual_version = str(extension.get("package", {}).get("version", ""))
    if actual_version != expected_extension_version:
        raise ValueError(
            "incompatible sl.sensor.camera version: "
            f"expected {expected_extension_version}, got {actual_version or '<missing>'}"
        )
    target_kit = extension.get("package", {}).get("target", {}).get("kit", [])
    if target_kit != [expected_kit_version]:
        raise ValueError(
            "incompatible sl.sensor.camera Kit target: "
            f"expected [{expected_kit_version!r}], got {target_kit!r}"
        )

    usd_path = resolved / usd_relative_path
    if not usd_path.is_file():
        raise FileNotFoundError(f"official ZED_XM USD does not exist: {usd_path}")
    actual_hash = _sha256(usd_path)
    if actual_hash != expected_hash:
        raise ValueError(
            "incompatible ZED_XM.usdc SHA-256: "
            f"expected {expected_hash}, got {actual_hash}"
        )
    if not (extension_root / "sl/sensor/camera/isaaclab_utils.py").is_file():
        raise FileNotFoundError(
            "the pinned sl.sensor.camera checkout is missing its Isaac Lab helpers"
        )
    return ZedIsaacSimDependency(resolved, extension_root, usd_path)


def resolve_zed_isaac_sim_root(
    root: str | os.PathLike[str] | None = None,
) -> ZedIsaacSimDependency:
    """Resolve explicit root, then ``ZED_ISAAC_SIM_ROOT``, then ignored build path."""

    if root is not None:
        candidate = root
    elif "ZED_ISAAC_SIM_ROOT" in os.environ:
        candidate = os.environ["ZED_ISAAC_SIM_ROOT"]
    else:
        candidate = DEFAULT_ZED_ISAAC_SIM_ROOT
    return validate_zed_isaac_sim_root(candidate)
