#!/usr/bin/env python3
# Copyright (c) 2026, Patrizio Acquadro.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prepare the pinned external Stereolabs ZED X Mini USD dependency."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
sys.path.insert(0, SOURCE_ROOT.as_posix())

from ihmc_alex_isaaclab.sensors.zed_x_mini_dependency import (  # noqa: E402
    DEFAULT_ZED_ISAAC_SIM_ROOT,
    load_zed_x_mini_manifest,
    validate_zed_isaac_sim_root,
)

_UPSTREAM = load_zed_x_mini_manifest()["upstream"]
ZED_ISAAC_SIM_REPOSITORY = str(_UPSTREAM["repository"])
ZED_ISAAC_SIM_COMMIT = str(_UPSTREAM["commit"])


def _run(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def prepare(destination: Path) -> None:
    """Create or validate the exact sparse checkout without building the plugin."""

    destination = destination.expanduser().resolve()
    if destination.exists():
        dependency = validate_zed_isaac_sim_root(destination)
        print(
            f"PASS: pinned ZED Isaac Sim dependency already prepared at {dependency.root}"
        )
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".zed-isaac-sim-", dir=destination.parent))
    try:
        _run(["git", "init"], cwd=temporary)
        _run(
            ["git", "remote", "add", "origin", ZED_ISAAC_SIM_REPOSITORY],
            cwd=temporary,
        )
        _run(["git", "sparse-checkout", "init", "--no-cone"], cwd=temporary)
        _run(
            ["git", "sparse-checkout", "set", "--no-cone", "/exts/sl.sensor.camera/"],
            cwd=temporary,
        )
        _run(
            ["git", "fetch", "--depth", "1", "origin", ZED_ISAAC_SIM_COMMIT],
            cwd=temporary,
        )
        _run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=temporary)
        validate_zed_isaac_sim_root(temporary)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"PASS: prepared pinned ZED Isaac Sim dependency at {destination}")


def check(destination: Path) -> None:
    """Validate the prepared checkout without creating or changing it."""

    dependency = validate_zed_isaac_sim_root(destination.expanduser().resolve())
    print(f"PASS: validated pinned ZED Isaac Sim dependency at {dependency.root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_ZED_ISAAC_SIM_ROOT,
        help="checkout root (default: ignored build/dependencies/zed-isaac-sim)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the prepared dependency without downloading or changing it",
    )
    args = parser.parse_args()
    try:
        if args.check:
            check(args.destination)
        else:
            prepare(args.destination)
    except (
        FileNotFoundError,
        OSError,
        subprocess.CalledProcessError,
        ValueError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
