#!/usr/bin/env python3
# Copyright (c) 2026, Patrizio Acquadro.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run one component or the complete PhysX/TGS GPU validation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
COMPONENT_GATES = {
    "alex_v2": ("scripts/validate/check_alex_v2.py",),
    "alex_purdue": ("scripts/validate/check_alex_purdue.py",),
    "wsg32": (
        "scripts/validate/check_alex_purdue_wsg32.py",
        "scripts/validate/check_wsg32_multi_env.py",
    ),
    "alex003": ("scripts/validate/check_purdue_alex003_scene.py",),
    "zed_x_mini": ("scripts/validate/check_zed_x_mini.py",),
    "realsense": ("scripts/validate/check_realsense.py",),
    "leap_hand": ("scripts/validate/check_leap_hand_v1.py",),
}
ALL_GATES = tuple(
    dict.fromkeys(gate for gates in COMPONENT_GATES.values() for gate in gates)
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=tuple(COMPONENT_GATES))
    return parser


def main() -> int:
    args = _parser().parse_args()
    gates = COMPONENT_GATES[args.component] if args.component else ALL_GATES
    for gate in gates:
        print(f"RUN: {gate}", flush=True)
        completed = subprocess.run(
            [sys.executable, gate], cwd=REPOSITORY_ROOT, check=False
        )
        if completed.returncode:
            print(f"NO-GO: {gate} failed with exit {completed.returncode}", flush=True)
            return completed.returncode
    scope = args.component or "release"
    print(f"GO: {scope} GPU validation passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
