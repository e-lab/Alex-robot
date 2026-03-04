#!/usr/bin/env python3
"""Minimal Isaac Lab runner for Alex V1."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Alex V1 in IsaacSim/Isaac Lab.")
    parser.add_argument(
        "--urdf",
        default="../alex_V1_description/rl_urdf/alex_v1.rlModel_fullBody_robotAccurate_fullCollisions.urdf",
        help="Path to the Alex URDF.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab.sim import SimulationContext

    import alex

    sim_cfg = sim_utils.SimulationCfg(dt=alex.SIM_DT)
    sim = SimulationContext(sim_cfg)

    # Spawn from URDF through the config defined in alex.py.
    robot_cfg = alex.ALEX_V1_FULLBODY_DEFAULT_CFG.replace(
        spawn=alex.ALEX_V1_FULLBODY_DEFAULT_CFG.spawn.replace(
            asset_path=args.urdf,
        )
    )
    robot = Articulation(robot_cfg)
    robot.spawn("/World/Alex")

    sim.reset()
    while simulation_app.is_running():
        sim.step()

    simulation_app.close()


if __name__ == "__main__":
    main()
