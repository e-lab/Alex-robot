#!/usr/bin/env python3
"""Run Alex V1 from an already-running Isaac Sim app (Script Editor/streaming mode).

This script is intended for:
- Isaac Sim main window Script Editor
- `kit ... --exec ...` where Kit is already running

It does NOT create a new SimulationApp/AppLauncher.
"""

import argparse
import asyncio
import builtins
import importlib.util
from pathlib import Path

import omni.kit.app


def _find_repo_root():
    env_root = Path(__import__("os").environ.get("ALEX_REPO_ROOT", "")).expanduser()
    candidates = []
    if str(env_root):
        candidates.append(env_root)
    candidates.extend(
        [
            Path.cwd(),
            Path("/home/culurciello/Work/Alex-robot"),
            Path("/Users/euge/Code/github/Alex-robot"),
        ]
    )
    for base in candidates:
        if (base / "alex-models" / "alex_V1_isaacsim" / "alex.py").exists():
            return base
    return None


def parse_args():
    repo = _find_repo_root()
    default_alex_py = ""
    default_urdf = ""
    if repo is not None:
        default_alex_py = str(repo / "alex-models" / "alex_V1_isaacsim" / "alex.py")
        default_urdf = str(
            repo
            / "alex-models"
            / "alex_V1_description"
            / "rl_urdf"
            / "alex_v1.rlModel_fullBody_robotAccurate_fullCollisions.urdf"
        )

    parser = argparse.ArgumentParser(description="Run Alex V1 in an existing Isaac Sim app.")
    parser.add_argument(
        "--alex-py",
        default=default_alex_py,
        help="Absolute path to alex.py config module.",
    )
    parser.add_argument(
        "--urdf",
        default=default_urdf,
        help="Absolute path to Alex URDF.",
    )
    parser.add_argument(
        "--prim-path",
        default="/World/Alex",
        help="Prim path to spawn the articulation.",
    )
    return parser.parse_known_args()[0]


def _import_alex_module(alex_py: Path):
    if not alex_py.exists():
        raise FileNotFoundError(f"alex.py not found: {alex_py}")
    spec = importlib.util.spec_from_file_location("alex_local_cfg", str(alex_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from: {alex_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def _run_loop(app, sim):
    while app.is_running():
        sim.step()
        await app.next_update_async()


def main() -> None:
    args = parse_args()
    app = omni.kit.app.get_app()
    if app is None:
        raise RuntimeError("No running Kit app found. Use this only inside Isaac Sim app/streaming mode.")

    alex_py = Path(args.alex_py).expanduser().resolve()
    urdf = Path(args.urdf).expanduser().resolve()
    if not urdf.exists():
        raise FileNotFoundError(f"URDF not found: {urdf}")

    alex = _import_alex_module(alex_py)

    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab.sim import SimulationContext

    # Cancel previous loop if script was run before in this app session.
    old_task = getattr(builtins, "_alex_in_app_task", None)
    if old_task is not None and not old_task.done():
        old_task.cancel()

    sim_cfg = sim_utils.SimulationCfg(dt=alex.SIM_DT)
    sim = SimulationContext(sim_cfg)

    robot_cfg = alex.ALEX_V1_FULLBODY_DEFAULT_CFG.replace(
        prim_path=args.prim_path,
        spawn=alex.ALEX_V1_FULLBODY_DEFAULT_CFG.spawn.replace(
            asset_path=str(urdf),
        )
    )
    robot = Articulation(robot_cfg)
    # Compatibility across Isaac Lab versions:
    # - some versions expect explicit spawn()
    # - others materialize from prim_path in cfg during initialization/workflow
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    prim_exists = stage is not None and stage.GetPrimAtPath(args.prim_path).IsValid()
    if hasattr(robot, "spawn") and not prim_exists:
        robot.spawn(args.prim_path)

    sim.reset()
    task = asyncio.ensure_future(_run_loop(app, sim))

    # Keep handles reachable across script reruns.
    builtins._alex_in_app_task = task
    builtins._alex_in_app_sim = sim
    builtins._alex_in_app_robot = robot

    print(f"[alex] Running in-app loop. prim_path={args.prim_path} urdf={urdf}")
    print("[alex] Re-run script to restart. Existing loop is cancelled automatically.")


if __name__ == "__main__":
    # Script Editor executes code in a runtime context where sys.argv may contain Kit args.
    # parse_known_args() above handles that safely.
    main()
