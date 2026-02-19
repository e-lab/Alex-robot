#!/usr/bin/env python3
"""Visual and numeric check for Alex stand-prep mapping in MuJoCo.

Usage:
  python3 test1.py
  python3 test1.py --scene scenes/alex-scenes/scene_alex_v1_full_body_mjx.xml
  python3 test1.py --no-viewer
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


# Approximate AlexStandPrepSetPoints mapped to MuJoCo actuator names.
STAND_PREP_TARGET = {
    "spine_z": 0.0,
    "neck_y": 0.0,
    "neck_z": 0.0,
    "left_hip_z": 0.0,
    "right_hip_z": 0.0,
    "left_hip_x": 0.1,
    "right_hip_x": -0.1,
    "left_hip_y": -0.45,
    "right_hip_y": -0.45,
    "left_knee": 0.7,
    "right_knee": 0.7,
    "left_ankle_y": -0.28,
    "right_ankle_y": -0.28,
    "left_ankle_x": 0.0,
    "right_ankle_x": 0.0,
    "left_shoulder_x": 0.4,
    "right_shoulder_x": -0.4,
    "left_shoulder_z": -0.4,
    "right_shoulder_z": 0.4,
    "left_shoulder_y": 0.7,
    "right_shoulder_y": 0.7,
    "left_elbow": -1.9,
    "right_elbow": -1.9,
    "left_wrist_x": 0.0,
    "right_wrist_x": 0.0,
    "left_wrist_z": 0.0,
    "right_wrist_z": 0.0,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_scene_path(scene: str) -> str:
    p = Path(scene)
    if p.is_absolute():
        return p.as_posix()
    return (repo_root() / p).resolve().as_posix()


def build_actuator_maps(model: mujoco.MjModel):
    name_to_id = {}
    qadr = np.zeros(model.nu, dtype=np.int32)
    vadr = np.zeros(model.nu, dtype=np.int32)
    for a in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        if name is None:
            raise RuntimeError(f"Actuator {a} has no name.")
        name_to_id[name] = a
        joint_id = model.actuator_trnid[a, 0]
        qadr[a] = model.jnt_qposadr[joint_id]
        vadr[a] = model.jnt_dofadr[joint_id]
    return name_to_id, qadr, vadr


def print_mapping_report(model: mujoco.MjModel, name_to_id: dict[str, int], qadr: np.ndarray) -> list[str]:
    print("=== Mapping Report ===")
    print(f"model.nu={model.nu} model.nq={model.nq} model.nv={model.nv}")
    missing = []

    for act_name, q_target in STAND_PREP_TARGET.items():
        if act_name not in name_to_id:
            missing.append(act_name)
            print(f"[MISSING] actuator={act_name}")
            continue

        a = name_to_id[act_name]
        j = model.actuator_trnid[a, 0]
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint#{j}"
        q_i = int(qadr[a])

        q_lo, q_hi = -np.inf, np.inf
        if model.jnt_limited[j]:
            q_lo = float(model.jnt_range[j, 0])
            q_hi = float(model.jnt_range[j, 1])
        in_range = (q_target >= q_lo) and (q_target <= q_hi)
        tag = "OK" if in_range else "OUT_OF_RANGE"
        print(
            f"[{tag}] actuator={act_name:<16} joint={joint_name:<16} "
            f"q_target={q_target:+.3f} q_range=[{q_lo:+.3f}, {q_hi:+.3f}] qadr={q_i}"
        )
    return missing


def apply_pose(data: mujoco.MjData, name_to_id: dict[str, int], qadr: np.ndarray, base_qpos: np.ndarray) -> None:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[:7] = base_qpos
    for act_name, q_target in STAND_PREP_TARGET.items():
        if act_name not in name_to_id:
            continue
        a = name_to_id[act_name]
        data.qpos[qadr[a]] = q_target


def run(args: argparse.Namespace) -> None:
    scene_path = resolve_scene_path(args.scene)
    if not os.path.exists(scene_path):
        raise FileNotFoundError(scene_path)

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    name_to_id, qadr, vadr = build_actuator_maps(model)
    missing = print_mapping_report(model, name_to_id, qadr)
    if missing:
        print("\nMissing actuator names above mean mapping is incomplete for this model.")

    base_qpos = np.array([args.base_x, args.base_y, args.base_z, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    apply_pose(data, name_to_id, qadr, base_qpos)
    mujoco.mj_forward(model, data)

    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
    if pelvis >= 0:
        rot = data.xmat[pelvis].reshape(3, 3)
        print(
            f"\nInitial pelvis: z={data.xpos[pelvis,2]:.3f} "
            f"upright(z·Z)={rot[2,2]:.3f}"
        )

    if args.no_viewer:
        return

    print("\nViewer controls:")
    print("- Script starts in stand-prep pose.")
    print("- With --hold, PD will hold the pose for easier visual inspection.")
    print("- Close viewer window to exit.\n")

    act_low = model.actuator_ctrlrange[:, 0].copy()
    act_high = model.actuator_ctrlrange[:, 1].copy()
    act_mid = 0.5 * (act_low + act_high)
    gear = model.actuator_gear[:, 0].copy()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            if args.hold:
                ctrl = act_mid.copy()
                for act_name, q_target in STAND_PREP_TARGET.items():
                    if act_name not in name_to_id:
                        continue
                    a = name_to_id[act_name]
                    q = data.qpos[qadr[a]]
                    qd = data.qvel[vadr[a]]
                    tau = args.kp * (q_target - q) - args.kd * qd
                    ctrl[a] = tau / max(abs(gear[a]), 1e-6)
                data.ctrl[:] = np.clip(ctrl, act_low, act_high)
            else:
                data.ctrl[:] = 0.0

            mujoco.mj_step(model, data)
            viewer.sync()

            remaining = model.opt.timestep - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--scene",
        type=str,
        default="scenes/alex-scenes/scene_alex_v1_full_body_mjx.xml",
    )
    p.add_argument("--base-x", type=float, default=0.0)
    p.add_argument("--base-y", type=float, default=0.0)
    p.add_argument("--base-z", type=float, default=0.98)
    p.add_argument("--hold", action="store_true", help="Hold stand-prep pose with PD during viewer run.")
    p.add_argument("--kp", type=float, default=80.0)
    p.add_argument("--kd", type=float, default=6.0)
    p.add_argument("--no-viewer", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
