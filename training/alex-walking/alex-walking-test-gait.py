#!/usr/bin/env python3
"""
Visual test for Alex reference walking gait.
This script plays back the reference gait cycle without any RL control.
"""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# Same paths and constants as in alex-walking-ppo.py
REPO_ROOT  = Path(__file__).resolve().parents[2]
SCENE_XML  = REPO_ROOT / "scenes/alex-scenes/scene_alex_v1_train.xml"
GAIT_PERIOD = 0.8

# Reference pose
STAND_PREP_TARGET = {
    "left_hip_x": 0.1, "right_hip_x": -0.1,
    "left_hip_y": -0.45, "right_hip_y": -0.45,
    "left_knee": 0.7, "right_knee": 0.7,
    "left_ankle_y": -0.28, "right_ankle_y": -0.28,
}

def get_ref_gait_offsets(phase: float) -> dict[str, float]:
    l_phase = phase
    r_phase = (phase + 0.5) % 1.0
    
    def leg_gait(p):
        if p < 0.5:
            s = np.sin(2 * np.pi * p)
            return -0.4 * s, 0.6 * s, 0.2 * s
        else:
            s = np.sin(2 * np.pi * (p - 0.5))
            return 0.3 * s, -0.1 * s, -0.1 * s

    l_hip, l_knee, l_ankle = leg_gait(l_phase)
    r_hip, r_knee, r_ankle = leg_gait(r_phase)
    
    return {
        "left_hip_y": l_hip, "left_knee": l_knee, "left_ankle_y": l_ankle,
        "right_hip_y": r_hip, "right_knee": r_knee, "right_ankle_y": r_ankle,
    }

def run():
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data  = mujoco.MjData(model)

    # Map actuator names to IDs
    act_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    name2id = {n: i for i, n in enumerate(act_names)}
    qadr = np.array([model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)])

    # Reset with stand-prep pose
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.98 # Height from test-init-pose
    data.qpos[3] = 1.0
    for nm, q in STAND_PREP_TARGET.items():
        if nm in name2id: data.qpos[qadr[name2id[nm]]] = q
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        while viewer.is_running():
            elapsed = time.time() - start_time
            phase = (elapsed / GAIT_PERIOD) % 1.0
            
            offsets = get_ref_gait_offsets(phase)
            
            # Reset q_des to base pose
            q_des = np.zeros(model.nu)
            for nm, q in STAND_PREP_TARGET.items():
                if nm in name2id: q_des[name2id[nm]] = q
            
            # Add gait offsets
            for nm, val in offsets.items():
                if nm in name2id: q_des[name2id[nm]] += val
            
            data.ctrl[:] = q_des
            
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

if __name__ == "__main__":
    run()
