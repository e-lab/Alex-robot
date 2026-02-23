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

# Reference pose (crouched for stability)
STAND_PREP_TARGET = {
    "left_hip_x": 0.1, "right_hip_x": -0.1,
    "left_hip_y": -0.45, "right_hip_y": -0.45,
    "left_knee": 0.7, "right_knee": 0.7,
    "left_ankle_y": -0.28, "right_ankle_y": -0.28,
}

def load_trained_stand_pose():
    """
    Attempts to load the trained stand pose from alex-stand RL models.
    Returns a dict of joint positions if successful, else None.
    """
    stand_model_path = REPO_ROOT / "training/alex-stand/rl_models/best/best_model.zip"
    stand_vec_norm_path = REPO_ROOT / "training/alex-stand/rl_models/vec_normalize_final.pkl"
    
    if not stand_model_path.exists():
        stand_model_path = REPO_ROOT / "training/alex-stand/rl_models/alex_stand_final.zip"
    
    if not stand_model_path.exists():
        print(f"No trained stand model found at {stand_model_path}")
        return None

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        import gymnasium as gym

        print(f"Loading trained stand pose from {stand_model_path}...")
        
        import sys
        stand_dir = REPO_ROOT / "training/alex-stand"
        if str(stand_dir) not in sys.path:
            sys.path.append(str(stand_dir))
        
        try:
            from alex_stand_ppo import AlexStandEnv, ACTION_SCALE as STAND_ACTION_SCALE, STAND_PREP_TARGET as STAND_REF
            
            # Create a temporary env
            env = AlexStandEnv(render_mode=None)
            vec_env = DummyVecEnv([lambda: env])
            if stand_vec_norm_path.exists():
                vec_env = VecNormalize.load(str(stand_vec_norm_path), vec_env)
                vec_env.training = False
                vec_env.norm_reward = False
            
            model_sb3 = PPO.load(str(stand_model_path), env=vec_env)
            obs = vec_env.reset()
            action, _ = model_sb3.predict(obs, deterministic=True)
            
            q_trained = {}
            for i, name in enumerate(env._act_names):
                val = STAND_REF.get(name, 0.0) + action[0][i] * STAND_ACTION_SCALE
                q_trained[name] = val
            
            print("Successfully loaded trained stand pose.")
            return q_trained
            
        except ImportError as e:
            print(f"Could not import AlexStandEnv: {e}")
            return None
        except Exception as e:
            print(f"Error loading stand model: {e}")
            return None
            
    except ImportError:
        print("stable-baselines3 not installed, skipping trained pose load.")
        return None

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

    # Try to load trained stand pose
    trained_pose = load_trained_stand_pose()
    if trained_pose:
        for k, v in trained_pose.items():
            STAND_PREP_TARGET[k] = v

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
