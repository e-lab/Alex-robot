#!/usr/bin/env python3
"""IHMC-style Alex standing runtime for MuJoCo.

This script mirrors the Java-side approach:
- No online PPO training.
- Pretrained ONNX policy inference only.
- RL controller state flow with stand-prep and transition phases.
- Observation packing based on pelvis angular velocity, projected gravity,
  joint residuals, and joint velocities.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List

import mujoco
import mujoco.viewer
import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


# Mirrors AlexJointOrderHelpers.getNubBreadthFirstNames().
ORDERED_JOINT_NAMES: List[str] = [
    "LEFT_HIP_X",
    "RIGHT_HIP_X",
    "SPINE_Z",
    "LEFT_HIP_Z",
    "RIGHT_HIP_Z",
    "LEFT_SHOULDER_Y",
    "NECK_Z",
    "RIGHT_SHOULDER_Y",
    "LEFT_HIP_Y",
    "RIGHT_HIP_Y",
    "LEFT_SHOULDER_X",
    "NECK_Y",
    "RIGHT_SHOULDER_X",
    "LEFT_KNEE_Y",
    "RIGHT_KNEE_Y",
    "LEFT_SHOULDER_Z",
    "RIGHT_SHOULDER_Z",
    "LEFT_ANKLE_Y",
    "RIGHT_ANKLE_Y",
    "LEFT_ELBOW_Y",
    "RIGHT_ELBOW_Y",
    "LEFT_ANKLE_X",
    "RIGHT_ANKLE_X",
]

# Mirrors AlexStandPrepSetPoints values and side sign conventions.
STAND_PREP_SETPOINTS: Dict[str, float] = {
    "SPINE_Z": 0.0,
    "NECK_Y": 0.0,
    "NECK_Z": 0.0,
    "LEFT_HIP_Z": 0.0,
    "RIGHT_HIP_Z": 0.0,
    "LEFT_HIP_X": 0.1,
    "RIGHT_HIP_X": -0.1,
    "LEFT_HIP_Y": -0.45,
    "RIGHT_HIP_Y": -0.45,
    "LEFT_KNEE_Y": 0.7,
    "RIGHT_KNEE_Y": 0.7,
    "LEFT_ANKLE_Y": -0.28,
    "RIGHT_ANKLE_Y": -0.28,
    "LEFT_ANKLE_X": 0.0,
    "RIGHT_ANKLE_X": 0.0,
    "LEFT_SHOULDER_X": 0.4,
    "RIGHT_SHOULDER_X": -0.4,
    "LEFT_SHOULDER_Z": -0.4,
    "RIGHT_SHOULDER_Z": 0.4,
    "LEFT_SHOULDER_Y": 0.7,
    "RIGHT_SHOULDER_Y": 0.7,
    "LEFT_ELBOW_Y": -1.9,
    "RIGHT_ELBOW_Y": -1.9,
    "LEFT_WRIST_X": 0.0,
    "RIGHT_WRIST_X": 0.0,
    "LEFT_WRIST_Z": 0.0,
    "RIGHT_WRIST_Z": 0.0,
}

# Mirrors the explicit hand-balance model action scaling map.
DEFAULT_ACTION_SCALE = 0.5


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(repo_root(), path)


def java_to_mj_name(java_joint_name: str) -> str:
    return java_joint_name.lower()


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


@dataclass
class ControllerState:
    name: str = "STAND_PREP"
    state_time: float = 0.0


class ONNXPolicy:
    def __init__(self, model_path: str):
        if ort is None:
            raise RuntimeError(
                "onnxruntime is not installed. Install it with: pip install onnxruntime"
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def act(self, obs: np.ndarray) -> np.ndarray:
        if obs.ndim == 1:
            obs = obs[None, :]
        out = self.sess.run([self.output_name], {self.input_name: obs.astype(np.float32)})[0]
        out = np.asarray(out, dtype=np.float32)
        if out.ndim == 2:
            out = out[0]
        return out


@dataclass
class PolicyDefinition:
    input_size: int
    action_scale: float
    home_positions: np.ndarray
    observations: List[str]


def load_policy_definition(def_path: str) -> PolicyDefinition:
    if not os.path.exists(def_path):
        raise FileNotFoundError(def_path)
    if yaml is None:
        raise RuntimeError("PyYAML is required to load policy YAML. Install with: pip install pyyaml")
    with open(def_path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)

    input_size = int(obj["inputSize"])
    action_scale = float(obj.get("actionScale", DEFAULT_ACTION_SCALE))
    observations = list(obj.get("observations", []))
    by_name = {jp["name"]: jp for jp in obj.get("jointParameters", [])}
    home_positions = np.array(
        [float(by_name.get(name, {}).get("homePosition", 0.0)) for name in ORDERED_JOINT_NAMES],
        dtype=np.float64,
    )
    return PolicyDefinition(
        input_size=input_size,
        action_scale=action_scale,
        home_positions=home_positions,
        observations=observations,
    )


class AlexIHMCStandRunner:
    def __init__(
        self,
        scene_path: str,
        policy_path: str,
        frame_skip: int,
        action_scale: float | None,
        kp: float,
        kd: float,
        stand_prep_seconds: float,
        to_rl_transition_seconds: float,
        rl_seconds: float,
        exit_seconds: float,
        policy_definition_path: str,
    ):
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.policy = ONNXPolicy(policy_path)
        self.policy_definition = load_policy_definition(policy_definition_path)
        self.frame_skip = frame_skip
        self.dt = self.model.opt.timestep * frame_skip
        self.action_scale = self.policy_definition.action_scale if action_scale is None else action_scale
        self.kp = kp
        self.kd = kd
        self.stand_prep_seconds = stand_prep_seconds
        self.to_rl_transition_seconds = to_rl_transition_seconds
        self.rl_seconds = rl_seconds
        self.exit_seconds = exit_seconds

        self.controller_state = ControllerState()
        self.runtime = 0.0

        self.pelvis_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
        if self.pelvis_bid < 0:
            raise RuntimeError("Body 'pelvis_link' not found")

        self.joint_info = self._build_joint_info()
        self.actuator_info = self._build_actuator_info()

        self.base_qpos = np.array([0.0, 0.0, 0.98, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.freeze_targets = np.zeros(len(ORDERED_JOINT_NAMES), dtype=np.float64)
        self.last_action = np.zeros(len(ORDERED_JOINT_NAMES), dtype=np.float64)

        self._reset_to_stand_prep()

    def _build_joint_info(self):
        qadr = np.full(len(ORDERED_JOINT_NAMES), -1, dtype=np.int32)
        vadr = np.full(len(ORDERED_JOINT_NAMES), -1, dtype=np.int32)
        qlo = np.full(len(ORDERED_JOINT_NAMES), -np.inf, dtype=np.float64)
        qhi = np.full(len(ORDERED_JOINT_NAMES), np.inf, dtype=np.float64)

        for i, jname in enumerate(ORDERED_JOINT_NAMES):
            mj_name = java_to_mj_name(jname)
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, mj_name)
            if jid < 0:
                raise RuntimeError(f"Joint '{mj_name}' (from {jname}) not found in model")
            qadr[i] = self.model.jnt_qposadr[jid]
            vadr[i] = self.model.jnt_dofadr[jid]
            if self.model.jnt_limited[jid]:
                qlo[i] = self.model.jnt_range[jid, 0]
                qhi[i] = self.model.jnt_range[jid, 1]

        stand_prep = np.array([STAND_PREP_SETPOINTS.get(n, 0.0) for n in ORDERED_JOINT_NAMES], dtype=np.float64)
        return {
            "qadr": qadr,
            "vadr": vadr,
            "qlo": qlo,
            "qhi": qhi,
            "stand_prep": stand_prep,
        }

    def _build_actuator_info(self):
        a_for_joint = np.full(len(ORDERED_JOINT_NAMES), -1, dtype=np.int32)
        act_low = self.model.actuator_ctrlrange[:, 0].copy()
        act_high = self.model.actuator_ctrlrange[:, 1].copy()
        gear = self.model.actuator_gear[:, 0].copy()

        for a in range(self.model.nu):
            jid = self.model.actuator_trnid[a, 0]
            if jid < 0:
                continue
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if not joint_name:
                continue
            upper = joint_name.upper()
            if upper in ORDERED_JOINT_NAMES:
                idx = ORDERED_JOINT_NAMES.index(upper)
                a_for_joint[idx] = a

        missing = [ORDERED_JOINT_NAMES[i] for i in range(len(ORDERED_JOINT_NAMES)) if a_for_joint[i] < 0]
        if missing:
            raise RuntimeError(f"Missing actuators for joints: {missing}")

        return {
            "a_for_joint": a_for_joint,
            "low": act_low,
            "high": act_high,
            "gear": gear,
        }

    def _reset_to_stand_prep(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = self.base_qpos
        self.data.qvel[:] = 0.0
        for i, q_i in enumerate(self.joint_info["qadr"]):
            self.data.qpos[q_i] = self.joint_info["stand_prep"][i]
        mujoco.mj_forward(self.model, self.data)

        self.controller_state = ControllerState(name="STAND_PREP", state_time=0.0)
        self.runtime = 0.0
        self.last_action[:] = 0.0

    def _build_observation(self) -> np.ndarray:
        n = len(ORDERED_JOINT_NAMES)
        obs = np.zeros(7 + 3 * n, dtype=np.float32)

        # Pelvis angular velocity (approximation for pelvis Z-up angular velocity).
        w = self.data.cvel[self.pelvis_bid, :3].copy()

        # Projected gravity in pelvis frame.
        xmat = self.data.xmat[self.pelvis_bid].reshape(3, 3)
        g_world = self.model.opt.gravity.copy()
        g_norm = np.linalg.norm(g_world)
        if g_norm > 1e-8:
            g_world = g_world / g_norm
        proj_g = xmat.T @ g_world

        obs[0:3] = w.astype(np.float32)
        obs[3:6] = proj_g.astype(np.float32)
        obs[6] = float(self.data.xpos[self.pelvis_bid, 2])

        q = self.data.qpos[self.joint_info["qadr"]]
        qd = self.data.qvel[self.joint_info["vadr"]]
        q_res = q - self.policy_definition.home_positions

        obs[7 : 7 + n] = q_res.astype(np.float32)
        obs[7 + n : 7 + 2 * n] = qd.astype(np.float32)
        obs[7 + 2 * n : 7 + 3 * n] = self.last_action.astype(np.float32)
        if obs.shape[0] != self.policy_definition.input_size:
            raise RuntimeError(
                f"Observation size mismatch: built {obs.shape[0]} but definition expects {self.policy_definition.input_size}"
            )
        return obs

    def _policy_target_positions(self) -> tuple[np.ndarray, np.ndarray]:
        obs = self._build_observation()
        action = self.policy.act(obs)
        n = len(ORDERED_JOINT_NAMES)
        if action.shape[0] < n:
            raise RuntimeError(f"Policy output too small: got {action.shape[0]}, need at least {n}")
        action = np.clip(action[:n], -1.0, 1.0)
        q_target = self.joint_info["stand_prep"] + self.action_scale * action
        q_target = np.clip(q_target, self.joint_info["qlo"], self.joint_info["qhi"])
        return q_target, action

    def _compute_pd_ctrl(self, q_target: np.ndarray) -> np.ndarray:
        ctrl = np.zeros(self.model.nu, dtype=np.float64)
        q = self.data.qpos[self.joint_info["qadr"]]
        qd = self.data.qvel[self.joint_info["vadr"]]

        for i, a in enumerate(self.actuator_info["a_for_joint"]):
            tau = self.kp * (q_target[i] - q[i]) - self.kd * qd[i]
            u = tau / max(abs(self.actuator_info["gear"][a]), 1e-6)
            ctrl[a] = clamp(u, self.actuator_info["low"][a], self.actuator_info["high"][a])
        return ctrl

    def _update_state_machine(self):
        name = self.controller_state.name
        t = self.controller_state.state_time

        if name == "STAND_PREP" and t >= self.stand_prep_seconds:
            self.controller_state = ControllerState(name="RL_TRANSITION", state_time=0.0)
            return

        if name == "RL_TRANSITION" and t >= self.to_rl_transition_seconds:
            self.controller_state = ControllerState(name="RL_CONTROL", state_time=0.0)
            return

        if name == "RL_CONTROL" and t >= self.rl_seconds:
            self.freeze_targets = self.data.qpos[self.joint_info["qadr"]].copy()
            self.controller_state = ControllerState(name="EXIT_RL", state_time=0.0)
            return

        if name == "EXIT_RL" and t >= self.exit_seconds:
            self.controller_state = ControllerState(name="FREEZE", state_time=0.0)
            return

    def step(self):
        state = self.controller_state.name
        if state == "STAND_PREP":
            q_target = self.joint_info["stand_prep"]
            action = np.zeros_like(self.last_action)
        elif state == "RL_TRANSITION":
            alpha = self.controller_state.state_time / max(self.to_rl_transition_seconds, 1e-6)
            alpha = clamp(alpha, 0.0, 1.0)
            q_rl, action = self._policy_target_positions()
            q_target = (1.0 - alpha) * self.joint_info["stand_prep"] + alpha * q_rl
        elif state == "RL_CONTROL":
            q_target, action = self._policy_target_positions()
        elif state == "EXIT_RL":
            alpha = self.controller_state.state_time / max(self.exit_seconds, 1e-6)
            alpha = clamp(alpha, 0.0, 1.0)
            q_target = (1.0 - alpha) * self.freeze_targets + alpha * self.joint_info["stand_prep"]
            action = np.zeros_like(self.last_action)
        elif state == "FREEZE":
            q_target = self.freeze_targets
            action = np.zeros_like(self.last_action)
        else:
            raise RuntimeError(f"Unknown controller state: {state}")

        self.data.ctrl[:] = self._compute_pd_ctrl(q_target)

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.last_action[:] = action
        self.controller_state.state_time += self.dt
        self.runtime += self.dt
        self._update_state_machine()


def parse_args():
    p = argparse.ArgumentParser(description="Run IHMC-style pretrained standing policy in MuJoCo")
    p.add_argument(
        "--scene",
        default="scenes/alex-scenes/scene_alex_v1_full_body_mjx.xml",
        help="MuJoCo scene XML",
    )
    p.add_argument(
        "--policy",
        default="training/rl_models/20251219_standing18.onnx",
        help="Pretrained ONNX policy path",
    )
    p.add_argument(
        "--policy-def",
        default="training/rl_models/20251219_standing18_def.yaml",
        help="Policy YAML definition path",
    )
    p.add_argument("--frame-skip", type=int, default=1)
    p.add_argument("--action-scale", type=float, default=None)
    p.add_argument("--kp", type=float, default=80.0)
    p.add_argument("--kd", type=float, default=6.0)
    p.add_argument("--stand-prep-seconds", type=float, default=2.0)
    p.add_argument("--to-rl-transition-seconds", type=float, default=1.0)
    p.add_argument("--rl-seconds", type=float, default=20.0)
    p.add_argument("--exit-seconds", type=float, default=1.0)
    p.add_argument("--no-viewer", action="store_true")
    p.add_argument("--steps", type=int, default=20000)
    return p.parse_args()


def run(args):
    scene_path = resolve_path(args.scene)
    policy_path = resolve_path(args.policy)
    policy_def_path = resolve_path(args.policy_def)

    runner = AlexIHMCStandRunner(
        scene_path=scene_path,
        policy_path=policy_path,
        frame_skip=args.frame_skip,
        action_scale=args.action_scale,
        kp=args.kp,
        kd=args.kd,
        stand_prep_seconds=args.stand_prep_seconds,
        to_rl_transition_seconds=args.to_rl_transition_seconds,
        rl_seconds=args.rl_seconds,
        exit_seconds=args.exit_seconds,
        policy_definition_path=policy_def_path,
    )

    if args.no_viewer:
        for _ in range(args.steps):
            runner.step()
        print(f"finished no-viewer run, final state={runner.controller_state.name}")
        return

    try:
        with mujoco.viewer.launch_passive(runner.model, runner.data) as viewer:
            while viewer.is_running():
                t0 = time.time()
                runner.step()
                viewer.sync()
                sleep = runner.dt - (time.time() - t0)
                if sleep > 0:
                    time.sleep(sleep)
    except RuntimeError as exc:
        if sys.platform == "darwin" and "mjpython" in str(exc):
            raise RuntimeError(
                "Viewer on macOS requires mjpython. Run: "
                f"mjpython {os.path.abspath(__file__)} {' '.join(sys.argv[1:])}"
            ) from exc
        raise


if __name__ == "__main__":
    run(parse_args())
