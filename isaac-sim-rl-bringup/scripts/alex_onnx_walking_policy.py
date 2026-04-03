"""
alex_onnx_walking_policy.py
----------------------------
Standalone Isaac Sim: run the ONNX walking policy in Python (no Java, no DDS).
Keyboard control lets you steer the robot in real time.

Run with:
    cd ~/pathtoFolder/IsaacLab
    ./isaaclab.sh -p ~/pathtoFolder/Alex-robot/isaac-sim-rl-bringup/scripts/alex_onnx_walking_policy.py

Keyboard controls (focus the Isaac Sim viewport first):
    Arrow Up    / Numpad 8  — walk forward
    Arrow Down  / Numpad 2  — walk backward
    Arrow Left  / Z         — turn left  (yaw +)
    Arrow Right / X         — turn right (yaw -)
    Q / Numpad 4            — strafe left
    E / Numpad 6            — strafe right
    L                       — stop / reset velocity to zero
    S                       — toggle standing mode (velocity = 0, standing_flag = 1)

CLI args (override defaults):
    --vx FLOAT      initial forward velocity  (default 0.3 m/s)
    --vy FLOAT      initial lateral velocity  (default 0.0 m/s)
    --yaw FLOAT     initial yaw rate          (default 0.0 rad/s)
    --standing      start in standing mode    (standing_flag = 1)

Close the viewer window to exit.
"""

# AppLauncher MUST come before any Isaac imports.
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Alex ONNX walking policy — keyboard control")
parser.add_argument("--vx",       type=float, default=0.3,  help="Initial forward velocity (m/s)")
parser.add_argument("--vy",       type=float, default=0.0,  help="Initial lateral velocity (m/s)")
parser.add_argument("--yaw",      type=float, default=0.0,  help="Initial yaw rate (rad/s)")
parser.add_argument("--standing", action="store_true",       help="Start in standing mode")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Isaac imports (after AppLauncher) ────────────────────────────────────────
import copy
import math
import pathlib
import time

import numpy as np
import torch
import onnxruntime as ort

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import quat_apply_inverse

from isaaclab_assets.ihmc.robots.alex import alex as alex_cfg

# ── Paths ─────────────────────────────────────────────────────────────────────
_BRINGUP_ROOT = pathlib.Path(__file__).resolve().parents[1]   # .../isaac-sim-rl-bringup
_ALEX_ROBOT   = pathlib.Path(__file__).resolve().parents[2]   # .../Alex-robot

# ONNX policy — models/ inside this bringup folder
ONNX_PATH = _BRINGUP_ROOT / "models" / "2026-03-17_23-20-27_flatfeet" / "policy.onnx"

# URDF — alex_models/ already in the Alex-robot repo
ISAACDATA = _ALEX_ROBOT / "alex_models" / "alex_V1_description"

# ── Policy config (from 2026-03-17_23-20-27_flatfeet/policy_cfg.yaml) ────────
# Joint order matches YAML jointParameters order = obs joint_pos_rel / last_action order.
YAML_JOINT_ORDER = [
    "LEFT_HIP_X",    "RIGHT_HIP_X",   "SPINE_Z",
    "LEFT_HIP_Z",    "RIGHT_HIP_Z",
    "LEFT_SHOULDER_Y", "NECK_Z",      "RIGHT_SHOULDER_Y",
    "LEFT_HIP_Y",    "RIGHT_HIP_Y",
    "LEFT_SHOULDER_X", "NECK_Y",      "RIGHT_SHOULDER_X",
    "LEFT_KNEE_Y",   "RIGHT_KNEE_Y",
    "LEFT_SHOULDER_Z", "RIGHT_SHOULDER_Z",
    "LEFT_ANKLE_Y",  "RIGHT_ANKLE_Y",
    "LEFT_ELBOW_Y",  "RIGHT_ELBOW_Y",
    "LEFT_ANKLE_X",  "RIGHT_ANKLE_X",
]

# homePositions from YAML
HOME_POS = {
    "LEFT_HIP_X":      0.0,   "RIGHT_HIP_X":     0.0,   "SPINE_Z":         0.0,
    "LEFT_HIP_Z":      0.0,   "RIGHT_HIP_Z":     0.0,
    "LEFT_SHOULDER_Y": 0.15,  "NECK_Z":          0.0,   "RIGHT_SHOULDER_Y":0.15,
    "LEFT_HIP_Y":     -0.35,  "RIGHT_HIP_Y":    -0.35,
    "LEFT_SHOULDER_X": 0.05,  "NECK_Y":          0.0,   "RIGHT_SHOULDER_X":-0.05,
    "LEFT_KNEE_Y":     0.7,   "RIGHT_KNEE_Y":    0.7,
    "LEFT_SHOULDER_Z": 0.05,  "RIGHT_SHOULDER_Z":-0.05,
    "LEFT_ANKLE_Y":   -0.35,  "RIGHT_ANKLE_Y":  -0.35,
    "LEFT_ELBOW_Y":   -0.5,   "RIGHT_ELBOW_Y":  -0.5,
    "LEFT_ANKLE_X":    0.0,   "RIGHT_ANKLE_X":   0.0,
}

# kp / kd from YAML
KP = {
    "LEFT_HIP_X":     80.35,  "RIGHT_HIP_X":    80.35,  "SPINE_Z":         80.35,
    "LEFT_HIP_Z":     70.5,   "RIGHT_HIP_Z":    70.5,
    "LEFT_SHOULDER_Y":26.783, "NECK_Z":          5.0,   "RIGHT_SHOULDER_Y":26.783,
    "LEFT_HIP_Y":    108.6,   "RIGHT_HIP_Y":   108.6,
    "LEFT_SHOULDER_X":26.783, "NECK_Y":          5.0,   "RIGHT_SHOULDER_X":26.783,
    "LEFT_KNEE_Y":   108.6,   "RIGHT_KNEE_Y":  108.6,
    "LEFT_SHOULDER_Z":23.5,   "RIGHT_SHOULDER_Z":23.5,
    "LEFT_ANKLE_Y":   96.8,   "RIGHT_ANKLE_Y":  96.8,
    "LEFT_ELBOW_Y":   23.5,   "RIGHT_ELBOW_Y":  23.5,
    "LEFT_ANKLE_X":   72.6,   "RIGHT_ANKLE_X":  72.6,
}
KD = {
    "LEFT_HIP_X":     8.035,  "RIGHT_HIP_X":   8.035,   "SPINE_Z":        8.035,
    "LEFT_HIP_Z":     7.05,   "RIGHT_HIP_Z":   7.05,
    "LEFT_SHOULDER_Y":8.0,    "NECK_Z":        1.0,      "RIGHT_SHOULDER_Y":8.0,
    "LEFT_HIP_Y":    10.86,   "RIGHT_HIP_Y":  10.86,
    "LEFT_SHOULDER_X":8.0,    "NECK_Y":        1.0,      "RIGHT_SHOULDER_X":8.0,
    "LEFT_KNEE_Y":   10.86,   "RIGHT_KNEE_Y": 10.86,
    "LEFT_SHOULDER_Z":4.0,    "RIGHT_SHOULDER_Z":4.0,
    "LEFT_ANKLE_Y":   9.68,   "RIGHT_ANKLE_Y": 9.68,
    "LEFT_ELBOW_Y":   4.0,    "RIGHT_ELBOW_Y": 4.0,
    "LEFT_ANKLE_X":   7.26,   "RIGHT_ANKLE_X": 7.26,
}

ACTION_SCALE = 0.3
OBS_SIZE     = 80

# Initial spawn height: bent-knee home pose (HIP_Y=-0.35, KNEE_Y=0.7, ANKLE_Y=-0.35)
SPAWN_HEIGHT = 0.93   # metres — adjust if robot spawns with feet above/below ground

SIM_DT     = 0.005
DECIMATION = 4         # 4 × 5 ms = 20 ms per policy tick = 50 Hz

# ── Mutable command state (updated by keyboard at runtime) ────────────────────
# [vx, vy, yaw_rate, standing_flag]  — populated in main() from CLI args
_cmd = np.zeros(4, dtype=np.float32)   # [vx, vy, yaw, standing]


# ── URDF resolution (same logic as alex_onnx_policy_test.py) ─────────────────
def resolve_urdf() -> str:
    pkg_prefix = "package://alex_V1_description/"
    abs_prefix = str(ISAACDATA) + "/"

    def _rewrite(src: pathlib.Path) -> pathlib.Path:
        dst = src.with_name(src.stem + "_abs_paths.urdf")
        if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
            text = src.read_text()
            if pkg_prefix in text:
                dst.write_text(text.replace(pkg_prefix, abs_prefix))
        return dst if dst.exists() else src

    candidate = ISAACDATA / "rl_urdf" / (
        "alex_v1.rlModel_nubForearms_robotAccurate_torsoFootCollisions.urdf"
    )
    if candidate.exists():
        result = _rewrite(candidate)
        print(f"[URDF] {result}")
        return str(result)

    isaaclab_root = pathlib.Path.home() / "alex" / "repository-group" / "IsaacLab"
    candidate2 = (isaaclab_root / alex_cfg.ALEX_V1_NUBFOREARMS_MINIMALCOLLISIONS_URDF).resolve()
    if candidate2.exists():
        result = _rewrite(candidate2)
        print(f"[URDF] {result}")
        return str(result)

    raise FileNotFoundError(f"Alex URDF not found. Tried:\n  {candidate}\n  {candidate2}")


# ── Scene setup ───────────────────────────────────────────────────────────────
def setup_scene():
    sim_cfg = sim_utils.SimulationCfg(dt=SIM_DT, device="cpu")
    sim = SimulationContext(sim_cfg)
    # sim.set_camera_view(eye=[-2.0, 8.0, 2.5], target=[4.0, 0.0, 0.5])
    sim.set_camera_view(eye=[10.0, 8.0, 2.5], target=[4.0, 0.0, 0.5])

    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=2000.0)
    )

    robot_cfg = copy.deepcopy(alex_cfg.ALEX_V1_NUBS_DEFAULT_CFG)
    robot_cfg.spawn.asset_path = resolve_urdf()
    robot_cfg.init_state.joint_pos = HOME_POS   # non-zero home pose (bent-knee standing)
    robot_cfg.init_state.pos = (0.0, 0.0, SPAWN_HEIGHT)
    robot = Articulation(robot_cfg.replace(prim_path="/World/Alex"))

    left_contact  = ContactSensor(ContactSensorCfg(prim_path="/World/Alex/LEFT_FOOT",  update_period=0.0, history_length=1))
    right_contact = ContactSensor(ContactSensorCfg(prim_path="/World/Alex/RIGHT_FOOT", update_period=0.0, history_length=1))

    return sim, robot, left_contact, right_contact


# ── Joint map: name → Isaac DOF index ────────────────────────────────────────
def build_joint_map(robot: Articulation) -> dict:
    jmap = {}
    for name in YAML_JOINT_ORDER:
        idx_list, _ = robot.find_joints(name)
        if idx_list:
            jmap[name] = idx_list[0]
        else:
            print(f"[WARNING] Joint not found in Isaac: {name}")
    return jmap


# ── Observation builder ───────────────────────────────────────────────────────
def build_obs(robot: Articulation, joint_map: dict, last_action: np.ndarray) -> np.ndarray:
    """
    Build the 80-dim observation vector for the 2026-03-17_23-20-27_flatfeet policy:
      [0:3]   base_ang_vel      — root angular velocity in body frame
      [3:6]   projected_gravity — gravity unit vector in body frame
      [6:9]   base_velocity     — commanded velocity [vx, vy, yaw_rate]
      [9]     standing_flag     — 0.0=walking, 1.0=stand still (CMD_STANDING_FLAG)
      [10]    base_height       — root z height above ground
      [11:34] joint_pos_rel     — q - homePos, YAML order
      [34:57] joint_vel_rel     — qd, YAML order
      [57:80] last_action       — previous raw policy output, YAML order
    """
    dev = robot.data.root_quat_w.device
    obs = np.zeros(OBS_SIZE, dtype=np.float32)

    # base_ang_vel [0:3]: root angular velocity in body frame
    obs[0:3] = robot.data.root_ang_vel_b[0].cpu().numpy()

    # projected_gravity [3:6]: gravity unit vector in body frame
    grav_world = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=dev)
    pg = quat_apply_inverse(robot.data.root_quat_w[0:1], grav_world)[0].cpu().numpy()
    obs[3:6] = pg

    # base_velocity_plus_standing [6:10]: [vx, vy, yaw_rate, standing_flag]
    obs[6]  = _cmd[0]   # vx
    obs[7]  = _cmd[1]   # vy
    obs[8]  = _cmd[2]   # yaw
    obs[9]  = _cmd[3]   # standing_flag

    # base_height [10]: commanded desired height (matches training — UniformHeightCommandCfg default)
    obs[10] = 0.93

    # joint_pos_rel [11:34] and joint_vel_rel [34:57]
    joint_pos = robot.data.joint_pos[0].cpu().numpy()
    joint_vel = robot.data.joint_vel[0].cpu().numpy()
    for i, name in enumerate(YAML_JOINT_ORDER):
        idx = joint_map.get(name)
        if idx is None:
            continue
        obs[11 + i] = joint_pos[idx] - HOME_POS[name]
        obs[34 + i] = joint_vel[idx]

    # last_action [57:80]
    obs[57:80] = last_action

    return obs


# ── Apply policy output to Isaac articulation ─────────────────────────────────
def apply_policy(robot: Articulation, joint_map: dict, action: np.ndarray) -> None:
    """
    action: 23-dim residual from policy (before actionScale).
    q_des = homePos + actionScale * action  (homePos=0.0 for all joints)
    Applies as PD position target with kp/kd from YAML.
    """
    num_dofs    = robot.num_joints
    pos_targets = torch.zeros(1, num_dofs)
    stiffness   = torch.zeros(1, num_dofs)
    damping     = torch.zeros(1, num_dofs)

    for i, name in enumerate(YAML_JOINT_ORDER):
        idx = joint_map.get(name)
        if idx is None:
            continue
        q_des = HOME_POS[name] + ACTION_SCALE * float(action[i])
        pos_targets[0, idx] = q_des
        stiffness[0, idx]   = KP[name]
        damping[0, idx]     = KD[name]

    robot.set_joint_position_target(pos_targets)
    robot.set_joint_effort_target(torch.zeros(1, num_dofs))
    robot.write_joint_stiffness_to_sim(stiffness)
    robot.write_joint_damping_to_sim(damping)
    robot.write_data_to_sim()


# ── Keyboard command update ───────────────────────────────────────────────────
def _make_keyboard(sim_device: str) -> Se2Keyboard:
    """Create and return an Se2Keyboard; also bind S to toggle standing mode."""
    kb = Se2Keyboard(Se2KeyboardCfg(
        v_x_sensitivity=0.6,
        v_y_sensitivity=0.6,
        omega_z_sensitivity=0.8,
        sim_device=sim_device,
    ))
    # Remap so Q/E handle strafe (LEFT arrow turns by default in Se2Keyboard)
    kb._INPUT_KEY_MAPPING.update({
        "Q": np.asarray([0.0,  1.0, 0.0]) * kb.v_y_sensitivity,
        "E": np.asarray([0.0, -1.0, 0.0]) * kb.v_y_sensitivity,
    })

    def _toggle_standing():
        _cmd[3] = 0.0 if _cmd[3] > 0.5 else 1.0
        mode = "STANDING" if _cmd[3] > 0.5 else "WALKING"
        print(f"[keyboard] mode → {mode}")

    kb.add_callback("S", _toggle_standing)
    return kb


def _update_cmd_from_keyboard(kb: Se2Keyboard) -> None:
    """Pull latest velocity from keyboard and write into _cmd."""
    vel = kb.advance().numpy()   # [vx, vy, yaw]
    _cmd[0] = float(vel[0])
    _cmd[1] = float(vel[1])
    _cmd[2] = float(vel[2])
    # _cmd[3] (standing_flag) is toggled by the S callback — leave it alone here


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global _cmd

    # Initialise command from CLI args
    _cmd[0] = args_cli.vx
    _cmd[1] = args_cli.vy
    _cmd[2] = args_cli.yaw
    _cmd[3] = 1.0 if args_cli.standing else 0.0

    print(f"Loading ONNX model: {ONNX_PATH}")
    assert ONNX_PATH.exists(), f"ONNX not found: {ONNX_PATH}"
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    print(f"  ONNX input:  {input_name}  shape: {sess.get_inputs()[0].shape}")
    print(f"  ONNX output: {sess.get_outputs()[0].name}  shape: {sess.get_outputs()[0].shape}")
    print(f"  obs_size={OBS_SIZE}  action_scale={ACTION_SCALE}")
    print(f"  initial cmd: vx={_cmd[0]:.2f} vy={_cmd[1]:.2f} yaw={_cmd[2]:.2f} standing={int(_cmd[3])}")

    sim, robot, left_contact, right_contact = setup_scene()
    sim.reset()

    # Cap render FPS to 60 at runtime — prevents "fast forward" visual effect.
    # Physics runs at fixed 50 Hz regardless; this only affects the display update rate.
    import carb
    carb.settings.get_settings().set("/app/runLoops/main/rateLimitFrequency", 60)
    carb.settings.get_settings().set("/app/runLoops/rendering/rateLimitFrequency", 60)
    carb.settings.get_settings().set("/app/runLoops/main/syncToPresent", False)

    left_contact.reset()
    right_contact.reset()

    joint_map = build_joint_map(robot)
    print(f"Joint map built: {len(joint_map)}/23 joints found")

    kb = _make_keyboard(sim_device="cpu")
    print(kb)   # prints key binding summary

    last_action = np.zeros(23, dtype=np.float32)  # zero-init, matches training episode start
    tick = 0
    last_print = time.time()

    print("\nStarting policy loop. Close the viewer window to exit.")

    while simulation_app.is_running():
        robot.update(SIM_DT)

        # Update velocity command from keyboard
        _update_cmd_from_keyboard(kb)

        # Build observation
        obs = build_obs(robot, joint_map, last_action)

        # Run ONNX policy
        raw_action = sess.run(None, {input_name: obs[np.newaxis, :]})[0][0]  # (23,)
        last_action = raw_action.copy()

        # Apply to Isaac
        apply_policy(robot, joint_map, raw_action)

        # Step physics (4 substeps); only render on the last substep to reduce
        # display compositor load on high-resolution monitors.
        for i in range(DECIMATION):
            sim.step(render=(i == DECIMATION - 1))
            robot.update(SIM_DT)
            left_contact.update(SIM_DT)
            right_contact.update(SIM_DT)

        # Per-second diagnostic print
        now = time.time()
        if now - last_print >= 1.0:
            root_x   = float(robot.data.root_pos_w[0, 0])
            root_y   = float(robot.data.root_pos_w[0, 1])
            root_z   = float(robot.data.root_pos_w[0, 2])
            l_hip_y  = float(robot.data.joint_pos[0, joint_map["LEFT_HIP_Y"]])
            r_hip_y  = float(robot.data.joint_pos[0, joint_map["RIGHT_HIP_Y"]])
            l_knee_y = float(robot.data.joint_pos[0, joint_map["LEFT_KNEE_Y"]])

            dev = robot.data.root_quat_w.device
            grav_world = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=dev)
            pg = quat_apply_inverse(robot.data.root_quat_w[0:1], grav_world)[0].cpu().numpy()

            l_fz = float(left_contact.data.net_forces_w[0, 0, 2]) if left_contact.data.net_forces_w is not None else 0.0
            r_fz = float(right_contact.data.net_forces_w[0, 0, 2]) if right_contact.data.net_forces_w is not None else 0.0

            standing_str = "STAND" if _cmd[3] > 0.5 else "WALK "
            print(
                f"[tick {tick:5d}] [{standing_str}] cmd=({_cmd[0]:+.2f},{_cmd[1]:+.2f},{_cmd[2]:+.2f})  "
                f"pos=({root_x:+.2f},{root_y:+.2f},{root_z:.3f})m  "
                f"hipY L{l_hip_y:+.3f}/R{r_hip_y:+.3f}  "
                f"kneeY L{l_knee_y:+.3f}  "
                f"projGrav=({pg[0]:+.3f},{pg[1]:+.3f},{pg[2]:+.3f})  "
                f"Fz L{l_fz:+.0f}/R{r_fz:+.0f}N  "
                f"action_max={np.abs(raw_action).max():.4f}"
            )

            if tick == 0:
                print(f"  obs[0:3]  base_ang_vel  = {obs[0:3]}")
                print(f"  obs[3:6]  proj_grav     = {obs[3:6]}")
                print(f"  obs[6:9]  base_velocity = {obs[6:9]}")
                print(f"  obs[9]    standing_flag = {obs[9]:.1f}")
                print(f"  obs[10]   base_height   = {obs[10]:.4f}")
                print(f"  obs[11:34] joint_pos_rel = {obs[11:34]}")
                print(f"  raw_action = {raw_action}")

            last_print = now

        tick += 1


if __name__ == "__main__":
    main()
    simulation_app.close()