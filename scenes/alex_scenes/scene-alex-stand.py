from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


BASE_QPOS = np.array([0.0, 0.0, 0.98, 1.0, 0.0, 0.0, 0.0], dtype=float)
HOLD_KP = 40.0
HOLD_KD = 4.0
PIN_BASE = False


def _set_stand_pose(data: mujoco.MjData) -> None:
    data.qpos[:7] = BASE_QPOS
    data.qpos[7:] = 0.0
    data.qvel[:] = 0.0


def main() -> None:
    xml_path = (Path(__file__).resolve().parent / "scene_alex_v1_full_body_mjx.xml").as_posix()
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    _set_stand_pose(data)
    mujoco.mj_forward(model, data)

    actuator_joint_ids = model.actuator_trnid[:, 0]
    actuator_gears = model.actuator_gear[:, 0]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            if PIN_BASE:
                data.qpos[:7] = BASE_QPOS
                data.qvel[:6] = 0.0

            # Keep actuated joints near the stand pose (all zeros from qpos[7:]).
            for a in range(model.nu):
                j = actuator_joint_ids[a]
                qadr = model.jnt_qposadr[j]
                vadr = model.jnt_dofadr[j]
                q = data.qpos[qadr]
                v = data.qvel[vadr]
                tau = -HOLD_KP * q - HOLD_KD * v
                data.ctrl[a] = np.clip(tau / max(abs(actuator_gears[a]), 1e-6), -1.0, 1.0)

            mujoco.mj_step(model, data)
            viewer.sync()
            remaining = model.opt.timestep - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
