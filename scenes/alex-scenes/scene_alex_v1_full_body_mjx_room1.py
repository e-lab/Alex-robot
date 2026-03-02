from pathlib import Path
import time
import mujoco
import mujoco.viewer

def main() -> None:
    xml_path = (Path(__file__).resolve().parent / "scene_alex_v1_full_body_mjx_room1.xml").as_posix()
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # freejoint base pose: [x, y, z, qw, qx, qy, qz]
    data.qpos[:7] = [1.2, -0.8, 1.0, 1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)  # recompute kinematics after qpos change

    with mujoco.viewer.launch_passive(model, data) as viewer:

        # use camera main view:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        with viewer.lock():
            viewer.cam.fixedcamid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_CAMERA, "main"
            )

        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            remaining = model.opt.timestep - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)

if __name__ == "__main__":
    main()
