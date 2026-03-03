from pathlib import Path
import cv2
import mujoco
import mujoco.viewer
import numpy as np

def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    xml_path = (
        repo_root / "scenes" / "alex-scenes" / "scene_alex_v1_full_body_mjx_room1.xml"
    ).as_posix()
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # freejoint base pose: [x, y, z, qw, qx, qy, qz]
    data.qpos[:7] = [1.2, -0.8, 1.0, 1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)  # recompute kinematics after qpos change

    rgb_camera_name = "alex_head_rgb"
    depth_camera_name = "alex_head_depth"
    rgb_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, rgb_camera_name)
    depth_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, depth_camera_name)
    if rgb_camera_id < 0 or depth_camera_id < 0:
        raise RuntimeError(
            f'Missing camera(s). Expected "{rgb_camera_name}" and "{depth_camera_name}" in model.'
        )

    width = 1280
    height = 720
    num_steps = 1000
    out_dir = Path(__file__).resolve().parent
    rgb_out_path = out_dir / "alex_head_rgb_first_1000_steps_720p.mp4"
    depth_out_path = out_dir / "alex_head_depth_first_1000_steps_720p.mp4"
    fps = int(round(1.0 / model.opt.timestep))
    max_depth_m = 5.0

    try:
        renderer = mujoco.Renderer(model, width=width, height=height)
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize MuJoCo renderer. Run this in a desktop session "
            "with OpenGL support."
        ) from exc
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    rgb_writer = cv2.VideoWriter(rgb_out_path.as_posix(), fourcc, fps, (width, height))
    depth_writer = cv2.VideoWriter(depth_out_path.as_posix(), fourcc, fps, (width, height))
    if not rgb_writer.isOpened() or not depth_writer.isOpened():
        renderer.close()
        raise RuntimeError("Failed to initialize OpenCV video writers.")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            main_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "main")
            if main_cam_id >= 0:
                with viewer.lock():
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = main_cam_id

            for _ in range(num_steps):
                if not viewer.is_running():
                    break

                mujoco.mj_step(model, data)
                viewer.sync()

                renderer.disable_depth_rendering()
                renderer.update_scene(data, camera=rgb_camera_id)
                rgb_frame = renderer.render()
                rgb_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                rgb_writer.write(rgb_bgr)

                renderer.enable_depth_rendering()
                renderer.update_scene(data, camera=depth_camera_id)
                depth_frame_m = renderer.render()
                depth_frame_m = np.nan_to_num(depth_frame_m, nan=0.0, posinf=0.0, neginf=0.0)
                depth_norm = np.clip(depth_frame_m / max_depth_m, 0.0, 1.0)
                depth_u8 = ((1.0 - depth_norm) * 255.0).astype(np.uint8)
                depth_bgr = cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2BGR)
                depth_writer.write(depth_bgr)
    finally:
        rgb_writer.release()
        depth_writer.release()

    renderer.close()
    print(f"Saved {num_steps} RGB steps from '{rgb_camera_name}' to {rgb_out_path}")
    print(f"Saved {num_steps} depth steps from '{depth_camera_name}' to {depth_out_path}")

if __name__ == "__main__":
    main()
