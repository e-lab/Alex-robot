# Full basic room with Alex, using reusable alex-models/alex_sensors.py helpers.

import importlib.util
from pathlib import Path
import sys
import mujoco
import mujoco.viewer


alex_sensors_path = "../../alex-models/alex_sensors.py"
spec = importlib.util.spec_from_file_location("alex_sensors", str(alex_sensors_path))
if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to load module spec: {alex_sensors_path}")
alex_sensors = importlib.util.module_from_spec(spec)
sys.modules["alex_sensors"] = alex_sensors
spec.loader.exec_module(alex_sensors)

def main() -> None:
    xml_path = "../../scenes/alex-scenes/scene_alex_v1_full_body_mjx_room1.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # freejoint base pose: [x, y, z, qw, qx, qy, qz]
    alex_sensors.set_base_pose(
        model,
        data,
        pos_xyz=(1.2, -0.8, 1.0),
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        forward=True,
    )

    camera_ids = alex_sensors.resolve_alex_camera_ids(model)
    obs_adapter = alex_sensors.make_default_alex_observation_adapter(
        history_length=1, concatenate_terms=True
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
    rgb_writer = alex_sensors.create_mp4_writer(rgb_out_path, fps, width, height)
    depth_writer = alex_sensors.create_mp4_writer(depth_out_path, fps, width, height)

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            alex_sensors.lock_view_to_main_camera(viewer, model)

            for i in range(num_steps):
                if not viewer.is_running():
                    break

                mujoco.mj_step(model, data)
                viewer.sync()

                rgb_bgr, depth_bgr = alex_sensors.render_alex_rgb_depth(
                    renderer=renderer,
                    data=data,
                    camera_ids=camera_ids,
                    max_depth_m=max_depth_m,
                )
                rgb_writer.write(rgb_bgr)
                depth_writer.write(depth_bgr)

                # Example observation-group compatible vector from the shared module.
                if i == 0:
                    terms = alex_sensors.extract_alex_observation_terms(model, data)
                    obs = obs_adapter.build(terms)
                    print(f"Observation vector shape: {obs.shape}")
    finally:
        rgb_writer.release()
        depth_writer.release()

    renderer.close()
    print(f"Saved {num_steps} RGB steps to {rgb_out_path}")
    print(f"Saved {num_steps} depth steps to {depth_out_path}")

if __name__ == "__main__":
    main()
