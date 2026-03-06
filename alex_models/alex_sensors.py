"""Reusable Alex MuJoCo helpers: cameras, joints, actuators, and sensors.

This module is file-based (not a Python package import path because directory
name includes a dash). Load it from other scripts with:

    import importlib.util
    import sys
    from pathlib import Path
    p = Path("alex-models/alex_sensors.py").resolve()
    spec = importlib.util.spec_from_file_location("alex_sensors", str(p))
    alex_sensors = importlib.util.module_from_spec(spec)
    sys.modules["alex_sensors"] = alex_sensors
    spec.loader.exec_module(alex_sensors)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import mujoco
import numpy as np


@dataclass(frozen=True)
class AlexCameraIds:
    rgb: int
    depth: int


@dataclass(frozen=True)
class AlexCameraNames:
    rgb: str = "alex_head_rgb"
    depth: str = "alex_head_depth"
    main: str = "main"


def resolve_alex_camera_ids(
    model: mujoco.MjModel,
    names: AlexCameraNames = AlexCameraNames(),
) -> AlexCameraIds:
    def _find_camera_id(name: str) -> int:
        # 1) Exact match.
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cid >= 0:
            return cid
        # 2) Prefixed-name fallback (e.g. "robot/alex_head_rgb").
        for i in range(model.ncam):
            cam_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if cam_name is None:
                continue
            if cam_name == name or cam_name.endswith("/" + name):
                return i
        return -1

    rgb_id = _find_camera_id(names.rgb)
    depth_id = _find_camera_id(names.depth)
    if rgb_id < 0 or depth_id < 0:
        available = []
        for i in range(model.ncam):
            cam_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if cam_name is not None:
                available.append(cam_name)
        raise RuntimeError(
            f'Missing Alex camera(s). Expected "{names.rgb}" and "{names.depth}". '
            f"Available cameras: {available}"
        )
    return AlexCameraIds(rgb=rgb_id, depth=depth_id)


def resolve_main_camera_id(
    model: mujoco.MjModel,
    names: AlexCameraNames = AlexCameraNames(),
) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, names.main)


def lock_view_to_main_camera(
    viewer: mujoco.viewer.Handle,
    model: mujoco.MjModel,
    names: AlexCameraNames = AlexCameraNames(),
) -> bool:
    main_id = resolve_main_camera_id(model, names)
    if main_id < 0:
        return False
    with viewer.lock():
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = main_id
    return True


def render_alex_rgb_depth(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    camera_ids: AlexCameraIds,
    max_depth_m: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera=camera_ids.rgb)
    rgb_frame = renderer.render()
    rgb_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera_ids.depth)
    depth_frame_m = renderer.render()
    depth_frame_m = np.nan_to_num(depth_frame_m, nan=0.0, posinf=0.0, neginf=0.0)
    depth_norm = np.clip(depth_frame_m / max_depth_m, 0.0, 1.0)
    depth_u8 = ((1.0 - depth_norm) * 255.0).astype(np.uint8)
    depth_bgr = cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2BGR)
    return rgb_bgr, depth_bgr


def create_mp4_writer(
    output_path: Path,
    fps: int,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path.as_posix(), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to initialize video writer: {output_path}")
    return writer


def joint_id(model: mujoco.MjModel, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise KeyError(f"Joint not found: {name}")
    return jid


def actuator_id(model: mujoco.MjModel, name: str) -> int:
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid < 0:
        raise KeyError(f"Actuator not found: {name}")
    return aid


def body_id(model: mujoco.MjModel, name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise KeyError(f"Body not found: {name}")
    return bid


def sensor_id(model: mujoco.MjModel, name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0:
        raise KeyError(f"Sensor not found: {name}")
    return sid


def get_joint_position(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> float:
    jid = joint_id(model, name)
    qpos_adr = model.jnt_qposadr[jid]
    return float(data.qpos[qpos_adr])


def get_joint_velocity(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> float:
    jid = joint_id(model, name)
    qvel_adr = model.jnt_dofadr[jid]
    return float(data.qvel[qvel_adr])


def get_joint_states(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_names: Iterable[str],
) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for name in joint_names:
        out[name] = (
            get_joint_position(model, data, name),
            get_joint_velocity(model, data, name),
        )
    return out


def set_joint_positions(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_targets: dict[str, float],
    forward: bool = True,
) -> None:
    for name, value in joint_targets.items():
        jid = joint_id(model, name)
        qpos_adr = model.jnt_qposadr[jid]
        data.qpos[qpos_adr] = float(value)
    if forward:
        mujoco.mj_forward(model, data)


def set_joint_velocities(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_targets: dict[str, float],
    forward: bool = True,
) -> None:
    for name, value in joint_targets.items():
        jid = joint_id(model, name)
        qvel_adr = model.jnt_dofadr[jid]
        data.qvel[qvel_adr] = float(value)
    if forward:
        mujoco.mj_forward(model, data)


def set_base_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pos_xyz: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float],
    freejoint_qpos_start: int = 0,
    forward: bool = True,
) -> None:
    i = freejoint_qpos_start
    data.qpos[i : i + 3] = np.asarray(pos_xyz, dtype=np.float64)
    data.qpos[i + 3 : i + 7] = np.asarray(quat_wxyz, dtype=np.float64)
    if forward:
        mujoco.mj_forward(model, data)


def get_base_pose(
    data: mujoco.MjData,
    freejoint_qpos_start: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    i = freejoint_qpos_start
    pos = np.array(data.qpos[i : i + 3], dtype=np.float64).copy()
    quat = np.array(data.qpos[i + 3 : i + 7], dtype=np.float64).copy()
    return pos, quat


def set_actuator_controls(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controls: dict[str, float],
    clip: bool = False,
) -> None:
    for name, value in controls.items():
        aid = actuator_id(model, name)
        v = float(value)
        if clip and model.actuator_ctrllimited[aid]:
            lo, hi = model.actuator_ctrlrange[aid]
            v = max(float(lo), min(float(hi), v))
        data.ctrl[aid] = v


def get_actuator_controls(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_names: Iterable[str],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in actuator_names:
        aid = actuator_id(model, name)
        out[name] = float(data.ctrl[aid])
    return out


def read_sensor(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    sid = sensor_id(model, name)
    adr = model.sensor_adr[sid]
    dim = model.sensor_dim[sid]
    return np.array(data.sensordata[adr : adr + dim], dtype=np.float64).copy()


def read_sensors(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    sensor_names: Iterable[str],
) -> dict[str, np.ndarray]:
    return {name: read_sensor(model, data, name) for name in sensor_names}


def actuated_joint_names(model: mujoco.MjModel) -> list[str]:
    """Return unique actuated joint names in actuator order."""
    names: list[str] = []
    seen: set[str] = set()
    for aid in range(model.nu):
        jid = int(model.actuator_trnid[aid, 0])
        if jid < 0:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name is None or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def extract_alex_observation_terms(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_names: Iterable[str] | None = None,
    include_actions: bool = True,
    include_base_imu: bool = True,
) -> dict[str, np.ndarray]:
    """Extract default Alex observation terms as numpy arrays.

    Terms are named to be compatible with common mjlab observation group usage:
    - ``joint_pos``
    - ``joint_vel``
    - ``base_lin_vel`` (from sensor ``imu_pelvis_linear_velocity`` if present)
    - ``base_ang_vel`` (from sensor ``imu_pelvis_gyro`` if present)
    - ``actions`` (from ``data.ctrl``)
    """
    names = list(joint_names) if joint_names is not None else actuated_joint_names(model)
    jpos = np.array([get_joint_position(model, data, n) for n in names], dtype=np.float32)
    jvel = np.array([get_joint_velocity(model, data, n) for n in names], dtype=np.float32)
    terms: dict[str, np.ndarray] = {
        "joint_pos": jpos,
        "joint_vel": jvel,
    }

    if include_base_imu:
        try:
            terms["base_lin_vel"] = read_sensor(model, data, "imu_pelvis_linear_velocity").astype(np.float32)
        except KeyError:
            pass
        try:
            terms["base_ang_vel"] = read_sensor(model, data, "imu_pelvis_gyro").astype(np.float32)
        except KeyError:
            pass

    if include_actions:
        terms["actions"] = np.array(data.ctrl, dtype=np.float32).copy()
    return terms


@dataclass
class ObservationGroupAdapter:
    """Lightweight observation-group adapter compatible with mjlab-style groups.

    Behavior mirrors key ``ObservationGroupCfg`` options:
    - ``concatenate_terms``
    - ``concatenate_dim`` (supports -1/0 for 1D arrays)
    - ``history_length`` + ``flatten_history_dim`` with term-major flattening
    """

    term_order: list[str]
    concatenate_terms: bool = True
    concatenate_dim: int = -1
    history_length: int = 0
    flatten_history_dim: bool = True

    def __post_init__(self) -> None:
        self._history: dict[str, deque[np.ndarray]] = {
            name: deque(maxlen=max(1, self.history_length)) for name in self.term_order
        }

    def _with_history(self, name: str, value: np.ndarray) -> np.ndarray:
        if self.history_length <= 0:
            return value.astype(np.float32, copy=False)
        hist = self._history[name]
        hist.append(value.astype(np.float32, copy=False))
        items = list(hist)
        if len(items) < self.history_length:
            pad = [items[0]] * (self.history_length - len(items))
            items = pad + items
        stacked = np.stack(items, axis=0)  # (H, D...)
        if self.flatten_history_dim:
            return stacked.reshape(-1).astype(np.float32, copy=False)
        return stacked.astype(np.float32, copy=False)

    def build(self, terms: dict[str, np.ndarray]) -> dict[str, np.ndarray] | np.ndarray:
        missing = [n for n in self.term_order if n not in terms]
        if missing:
            raise KeyError(f"Missing observation terms: {missing}")
        ordered = {name: self._with_history(name, np.asarray(terms[name])) for name in self.term_order}
        if not self.concatenate_terms:
            return ordered
        arrays = [ordered[name].reshape(-1) for name in self.term_order]
        axis = 0 if self.concatenate_dim in (-1, 0) else self.concatenate_dim
        return np.concatenate(arrays, axis=axis).astype(np.float32, copy=False)


def make_default_alex_observation_adapter(
    include_base_imu: bool = True,
    include_actions: bool = True,
    history_length: int = 0,
    flatten_history_dim: bool = True,
    concatenate_terms: bool = True,
) -> ObservationGroupAdapter:
    """Create a default adapter with common Alex term ordering."""
    order = ["joint_pos", "joint_vel"]
    if include_base_imu:
        order.extend(["base_lin_vel", "base_ang_vel"])
    if include_actions:
        order.append("actions")
    return ObservationGroupAdapter(
        term_order=order,
        concatenate_terms=concatenate_terms,
        concatenate_dim=-1,
        history_length=history_length,
        flatten_history_dim=flatten_history_dim,
    )
