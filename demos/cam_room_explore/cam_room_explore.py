#!/usr/bin/env python3
"""Explore the iTHOR room with a keyboard-driven or scripted camera robot."""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import time
from pathlib import Path
from threading import Lock

import cv2
import mujoco
import mujoco.viewer
import numpy as np
from mujoco.glfw import glfw


ROOM_ATTACH_Z_OFFSET_M = 0.1
DEFAULT_FLOORPLAN_XML = "scenes/ithor/FloorPlan1_physics_simple.xml"
CAMERA_ROBOT_RADIUS_M = 0.25
CAMERA_ROBOT_HEIGHT_M = 1.6
CAMERA_ROBOT_CAMERA_Z_M = 1.5
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 360
DEFAULT_DEPTH_MAX_M = 6.0
DASHBOARD_MARGIN_PX = 12
DASHBOARD_GAP_PX = 12
DASHBOARD_MAP_SIZE_PX = 360

ACTION_TO_MOTION = {
  "stop": (0.0, 0.0, 0.0),
  "forward": (1.0, 0.0, 0.0),
  "backward": (-1.0, 0.0, 0.0),
  "strafe_left": (0.0, -1.0, 0.0),
  "strafe_right": (0.0, 1.0, 0.0),
  "turn_left": (0.0, 0.0, 1.0),
  "turn_right": (0.0, 0.0, -1.0),
}


def build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Drive a first-person camera robot around the room with the keyboard."
  )
  parser.add_argument("--floorplan-xml", default=DEFAULT_FLOORPLAN_XML)
  parser.add_argument("--start-x", type=float, default=1.2)
  parser.add_argument("--start-y", type=float, default=-0.8)
  parser.add_argument("--start-z", type=float, default=ROOM_ATTACH_Z_OFFSET_M)
  parser.add_argument(
    "--start-yaw-deg",
    type=float,
    default=0.0,
    help="Initial camera_robot yaw in degrees.",
  )
  parser.add_argument("--move-speed", type=float, default=0.8, help="m/s")
  parser.add_argument("--turn-speed-deg", type=float, default=90.0, help="deg/s")
  parser.add_argument("--fovy", type=float, default=75.0)
  parser.add_argument("--camera-width", type=int, default=DEFAULT_CAMERA_WIDTH)
  parser.add_argument("--camera-height", type=int, default=DEFAULT_CAMERA_HEIGHT)
  parser.add_argument("--depth-max-m", type=float, default=DEFAULT_DEPTH_MAX_M)
  return parser


def parse_args() -> argparse.Namespace:
  return build_arg_parser().parse_args()


def _attach_explore_scene(scene_spec: mujoco.MjSpec, floorplan_xml: Path, fovy: float) -> None:
  floorplan_spec = mujoco.MjSpec.from_file(str(floorplan_xml))
  frame = scene_spec.worldbody.add_frame()
  frame.pos = (0.0, 0.0, ROOM_ATTACH_Z_OFFSET_M)
  scene_spec.attach(floorplan_spec, prefix="room/", frame=frame)
  scene_spec.worldbody.add_camera(
    name="main",
    pos=(1.6, -2.5, 1.8),
    xyaxes=(0.786, 0.618, -0.000, -0.236, 0.300, 0.924),
    fovy=75.0,
  )

  camera_robot = scene_spec.worldbody.add_body(name="camera_robot", mocap=True)
  camera_robot.add_geom(
    name="camera_robot_marker",
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    pos=(0.0, 0.0, CAMERA_ROBOT_HEIGHT_M * 0.5),
    size=(CAMERA_ROBOT_RADIUS_M, CAMERA_ROBOT_HEIGHT_M * 0.5),
    rgba=(0.10, 0.65, 0.95, 0.35),
  )
  camera_robot.add_camera(
    name="camera_robot_rgb",
    pos=(0.08, -0.03, CAMERA_ROBOT_CAMERA_Z_M),
    quat=(-0.5, -0.5, 0.5, 0.5),
    fovy=fovy,
  )
  camera_robot.add_camera(
    name="camera_robot_depth",
    pos=(0.08, 0.03, CAMERA_ROBOT_CAMERA_Z_M),
    quat=(-0.5, -0.5, 0.5, 0.5),
    fovy=fovy,
  )


def _yaw_to_quat_wxyz(yaw_rad: float) -> tuple[float, float, float, float]:
  half_yaw = 0.5 * yaw_rad
  return (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))


def draw_detections_bgr(
  image_bgr: np.ndarray,
  detections: list[dict],
  *,
  color: tuple[int, int, int],
) -> np.ndarray:
  output = image_bgr.copy()
  for detection in detections:
    x1, y1, x2, y2 = [int(round(v)) for v in detection["bbox_xyxy"]]
    x1 = max(0, min(output.shape[1] - 1, x1))
    x2 = max(0, min(output.shape[1] - 1, x2))
    y1 = max(0, min(output.shape[0] - 1, y1))
    y2 = max(0, min(output.shape[0] - 1, y2))
    cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
    label = f"{detection['label']} {detection['confidence']:.2f}"
    cv2.putText(
      output,
      label,
      (x1, max(18, y1 - 6)),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.55,
      color,
      2,
      cv2.LINE_AA,
    )
  return output


def depth_to_bgr(depth_image: np.ndarray, max_depth_m: float) -> np.ndarray:
  depth_norm = np.clip(depth_image / max(max_depth_m, 1e-6), 0.0, 1.0)
  depth_u8 = np.round(255.0 * (1.0 - depth_norm)).astype(np.uint8)
  return cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)


def _dashboard_window_worker(window_name: str, image_queue: mp.Queue) -> None:
  try:
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
  except cv2.error as exc:
    print(f"Dashboard disabled: OpenCV window creation failed: {exc}")
    return

  while True:
    image = image_queue.get()
    if image is None:
      break
    cv2.imshow(window_name, image)
    cv2.waitKey(1)

  try:
    cv2.destroyWindow(window_name)
  except cv2.error:
    pass


class DashboardWindow:
  def __init__(
    self,
    robot: "CameraRobotController",
    *,
    detector=None,
    update_hz: float = 8.0,
    window_name: str = "cam_room_explore_dashboard",
  ) -> None:
    self.robot = robot
    self.detector = detector
    self.window_name = window_name
    self.update_interval_s = 1.0 / max(update_hz, 1e-6)
    self._last_update_s = 0.0
    self._occupancy_cells: dict[tuple[int, int], int] = {}
    self._occupancy_resolution_m = 0.10
    self._object_index: dict[str, list[int]] = {}
    self._views: list[dict] = []
    self._map_owned_by_dashboard = True
    self._window_process: mp.Process | None = None
    self._image_queue: mp.Queue | None = None
    self._window_failed = False

  def set_map_state(
    self,
    *,
    occupancy_cells: dict[tuple[int, int], int],
    resolution_m: float,
    object_index: dict[str, list[int]],
    views: list[dict],
  ) -> None:
    self._occupancy_cells = dict(occupancy_cells)
    self._occupancy_resolution_m = resolution_m
    self._object_index = {label: list(ids) for label, ids in object_index.items()}
    self._views = [dict(view) for view in views]
    self._map_owned_by_dashboard = False

  def _cell(self, x_m: float, y_m: float) -> tuple[int, int]:
    return (
      int(math.floor(x_m / self._occupancy_resolution_m)),
      int(math.floor(y_m / self._occupancy_resolution_m)),
    )

  def _mark(self, cell: tuple[int, int], weight: int) -> None:
    self._occupancy_cells[cell] = self._occupancy_cells.get(cell, 0) + weight

  def _raytrace(self, start_xy: tuple[float, float], end_xy: tuple[float, float]) -> None:
    dx = end_xy[0] - start_xy[0]
    dy = end_xy[1] - start_xy[1]
    dist = math.hypot(dx, dy)
    if dist <= 1e-6:
      return
    steps = max(1, int(dist / self._occupancy_resolution_m))
    for idx in range(steps):
      alpha = idx / steps
      sample = (start_xy[0] + alpha * dx, start_xy[1] + alpha * dy)
      self._mark(self._cell(*sample), -1)
    self._mark(self._cell(*end_xy), 3)

  def _update_map_from_depth(self, pose: dict, depth_image: np.ndarray) -> None:
    height, width = depth_image.shape
    cx = (width - 1) * 0.5
    fy = 0.5 * height / math.tan(0.5 * self.robot.fovy_rad)
    fx = fy
    robot_xy = (float(pose["x"]), float(pose["y"]))
    yaw_rad = float(pose["yaw_rad"])
    for col in range(0, width, 6):
      band = depth_image[height // 3:(2 * height) // 3, col]
      valid = band[np.isfinite(band)]
      valid = valid[(valid > 0.15) & (valid < self.robot.depth_max_m)]
      if valid.size == 0:
        continue
      distance_m = float(np.median(valid))
      angle_offset = math.atan2(col - cx, fx)
      ray_yaw = yaw_rad + angle_offset
      end_xy = (
        robot_xy[0] + distance_m * math.cos(ray_yaw),
        robot_xy[1] + distance_m * math.sin(ray_yaw),
      )
      self._raytrace(robot_xy, end_xy)

  def _render_map_panel(self, pose: dict) -> np.ndarray:
    panel = np.full((DASHBOARD_MAP_SIZE_PX, DASHBOARD_MAP_SIZE_PX, 3), 24, dtype=np.uint8)
    cv2.putText(
      panel,
      "Occupancy Map",
      (10, 24),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.7,
      (255, 255, 255),
      2,
      cv2.LINE_AA,
    )

    if not self._occupancy_cells:
      cv2.putText(
        panel,
        "No map data yet",
        (90, DASHBOARD_MAP_SIZE_PX // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (180, 180, 180),
        2,
        cv2.LINE_AA,
      )
      return panel

    robot_cell = (
      int(math.floor(float(pose["x"]) / self._occupancy_resolution_m)),
      int(math.floor(float(pose["y"]) / self._occupancy_resolution_m)),
    )
    xs = [cell[0] for cell in self._occupancy_cells] + [robot_cell[0]]
    ys = [cell[1] for cell in self._occupancy_cells] + [robot_cell[1]]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    span_x = max(1, max_x - min_x + 1)
    span_y = max(1, max_y - min_y + 1)
    drawable = DASHBOARD_MAP_SIZE_PX - 2 * DASHBOARD_MARGIN_PX
    cell_px = max(1, min(drawable // span_x, drawable // span_y))
    map_width = span_x * cell_px
    map_height = span_y * cell_px
    origin_x = (DASHBOARD_MAP_SIZE_PX - map_width) // 2
    origin_y = (DASHBOARD_MAP_SIZE_PX - map_height) // 2

    for (cell_x, cell_y), score in self._occupancy_cells.items():
      x0 = origin_x + (cell_x - min_x) * cell_px
      y0 = origin_y + (max_y - cell_y) * cell_px
      color = (210, 210, 210) if score > 0 else (70, 70, 70)
      cv2.rectangle(panel, (x0, y0), (x0 + cell_px - 1, y0 + cell_px - 1), color, -1)

    for label, view_ids in self._object_index.items():
      if not view_ids:
        continue
      view_id = view_ids[-1]
      if view_id >= len(self._views):
        continue
      robot_xy = self._views[view_id]["robot_xy"]
      cell_x = int(math.floor(float(robot_xy[0]) / self._occupancy_resolution_m))
      cell_y = int(math.floor(float(robot_xy[1]) / self._occupancy_resolution_m))
      cx = origin_x + (cell_x - min_x) * cell_px + cell_px // 2
      cy = origin_y + (max_y - cell_y) * cell_px + cell_px // 2
      cv2.circle(panel, (cx, cy), max(2, cell_px // 2), (0, 220, 255), -1)
      cv2.putText(
        panel,
        label,
        (min(DASHBOARD_MAP_SIZE_PX - 80, cx + 5), max(18, cy - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 220, 255),
        1,
        cv2.LINE_AA,
      )

    cx = origin_x + (robot_cell[0] - min_x) * cell_px + cell_px // 2
    cy = origin_y + (max_y - robot_cell[1]) * cell_px + cell_px // 2
    cv2.circle(panel, (cx, cy), max(4, cell_px // 2), (0, 255, 0), -1)
    tip = (
      int(round(cx + math.cos(float(pose["yaw_rad"])) * max(8, cell_px * 1.5))),
      int(round(cy - math.sin(float(pose["yaw_rad"])) * max(8, cell_px * 1.5))),
    )
    cv2.arrowedLine(panel, (cx, cy), tip, (0, 255, 0), 2, tipLength=0.35)
    return panel

  def _compose_dashboard(
    self,
    rgb_image: np.ndarray,
    depth_image: np.ndarray,
    detections: list[dict],
    pose: dict,
  ) -> np.ndarray:
    rgb_panel = draw_detections_bgr(
      cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR),
      detections,
      color=(80, 255, 80),
    )
    depth_panel = depth_to_bgr(depth_image, self.robot.depth_max_m)
    map_panel = self._render_map_panel(pose)

    cv2.putText(rgb_panel, "RGB + YOLO", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 255, 80), 2, cv2.LINE_AA)
    cv2.putText(depth_panel, "Depth", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    top_height = max(rgb_panel.shape[0], depth_panel.shape[0])
    top_width = rgb_panel.shape[1] + depth_panel.shape[1] + DASHBOARD_GAP_PX
    canvas_width = max(top_width, map_panel.shape[1]) + 2 * DASHBOARD_MARGIN_PX
    canvas_height = top_height + map_panel.shape[0] + DASHBOARD_GAP_PX + 2 * DASHBOARD_MARGIN_PX
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    x_rgb = (canvas_width - top_width) // 2
    y_top = DASHBOARD_MARGIN_PX
    x_depth = x_rgb + rgb_panel.shape[1] + DASHBOARD_GAP_PX
    canvas[y_top:y_top + rgb_panel.shape[0], x_rgb:x_rgb + rgb_panel.shape[1]] = rgb_panel
    canvas[y_top:y_top + depth_panel.shape[0], x_depth:x_depth + depth_panel.shape[1]] = depth_panel

    y_map = y_top + top_height + DASHBOARD_GAP_PX
    x_map = (canvas_width - map_panel.shape[1]) // 2
    canvas[y_map:y_map + map_panel.shape[0], x_map:x_map + map_panel.shape[1]] = map_panel
    return canvas

  def update(self, viewer: mujoco.viewer.Handle) -> None:
    del viewer
    now_s = time.time()
    if self._window_failed or (now_s - self._last_update_s) < self.update_interval_s:
      return
    rgb_image, depth_image = self.robot.capture_rgb_depth()
    detections = self.detector.detect(rgb_image) if self.detector is not None else []
    pose = self.robot.get_pose()
    if self._map_owned_by_dashboard:
      self._update_map_from_depth(pose, depth_image)
    dashboard = self._compose_dashboard(rgb_image, depth_image, detections, pose)
    if self._window_process is None:
      try:
        ctx = mp.get_context("spawn")
        self._image_queue = ctx.Queue(maxsize=1)
        self._window_process = ctx.Process(
          target=_dashboard_window_worker,
          args=(self.window_name, self._image_queue),
          daemon=True,
        )
        self._window_process.start()
      except Exception as exc:
        print(f"Dashboard disabled: failed to start window process: {exc}")
        self._window_failed = True
        self._window_process = None
        self._image_queue = None
        return
    assert self._image_queue is not None
    try:
      if self._image_queue.full():
        try:
          self._image_queue.get_nowait()
        except Exception:
          pass
      self._image_queue.put_nowait(dashboard)
    except Exception as exc:
      print(f"Dashboard disabled: failed to send image frame: {exc}")
      self._window_failed = True
    self._last_update_s = now_s

  def close(self) -> None:
    if self._image_queue is not None:
      try:
        self._image_queue.put_nowait(None)
      except Exception:
        pass
      self._image_queue = None
    if self._window_process is not None:
      self._window_process.join(timeout=0.5)
      if self._window_process.is_alive():
        self._window_process.terminate()
      self._window_process = None


class CameraRobotController:
  def __init__(
    self,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    start_pos: tuple[float, float, float],
    start_yaw_rad: float,
    move_speed: float,
    turn_speed_rad: float,
    camera_width: int = DEFAULT_CAMERA_WIDTH,
    camera_height: int = DEFAULT_CAMERA_HEIGHT,
    depth_max_m: float = DEFAULT_DEPTH_MAX_M,
  ) -> None:
    self.model = model
    self.data = data
    self.pos = list(start_pos)
    self.yaw_rad = start_yaw_rad
    self.start_pos = tuple(start_pos)
    self.start_yaw_rad = start_yaw_rad
    self.move_speed = move_speed
    self.turn_speed_rad = turn_speed_rad
    self.camera_width = camera_width
    self.camera_height = camera_height
    self.depth_max_m = depth_max_m
    self.key_hold_timeout_s = 0.20
    self.last_update_s = time.time()
    self._active_action = "stop"
    self._action_until_s = 0.0
    self._last_key_time = {
      "forward": 0.0,
      "backward": 0.0,
      "strafe_left": 0.0,
      "strafe_right": 0.0,
      "turn_left": 0.0,
      "turn_right": 0.0,
    }

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "camera_robot")
    self._mocap_id = int(model.body_mocapid[body_id])
    self._main_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "main")
    self._rgb_camera_id = mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_CAMERA, "camera_robot_rgb"
    )
    self._depth_camera_id = mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_CAMERA, "camera_robot_depth"
    )
    self._robot_geom_id = mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_GEOM, "camera_robot_marker"
    )
    self._rgb_renderer: mujoco.Renderer | None = None
    self._depth_renderer: mujoco.Renderer | None = None
    self._mj_lock = Lock()
    self._apply_pose()

  @property
  def fovy_rad(self) -> float:
    return math.radians(float(self.model.cam_fovy[self._rgb_camera_id]))

  def _is_key_active(self, key_name: str, now_s: float) -> bool:
    return (now_s - self._last_key_time[key_name]) <= self.key_hold_timeout_s

  def _apply_pose(self) -> None:
    self.data.mocap_pos[self._mocap_id] = self.pos
    self.data.mocap_quat[self._mocap_id] = _yaw_to_quat_wxyz(self.yaw_rad)
    mujoco.mj_forward(self.model, self.data)

  def _robot_in_collision(self) -> bool:
    for contact_idx in range(self.data.ncon):
      contact = self.data.contact[contact_idx]
      if contact.geom1 == self._robot_geom_id or contact.geom2 == self._robot_geom_id:
        return True
    return False

  def _current_action_motion(self, now_s: float) -> tuple[float, float, float]:
    if self._active_action != "stop" and now_s <= self._action_until_s:
      return ACTION_TO_MOTION[self._active_action]
    if now_s > self._action_until_s:
      self._active_action = "stop"

    forward = 1.0 if self._is_key_active("forward", now_s) else 0.0
    backward = 1.0 if self._is_key_active("backward", now_s) else 0.0
    strafe_left = 1.0 if self._is_key_active("strafe_left", now_s) else 0.0
    strafe_right = 1.0 if self._is_key_active("strafe_right", now_s) else 0.0
    turn_left = 1.0 if self._is_key_active("turn_left", now_s) else 0.0
    turn_right = 1.0 if self._is_key_active("turn_right", now_s) else 0.0
    return (
      forward - backward,
      strafe_right - strafe_left,
      turn_left - turn_right,
    )

  def _ensure_renderers(self) -> None:
    if self._rgb_renderer is None:
      try:
        self._rgb_renderer = mujoco.Renderer(
          self.model, width=self.camera_width, height=self.camera_height
        )
      except Exception as exc:
        self._rgb_renderer = None
        raise RuntimeError(
          "Failed to initialize the RGB renderer. Run this demo in a desktop session "
          "with a working OpenGL context."
        ) from exc
    if self._depth_renderer is None:
      try:
        self._depth_renderer = mujoco.Renderer(
          self.model, width=self.camera_width, height=self.camera_height
        )
      except Exception as exc:
        self._depth_renderer = None
        raise RuntimeError(
          "Failed to initialize the depth renderer. Run this demo in a desktop session "
          "with a working OpenGL context."
        ) from exc

  def set_view(self, viewer: mujoco.viewer.Handle, *, first_person: bool) -> None:
    with viewer.lock():
      viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
      viewer.cam.fixedcamid = self._rgb_camera_id if first_person else self._main_camera_id

  def reset_pose(self) -> None:
    self.pos[:] = self.start_pos
    self.yaw_rad = self.start_yaw_rad
    self.stop()
    self._apply_pose()

  def stop(self) -> None:
    self._active_action = "stop"
    self._action_until_s = 0.0
    for action_name in self._last_key_time:
      self._last_key_time[action_name] = 0.0

  def set_action(self, action_name: str, duration_s: float | None = None) -> None:
    if action_name not in ACTION_TO_MOTION:
      raise ValueError(f"Unknown action '{action_name}'")
    self.stop()
    self._active_action = action_name
    self._action_until_s = time.time() + max(0.0, duration_s or 0.0)
    if duration_s is None:
      self._action_until_s = float("inf")

  def get_pose(self) -> dict:
    return {
      "x": float(self.pos[0]),
      "y": float(self.pos[1]),
      "z": float(self.pos[2]),
      "yaw_rad": float(self.yaw_rad),
      "timestamp_s": time.time(),
    }

  def capture_rgb(self) -> np.ndarray:
    self._ensure_renderers()
    assert self._rgb_renderer is not None
    self._rgb_renderer.disable_depth_rendering()
    self._rgb_renderer.update_scene(self.data, camera=self._rgb_camera_id)
    return self._rgb_renderer.render().copy()

  def capture_depth(self) -> np.ndarray:
    self._ensure_renderers()
    assert self._depth_renderer is not None
    self._depth_renderer.enable_depth_rendering()
    self._depth_renderer.update_scene(self.data, camera=self._depth_camera_id)
    depth = self._depth_renderer.render().copy()
    self._depth_renderer.disable_depth_rendering()
    return np.clip(depth, 0.0, self.depth_max_m)

  def capture_rgb_depth(self) -> tuple[np.ndarray, np.ndarray]:
    return self.capture_rgb(), self.capture_depth()

  def on_key(self, key: int, viewer: mujoco.viewer.Handle) -> None:
    now_s = time.time()
    self._active_action = "stop"
    self._action_until_s = 0.0
    if key in (glfw.KEY_UP, glfw.KEY_W):
      self._last_key_time["forward"] = now_s
    elif key in (glfw.KEY_DOWN, glfw.KEY_S):
      self._last_key_time["backward"] = now_s
    elif key == glfw.KEY_A:
      self._last_key_time["strafe_left"] = now_s
    elif key == glfw.KEY_D:
      self._last_key_time["strafe_right"] = now_s
    elif key in (glfw.KEY_LEFT, glfw.KEY_Q):
      self._last_key_time["turn_left"] = now_s
    elif key in (glfw.KEY_RIGHT, glfw.KEY_E):
      self._last_key_time["turn_right"] = now_s
    elif key == glfw.KEY_1:
      self.set_view(viewer, first_person=False)
    elif key == glfw.KEY_2:
      self.set_view(viewer, first_person=True)
    elif key == glfw.KEY_SPACE:
      self.stop()
    elif key == glfw.KEY_R:
      self.reset_pose()

  def step(self) -> None:
    now_s = time.time()
    dt = max(0.0, now_s - self.last_update_s)
    self.last_update_s = now_s

    lin_x_scale, lin_y_scale, yaw_scale = self._current_action_motion(now_s)
    candidate_yaw_rad = self.yaw_rad + self.turn_speed_rad * yaw_scale * dt
    cos_yaw = math.cos(candidate_yaw_rad)
    sin_yaw = math.sin(candidate_yaw_rad)

    forward_step = self.move_speed * lin_x_scale * dt
    strafe_step = self.move_speed * lin_y_scale * dt

    prev_pos = tuple(self.pos)
    prev_yaw_rad = self.yaw_rad
    self.pos[0] += forward_step * cos_yaw + strafe_step * sin_yaw
    self.pos[1] += forward_step * sin_yaw - strafe_step * cos_yaw
    self.yaw_rad = candidate_yaw_rad
    self._apply_pose()
    if self._robot_in_collision():
      self.pos[:] = prev_pos
      self.yaw_rad = prev_yaw_rad
      self._apply_pose()

  def tick(
    self,
    viewer: mujoco.viewer.Handle | None = None,
    dashboard: DashboardWindow | None = None,
  ) -> None:
    step_start = time.time()
    self.step()
    mujoco.mj_step(self.model, self.data)
    if viewer is not None:
      if dashboard is not None:
        dashboard.update(viewer)
      viewer.sync(state_only=True)
    remaining = self.model.opt.timestep - (time.time() - step_start)
    if remaining > 0:
      time.sleep(remaining)

  def close(self) -> None:
    if self._rgb_renderer is not None:
      self._rgb_renderer.close()
      self._rgb_renderer = None
    if self._depth_renderer is not None:
      self._depth_renderer.close()
      self._depth_renderer = None


def build_model(floorplan_xml: Path, fovy: float) -> mujoco.MjModel:
  scene_spec = mujoco.MjSpec()
  _attach_explore_scene(scene_spec, floorplan_xml, fovy)
  return scene_spec.compile()


def resolve_floorplan_xml(floorplan_xml_arg: str) -> Path:
  repo_root = Path(__file__).resolve().parents[2]
  floorplan_xml = (
    Path(floorplan_xml_arg).expanduser()
    if Path(floorplan_xml_arg).is_absolute()
    else (repo_root / floorplan_xml_arg)
  )
  if not floorplan_xml.exists():
    raise FileNotFoundError(f"Floorplan XML not found: {floorplan_xml}")
  return floorplan_xml


def create_camera_robot_from_args(args: argparse.Namespace) -> CameraRobotController:
  model = build_model(floorplan_xml=resolve_floorplan_xml(args.floorplan_xml), fovy=args.fovy)
  data = mujoco.MjData(model)
  return CameraRobotController(
    model,
    data,
    start_pos=(args.start_x, args.start_y, args.start_z),
    start_yaw_rad=math.radians(args.start_yaw_deg),
    move_speed=args.move_speed,
    turn_speed_rad=math.radians(args.turn_speed_deg),
    camera_width=args.camera_width,
    camera_height=args.camera_height,
    depth_max_m=args.depth_max_m,
  )


def configure_viewer(viewer: mujoco.viewer.Handle, model: mujoco.MjModel) -> None:
  model.vis.headlight.active = 0
  model.light_castshadow[:] = 0
  with viewer.lock():
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0


def run_manual(
  args: argparse.Namespace | None = None,
  *,
  detector=None,
) -> None:
  args = args or parse_args()
  controller = create_camera_robot_from_args(args)
  dashboard = DashboardWindow(controller, detector=detector)
  print("Controls: W/Up forward, S/Down backward, A/D strafe, Q/E or Left/Right turn.")
  print("           Space stop, R reset pose, 1 overview camera, 2 camera_robot RGB view.")

  def key_callback(key: int) -> None:
    controller.on_key(key, viewer)

  try:
    with mujoco.viewer.launch_passive(
      controller.model,
      controller.data,
      key_callback=key_callback,
      show_left_ui=False,
      show_right_ui=False,
    ) as viewer:
      configure_viewer(viewer, controller.model)
      controller.set_view(viewer, first_person=True)
      while viewer.is_running():
        controller.tick(viewer, dashboard=dashboard)
  finally:
    dashboard.close()
    controller.close()


def main() -> None:
  run_manual(parse_args())


if __name__ == "__main__":
  main()
