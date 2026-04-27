#!/usr/bin/env python3
"""Load Alex in the room scene and allow mouse-based placement."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
from mujoco.glfw import glfw

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from alex_models import alex_sensors

DEFAULT_SCENE_XML = _REPO_ROOT / "scenes" / "alex_scenes" / "scene_alex_v1_full_body_mjx_room1.xml"
DEFAULT_DOOR_NAME = "door_f2d4ffc256f54f0897c1add29e9536e8_1_1_0"
DEFAULT_STANDOFF_M = 0.9
DEFAULT_BASE_Z_M = 1.0
WINDOW_WIDTH = 1440
WINDOW_HEIGHT = 900
HEAD_VIEW_WIDTH = 320
HEAD_VIEW_HEIGHT = 180
HEAD_VIEW_MAX_DEPTH_M = 5.0


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Load Alex and place it in the room with mouse dragging.",
  )
  parser.add_argument(
    "--scene-xml",
    default=str(DEFAULT_SCENE_XML),
    help="Path to the MuJoCo scene XML.",
  )
  parser.add_argument(
    "--door-body",
    default=DEFAULT_DOOR_NAME,
    help="Door body name or suffix for the initial placement.",
  )
  parser.add_argument(
    "--standoff-m",
    type=float,
    default=DEFAULT_STANDOFF_M,
    help="Initial distance from the door surface.",
  )
  parser.add_argument(
    "--base-z-m",
    type=float,
    default=DEFAULT_BASE_Z_M,
    help="Robot base height for the free joint.",
  )
  parser.add_argument(
    "--print-doors",
    action="store_true",
    help="List available door bodies and exit.",
  )
  return parser.parse_args()


def _find_body_id_by_name_suffix(model: mujoco.MjModel, target_name: str) -> int:
  body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_name)
  if body_id >= 0:
    return body_id
  for idx in range(model.nbody):
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, idx)
    if body_name == target_name or (body_name is not None and body_name.endswith("/" + target_name)):
      return idx
  return -1


def _door_body_names(model: mujoco.MjModel) -> list[str]:
  names: list[str] = []
  for idx in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, idx)
    if name and "door_" in name:
      names.append(name)
  return names


def _resolve_door_geom_id(model: mujoco.MjModel, door_body_id: int) -> int:
  prefix = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, door_body_id)
  if prefix is None:
    raise ValueError(f"Body id {door_body_id} has no name.")

  for geom_id in range(model.ngeom):
    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    if geom_name and geom_name.startswith(prefix) and "collision" in geom_name:
      return geom_id
  for geom_id in range(model.ngeom):
    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    if geom_name and geom_name.startswith(prefix):
      return geom_id
  raise ValueError(f"No geom found for door body '{prefix}'.")


def _quat_from_yaw(yaw_rad: float) -> tuple[float, float, float, float]:
  half = 0.5 * yaw_rad
  return (math.cos(half), 0.0, 0.0, math.sin(half))


def _compute_spawn_pose(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  door_body_name: str,
  standoff_m: float,
  base_z_m: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float], str]:
  door_body_id = _find_body_id_by_name_suffix(model, door_body_name)
  if door_body_id < 0:
    available = ", ".join(_door_body_names(model))
    raise ValueError(f"Door body '{door_body_name}' not found. Available: {available}")

  geom_id = _resolve_door_geom_id(model, door_body_id)
  geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom-{geom_id}"
  geom_center = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
  geom_axes = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
  normal_xy = geom_axes[:2, 2]
  normal_xy_norm = np.linalg.norm(normal_xy)
  if normal_xy_norm < 1e-6:
    raise ValueError(f"Door geom '{geom_name}' has no usable horizontal normal.")
  normal_xy = normal_xy / normal_xy_norm

  spawn_xy = geom_center[:2] + normal_xy * standoff_m
  facing_xy = geom_center[:2] - spawn_xy
  yaw_rad = math.atan2(float(facing_xy[1]), float(facing_xy[0]))
  pos_xyz = (float(spawn_xy[0]), float(spawn_xy[1]), float(base_z_m))
  quat_wxyz = _quat_from_yaw(yaw_rad)
  return pos_xyz, quat_wxyz, geom_name


class InteractivePlacementViewer:
  def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, base_z_m: float):
    self.model = model
    self.data = data
    self.base_z_m = base_z_m

    self.window = None
    self.cam = mujoco.MjvCamera()
    self.opt = mujoco.MjvOption()
    self.pert = mujoco.MjvPerturb()
    self.scn = mujoco.MjvScene(model, maxgeom=20_000)
    self.ctx = None
    self.viewport = mujoco.MjrRect(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
    self.head_renderer = None
    self.head_camera_ids = None

    self.last_x = 0.0
    self.last_y = 0.0
    self.left_drag_active = False
    self.right_drag_active = False
    self.middle_drag_active = False

    mujoco.mjv_defaultFreeCamera(model, self.cam)
    self.main_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "main")
    if self.main_camera_id >= 0:
      self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
      self.cam.fixedcamid = self.main_camera_id

  def run(self) -> None:
    if not glfw.init():
      raise RuntimeError("Failed to initialize GLFW.")

    try:
      self.window = glfw.create_window(
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        "Alex Open Door Placement",
        None,
        None,
      )
      if self.window is None:
        raise RuntimeError("Failed to create GLFW window.")

      glfw.make_context_current(self.window)
      glfw.swap_interval(1)
      self.ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
      self.head_camera_ids = alex_sensors.resolve_alex_camera_ids(self.model)
      self.head_renderer = mujoco.Renderer(
        self.model,
        width=HEAD_VIEW_WIDTH,
        height=HEAD_VIEW_HEIGHT,
      )
      glfw.set_cursor_pos_callback(self.window, self._cursor_pos_callback)
      glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
      glfw.set_scroll_callback(self.window, self._scroll_callback)
      glfw.set_key_callback(self.window, self._key_callback)

      print("Controls:")
      print("  Left drag: move Alex on the scene")
      print("  Right drag / Middle drag / Mouse wheel: camera controls")
      print("  Q / E: rotate Alex in place")
      print("  R: reset to the scene main camera")
      print("  Alex RGB/depth views open in separate windows")
      print("  Esc: quit")

      while not glfw.window_should_close(self.window):
        glfw.make_context_current(self.window)
        width, height = glfw.get_framebuffer_size(self.window)
        self.viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.ctx)
        mujoco.mjv_updateScene(
          self.model,
          self.data,
          self.opt,
          self.pert,
          self.cam,
          mujoco.mjtCatBit.mjCAT_ALL.value,
          self.scn,
        )
        mujoco.mjr_render(self.viewport, self.scn, self.ctx)
        self._show_head_camera_views()
        glfw.swap_buffers(self.window)
        glfw.poll_events()
        if cv2.waitKey(1) & 0xFF == 27:
          glfw.set_window_should_close(self.window, True)
    finally:
      if self.head_renderer is not None:
        self.head_renderer.close()
      cv2.destroyAllWindows()
      if self.window is not None:
        glfw.destroy_window(self.window)
      glfw.terminate()

  def _show_head_camera_views(self) -> None:
    if self.head_renderer is None or self.head_camera_ids is None:
      return
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.ctx)
    rgb_bgr, depth_bgr = alex_sensors.render_alex_rgb_depth(
      renderer=self.head_renderer,
      data=self.data,
      camera_ids=self.head_camera_ids,
      max_depth_m=HEAD_VIEW_MAX_DEPTH_M,
    )
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.ctx)
    cv2.imshow("Alex Head RGB", rgb_bgr)
    cv2.imshow("Alex Head Depth", depth_bgr)

  def _place_robot_at_cursor(self, xpos: float, ypos: float) -> None:
    if self.viewport.width <= 0 or self.viewport.height <= 0:
      return

    relx = xpos / self.viewport.width
    rely = (self.viewport.height - ypos) / self.viewport.height
    aspect_ratio = self.viewport.width / max(1, self.viewport.height)
    selpnt = np.zeros(3, dtype=np.float64)
    geomid = np.array([-1], dtype=np.int32)
    flexid = np.array([-1], dtype=np.int32)
    skinid = np.array([-1], dtype=np.int32)

    body_id = mujoco.mjv_select(
      self.model,
      self.data,
      self.opt,
      aspect_ratio,
      relx,
      rely,
      self.scn,
      selpnt,
      geomid,
      flexid,
      skinid,
    )
    if body_id < 0:
      return

    current_quat = tuple(float(v) for v in self.data.qpos[3:7])
    alex_sensors.set_base_pose(
      self.model,
      self.data,
      pos_xyz=(float(selpnt[0]), float(selpnt[1]), float(self.base_z_m)),
      quat_wxyz=current_quat,
      forward=True,
    )

  def _cursor_pos_callback(self, _window, xpos: float, ypos: float) -> None:
    dx = xpos - self.last_x
    dy = ypos - self.last_y
    self.last_x = xpos
    self.last_y = ypos

    if self.left_drag_active:
      self._place_robot_at_cursor(xpos, ypos)
      return

    if not (self.right_drag_active or self.middle_drag_active):
      return
    if self.cam.type == mujoco.mjtCamera.mjCAMERA_FIXED:
      return

    width = max(1, self.viewport.width)
    height = max(1, self.viewport.height)
    reldx = dx / height
    reldy = dy / height
    shift_pressed = (
      glfw.get_key(self.window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
      or glfw.get_key(self.window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    )

    if self.middle_drag_active or shift_pressed:
      action = mujoco.mjtMouse.mjMOUSE_MOVE_H if abs(dx) >= abs(dy) else mujoco.mjtMouse.mjMOUSE_MOVE_V
    else:
      action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if abs(dx) >= abs(dy) else mujoco.mjtMouse.mjMOUSE_ROTATE_V

    mujoco.mjv_moveCamera(self.model, action, reldx, reldy, self.scn, self.cam)

  def _mouse_button_callback(self, window, button: int, action: int, _mods: int) -> None:
    xpos, ypos = glfw.get_cursor_pos(window)
    self.last_x = xpos
    self.last_y = ypos

    is_press = action == glfw.PRESS
    if button == glfw.MOUSE_BUTTON_LEFT:
      self.left_drag_active = is_press
      if is_press:
        self._place_robot_at_cursor(xpos, ypos)
    elif button == glfw.MOUSE_BUTTON_RIGHT:
      self.right_drag_active = is_press
    elif button == glfw.MOUSE_BUTTON_MIDDLE:
      self.middle_drag_active = is_press

  def _scroll_callback(self, _window, _xoffset: float, yoffset: float) -> None:
    if self.cam.type == mujoco.mjtCamera.mjCAMERA_FIXED:
      return
    mujoco.mjv_moveCamera(
      self.model,
      mujoco.mjtMouse.mjMOUSE_ZOOM,
      0.0,
      -0.05 * yoffset,
      self.scn,
      self.cam,
    )

  def _rotate_robot(self, delta_yaw_rad: float) -> None:
    quat = np.asarray(self.data.qpos[3:7], dtype=np.float64)
    yaw_rad = math.atan2(
      2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
      1.0 - 2.0 * (quat[2] * quat[2] + quat[3] * quat[3]),
    )
    new_quat = _quat_from_yaw(yaw_rad + delta_yaw_rad)
    alex_sensors.set_base_pose(
      self.model,
      self.data,
      pos_xyz=tuple(float(v) for v in self.data.qpos[:3]),
      quat_wxyz=new_quat,
      forward=True,
    )

  def _key_callback(self, window, key: int, _scancode: int, action: int, _mods: int) -> None:
    if action not in (glfw.PRESS, glfw.REPEAT):
      return
    if key == glfw.KEY_ESCAPE:
      glfw.set_window_should_close(window, True)
    elif key == glfw.KEY_R:
      if self.main_camera_id >= 0:
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = self.main_camera_id
      else:
        mujoco.mjv_defaultFreeCamera(self.model, self.cam)
    elif key == glfw.KEY_Q:
      self._rotate_robot(+0.1)
    elif key == glfw.KEY_E:
      self._rotate_robot(-0.1)


def main() -> None:
  args = parse_args()
  model = mujoco.MjModel.from_xml_path(str(Path(args.scene_xml).resolve()))

  if args.print_doors:
    for name in _door_body_names(model):
      print(name)
    return

  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)

  pos_xyz, quat_wxyz, geom_name = _compute_spawn_pose(
    model=model,
    data=data,
    door_body_name=args.door_body,
    standoff_m=args.standoff_m,
    base_z_m=args.base_z_m,
  )
  alex_sensors.set_base_pose(
    model,
    data,
    pos_xyz=pos_xyz,
    quat_wxyz=quat_wxyz,
    forward=True,
  )

  print(f"Scene: {Path(args.scene_xml).resolve()}")
  print(f"Initial door body: {args.door_body}")
  print(f"Initial door geom: {geom_name}")
  print(f"Initial robot pose: pos={pos_xyz}, quat={quat_wxyz}")

  InteractivePlacementViewer(model, data, args.base_z_m).run()


if __name__ == "__main__":
  main()
