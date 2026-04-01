#!/usr/bin/env python3
"""Automatic exploration and target seeking for the camera robot."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from demos.cam_room_explore.cam_room_explore import (
  CameraRobotController,
  DashboardWindow,
  POINT_CLOUD_SAMPLE_STRIDE_PX,
  POINT_CLOUD_VOXEL_SIZE_M,
)

try:
  from ultralytics import YOLO
except Exception:
  YOLO = None


TARGET_OBJECTS = [
  "door_panel", "door_lever", "door_knob", "door_push_bar", "door_pull_handle",
  "person", "trash_can", "bottle", "storage_container", "traffic_barrier",
]


def _normalize_angle(angle_rad: float) -> float:
  return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


class YoloDetector:
  def __init__(
    self,
    model_name: str = "yolov8n.pt",
    target_labels: list[str] | None = None,
    confidence_threshold: float = 0.25,
  ) -> None:
    self.model_name = model_name
    self.target_labels = set(target_labels or TARGET_OBJECTS)
    self.confidence_threshold = confidence_threshold
    if YOLO is None:
      raise RuntimeError(
        "ultralytics is not available. Install it or provide an environment with YOLOv8."
      )
    try:
      import onnx as _onnx
      _m = _onnx.load(model_name)
      _meta = {p.key: p.value for p in _m.metadata_props}
      _task = _meta.get("task", "detect")
      _imgsz = _meta.get("imgsz", None)
      self._imgsz = eval(_imgsz) if _imgsz else None  # e.g. [736, 1280]
    except Exception:
      _task = "detect"
      self._imgsz = None
    self._model = YOLO(model_name, task=_task)

  def detect(self, rgb_image: np.ndarray) -> list[dict]:
    kwargs = {"verbose": False}
    if self._imgsz is not None:
      kwargs["imgsz"] = self._imgsz
    result = self._model.predict(rgb_image, **kwargs)[0]
    detections: list[dict] = []
    names = result.names
    for box in result.boxes:
      cls_id = int(box.cls.item())
      label = names[cls_id]
      conf = float(box.conf.item())
      if conf < self.confidence_threshold:
        continue
      if self.target_labels and label not in self.target_labels:
        continue
      x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
      detections.append(
        {
          "label": label,
          "confidence": conf,
          "bbox_xyxy": [x1, y1, x2, y2],
          "image_center_uv": [(x1 + x2) * 0.5, (y1 + y2) * 0.5],
          "bbox_area": max(0.0, x2 - x1) * max(0.0, y2 - y1),
        }
      )
    detections.sort(key=lambda det: det["confidence"], reverse=True)
    return detections


@dataclass
class FusedPointCloudMap:
  voxel_size_m: float = POINT_CLOUD_VOXEL_SIZE_M

  def __post_init__(self) -> None:
    self.voxels: dict[tuple[int, int, int], int] = {}

  def _voxel(self, x_m: float, y_m: float, z_m: float) -> tuple[int, int, int]:
    return (
      int(math.floor(x_m / self.voxel_size_m)),
      int(math.floor(y_m / self.voxel_size_m)),
      int(math.floor(z_m / self.voxel_size_m)),
    )

  def _integrate_point(self, point_xyz: tuple[float, float, float]) -> None:
    voxel = self._voxel(*point_xyz)
    self.voxels[voxel] = self.voxels.get(voxel, 0) + 1

  def update_from_depth(
    self,
    pose: dict,
    depth_image: np.ndarray,
    *,
    camera_local_pos: tuple[float, float, float],
    fovy_rad: float,
    max_depth_m: float,
  ) -> None:
    height, width = depth_image.shape
    cx = (width - 1) * 0.5
    fy = 0.5 * height / math.tan(0.5 * fovy_rad)
    fx = fy
    robot_x = float(pose["x"])
    robot_y = float(pose["y"])
    robot_z = float(pose["z"])
    yaw_rad = float(pose["yaw_rad"])
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    cy = (height - 1) * 0.5
    min_depth_m = 0.15
    max_height_m = robot_z + camera_local_pos[2] + 0.75

    for row in range(height // 4, height, POINT_CLOUD_SAMPLE_STRIDE_PX):
      for col in range(0, width, POINT_CLOUD_SAMPLE_STRIDE_PX):
        depth_m = float(depth_image[row, col])
        if not np.isfinite(depth_m) or depth_m <= min_depth_m or depth_m >= max_depth_m:
          continue
        local_forward_m = depth_m
        local_right_m = ((col - cx) / fx) * depth_m
        local_up_m = ((cy - row) / fy) * depth_m
        local_x = camera_local_pos[0] + local_forward_m
        local_y = camera_local_pos[1] + local_right_m
        local_z = camera_local_pos[2] + local_up_m
        world_x = robot_x + local_x * cos_yaw + local_y * sin_yaw
        world_y = robot_y + local_x * sin_yaw - local_y * cos_yaw
        world_z = robot_z + local_z
        if world_z <= robot_z + 0.02 or world_z >= max_height_m:
          continue
        self._integrate_point((world_x, world_y, world_z))

  def summary(self) -> dict:
    projected_xy = {(vx, vy) for (vx, vy, _vz) in self.voxels}
    return {
      "num_voxels": len(self.voxels),
      "num_xy_cells": len(projected_xy),
      "voxel_size_m": self.voxel_size_m,
    }


class AutoExploreController:
  def __init__(
    self,
    robot: CameraRobotController,
    *,
    detector: YoloDetector,
    target_labels: list[str] | None = None,
    max_depth_m: float | None = None,
    dashboard: DashboardWindow | None = None,
  ) -> None:
    self.robot = robot
    self.detector = detector
    self.target_labels = target_labels or TARGET_OBJECTS
    self.max_depth_m = max_depth_m or robot.depth_max_m
    self.dashboard = dashboard
    self.scene_graph = {
      "views": [],
      "object_index": {},
      "point_cloud_map": {},
    }
    self.point_cloud_map = FusedPointCloudMap()

  def _project_detection_to_world_xy(
    self,
    detection: dict,
    depth_image: np.ndarray,
    pose: dict,
  ) -> tuple[float, float] | None:
    height, width = depth_image.shape
    x1, y1, x2, y2 = detection["bbox_xyxy"]
    x1_i = int(np.clip(math.floor(x1), 0, width - 1))
    x2_i = int(np.clip(math.ceil(x2), 0, width - 1))
    y1_i = int(np.clip(math.floor(y1), 0, height - 1))
    y2_i = int(np.clip(math.ceil(y2), 0, height - 1))
    if x2_i < x1_i or y2_i < y1_i:
      return None

    patch = depth_image[y1_i:y2_i + 1, x1_i:x2_i + 1]
    valid = patch[np.isfinite(patch)]
    valid = valid[(valid > 0.15) & (valid < self.max_depth_m)]
    if valid.size == 0:
      return None

    depth_m = float(np.median(valid))
    cx = (width - 1) * 0.5
    fy = 0.5 * height / math.tan(0.5 * self.robot.fovy_rad)
    fx = fy
    center_u = float(detection["image_center_uv"][0])
    local_forward_m = depth_m
    local_right_m = ((center_u - cx) / fx) * depth_m
    cam_offset = self.robot.depth_camera_local_pos
    local_x = cam_offset[0] + local_forward_m
    local_y = cam_offset[1] + local_right_m
    yaw_rad = float(pose["yaw_rad"])
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    world_x = float(pose["x"]) + local_x * cos_yaw + local_y * sin_yaw
    world_y = float(pose["y"]) + local_x * sin_yaw - local_y * cos_yaw
    return (world_x, world_y)

  def _tick_for_duration(self, duration_s: float, viewer=None) -> None:
    end_time = time.time() + duration_s
    while time.time() < end_time:
      if viewer is not None and not viewer.is_running():
        break
      self.robot.tick(viewer, dashboard=self.dashboard)

  def _run_action(self, action_name: str, duration_s: float, viewer=None) -> None:
    self.robot.set_action(action_name, duration_s)
    self._tick_for_duration(duration_s, viewer)
    self.robot.stop()

  def _observe_scene(self) -> dict:
    rgb_image, depth_image = self.robot.capture_rgb_depth()
    detections = self.detector.detect(rgb_image)
    pose = self.robot.get_pose()
    for detection in detections:
      world_xy = self._project_detection_to_world_xy(detection, depth_image, pose)
      if world_xy is not None:
        detection["world_xy"] = [world_xy[0], world_xy[1]]
    self.point_cloud_map.update_from_depth(
      pose,
      depth_image,
      camera_local_pos=self.robot.depth_camera_local_pos,
      fovy_rad=self.robot.fovy_rad,
      max_depth_m=self.max_depth_m,
    )

    view_id = len(self.scene_graph["views"])
    view_entry = {
      "view_id": view_id,
      "robot_xy": [pose["x"], pose["y"]],
      "robot_z": pose["z"],
      "yaw_rad": pose["yaw_rad"],
      "timestamp_s": pose["timestamp_s"],
      "objects": detections,
    }
    self.scene_graph["views"].append(view_entry)
    for detection in detections:
      self.scene_graph["object_index"].setdefault(detection["label"], []).append(view_id)
    self.scene_graph["point_cloud_map"] = self.point_cloud_map.summary()
    if self.dashboard is not None:
      self.dashboard.set_map_state(
        point_cloud_voxels=self.point_cloud_map.voxels,
        voxel_size_m=self.point_cloud_map.voxel_size_m,
        object_index=self.scene_graph["object_index"],
        views=self.scene_graph["views"],
      )
    return view_entry

  def explore_room(
    self,
    *,
    num_scan_steps: int = 16,
    turn_step_s: float = 0.25,
    viewer=None,
  ) -> dict:
    for _ in range(num_scan_steps):
      self._observe_scene()
      self._run_action("turn_right", turn_step_s, viewer)

    if not any(label in self.scene_graph["object_index"] for label in self.target_labels):
      self._run_action("forward", 0.8, viewer)
      for _ in range(num_scan_steps):
        self._observe_scene()
        self._run_action("turn_left", turn_step_s, viewer)

    self.robot.stop()
    return self.scene_graph

  def _best_view_for_label(self, target_label: str) -> dict | None:
    view_ids = self.scene_graph["object_index"].get(target_label, [])
    best_view = None
    best_score = -1.0
    for view_id in view_ids:
      view = self.scene_graph["views"][view_id]
      target_dets = [obj for obj in view["objects"] if obj["label"] == target_label]
      if not target_dets:
        continue
      score = max(det["confidence"] for det in target_dets)
      if score > best_score:
        best_score = score
        best_view = view
    return best_view

  def _turn_toward(self, target_yaw_rad: float, viewer=None) -> None:
    for _ in range(80):
      yaw_err = _normalize_angle(target_yaw_rad - self.robot.get_pose()["yaw_rad"])
      if abs(yaw_err) < 0.08:
        break
      action = "turn_left" if yaw_err > 0.0 else "turn_right"
      self._run_action(action, min(0.15, abs(yaw_err) / max(self.robot.turn_speed_rad, 1e-6)), viewer)

  def _move_toward_xy(self, target_x: float, target_y: float, viewer=None) -> None:
    for _ in range(160):
      pose = self.robot.get_pose()
      dx = target_x - pose["x"]
      dy = target_y - pose["y"]
      dist = math.hypot(dx, dy)
      if dist < 0.35:
        break
      self._turn_toward(math.atan2(dy, dx), viewer)
      self._run_action("forward", min(0.35, dist / max(self.robot.move_speed, 1e-6)), viewer)

  def _find_target_in_current_view(self, target_label: str) -> dict | None:
    view = self._observe_scene()
    targets = [obj for obj in view["objects"] if obj["label"] == target_label]
    if not targets:
      return None
    targets.sort(key=lambda det: det["confidence"], reverse=True)
    return targets[0]

  def walk_to_target(self, target_label: str, viewer=None) -> bool:
    best_view = self._best_view_for_label(target_label)
    if best_view is None:
      return False

    target_dets = [obj for obj in best_view["objects"] if obj["label"] == target_label]
    target_dets = [obj for obj in target_dets if "world_xy" in obj]
    if target_dets:
      target_dets.sort(key=lambda det: det["confidence"], reverse=True)
      target_xy = target_dets[0]["world_xy"]
    else:
      target_xy = best_view["robot_xy"]
    self._move_toward_xy(target_xy[0], target_xy[1], viewer)

    image_width = float(self.robot.camera_width)
    for _ in range(24):
      detection = self._find_target_in_current_view(target_label)
      if detection is None:
        self.robot.stop()
        continue

      center_u = float(detection["image_center_uv"][0])
      depth_image = self.robot.capture_depth()
      u = int(np.clip(round(center_u), 0, depth_image.shape[1] - 1))
      v = int(np.clip(round(detection["image_center_uv"][1]), 0, depth_image.shape[0] - 1))
      local_patch = depth_image[max(0, v - 3): v + 4, max(0, u - 3): u + 4]
      valid = local_patch[np.isfinite(local_patch)]
      valid = valid[(valid > 0.15) & (valid < self.max_depth_m)]
      depth_to_target = float(np.median(valid)) if valid.size else 2.0
      if depth_to_target <= 1.0 or detection["bbox_area"] >= 0.15 * (
        self.robot.camera_width * self.robot.camera_height
      ):
        self.robot.stop()
        return True

      center_err = (center_u - image_width * 0.5) / image_width
      if abs(center_err) > 0.08:
        self._run_action("turn_right" if center_err > 0.0 else "turn_left", 0.12, viewer)
        continue

      self._run_action("forward", min(0.30, depth_to_target / 4.0), viewer)

    self.robot.stop()
    return False
