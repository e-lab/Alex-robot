#!/usr/bin/env python3
"""Automatic exploration and target seeking for the camera robot."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from demos.cam_room_explore.cam_room_explore import CameraOverlayManager, CameraRobotController

try:
  from ultralytics import YOLO
except Exception:
  YOLO = None


TARGET_OBJECTS = ["person", "dining table", "microwave", "oven", "toaster", "sink"]


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
      self._model = YOLO(model_name)
    except Exception as exc:
      raise RuntimeError(
        f"Failed to load YOLO model '{model_name}'. Provide a local weights path or "
        "ensure the default model is already available."
      ) from exc

  def detect(self, rgb_image: np.ndarray) -> list[dict]:
    result = self._model.predict(rgb_image, verbose=False)[0]
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
class OccupancyMap2D:
  resolution_m: float = 0.10
  free_weight: int = -1
  occupied_weight: int = 3

  def __post_init__(self) -> None:
    self.cells: dict[tuple[int, int], int] = {}

  def _cell(self, x_m: float, y_m: float) -> tuple[int, int]:
    return (
      int(math.floor(x_m / self.resolution_m)),
      int(math.floor(y_m / self.resolution_m)),
    )

  def _mark(self, cell: tuple[int, int], weight: int) -> None:
    self.cells[cell] = self.cells.get(cell, 0) + weight

  def _raytrace(self, start_xy: tuple[float, float], end_xy: tuple[float, float]) -> None:
    dx = end_xy[0] - start_xy[0]
    dy = end_xy[1] - start_xy[1]
    dist = math.hypot(dx, dy)
    if dist <= 1e-6:
      return
    steps = max(1, int(dist / self.resolution_m))
    for idx in range(steps):
      alpha = idx / steps
      sample = (
        start_xy[0] + alpha * dx,
        start_xy[1] + alpha * dy,
      )
      self._mark(self._cell(*sample), self.free_weight)
    self._mark(self._cell(*end_xy), self.occupied_weight)

  def update_from_depth(
    self,
    pose: dict,
    depth_image: np.ndarray,
    *,
    fovy_rad: float,
    max_depth_m: float,
  ) -> None:
    height, width = depth_image.shape
    cx = (width - 1) * 0.5
    fy = 0.5 * height / math.tan(0.5 * fovy_rad)
    fx = fy
    robot_xy = (float(pose["x"]), float(pose["y"]))
    yaw_rad = float(pose["yaw_rad"])

    for col in range(0, width, 6):
      band = depth_image[height // 3: (2 * height) // 3, col]
      valid = band[np.isfinite(band)]
      valid = valid[(valid > 0.15) & (valid < max_depth_m)]
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

  def summary(self) -> dict:
    occupied = [cell for cell, score in self.cells.items() if score > 0]
    return {
      "num_cells": len(self.cells),
      "num_occupied_cells": len(occupied),
      "resolution_m": self.resolution_m,
    }


class AutoExploreController:
  def __init__(
    self,
    robot: CameraRobotController,
    *,
    detector: YoloDetector,
    target_labels: list[str] | None = None,
    max_depth_m: float | None = None,
    overlay_manager: CameraOverlayManager | None = None,
  ) -> None:
    self.robot = robot
    self.detector = detector
    self.target_labels = target_labels or TARGET_OBJECTS
    self.max_depth_m = max_depth_m or robot.depth_max_m
    self.overlay_manager = overlay_manager
    self.scene_graph = {
      "views": [],
      "object_index": {},
      "occupancy_map": {},
    }
    self.occupancy_map = OccupancyMap2D()

  def _tick_for_duration(self, duration_s: float, viewer=None) -> None:
    end_time = time.time() + duration_s
    while time.time() < end_time:
      if viewer is not None and not viewer.is_running():
        break
      self.robot.tick(viewer, overlay_manager=self.overlay_manager)

  def _run_action(self, action_name: str, duration_s: float, viewer=None) -> None:
    self.robot.set_action(action_name, duration_s)
    self._tick_for_duration(duration_s, viewer)
    self.robot.stop()

  def _observe_scene(self) -> dict:
    rgb_image, depth_image = self.robot.capture_rgb_depth()
    detections = self.detector.detect(rgb_image)
    if self.overlay_manager is not None:
      self.overlay_manager.set_latest_capture(rgb_image, depth_image, detections)
    pose = self.robot.get_pose()
    self.occupancy_map.update_from_depth(
      pose,
      depth_image,
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
    self.scene_graph["occupancy_map"] = self.occupancy_map.summary()
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

    self._move_toward_xy(best_view["robot_xy"][0], best_view["robot_xy"][1], viewer)

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
