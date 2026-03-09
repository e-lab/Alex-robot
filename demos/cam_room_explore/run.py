#!/usr/bin/env python3
"""Entry point for manual or automatic camera-robot exploration."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import mujoco.viewer

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from demos.cam_room_explore.cam_controller import AutoExploreController, TARGET_OBJECTS, YoloDetector
from demos.cam_room_explore.cam_room_explore import (
  CameraOverlayManager,
  build_arg_parser,
  configure_viewer,
  create_camera_robot_from_args,
  run_manual,
)


def parse_args() -> argparse.Namespace:
  parser = build_arg_parser()
  parser.description = "Run the camera robot in manual mode or prompt-driven automatic mode."
  parser.add_argument("--prompt", default=None, help="Target object label, for example 'door'.")
  parser.add_argument("--yolo-model", default="yolov8n.pt")
  parser.add_argument("--target-labels", nargs="*", default=TARGET_OBJECTS)
  parser.add_argument("--confidence-threshold", type=float, default=0.25)
  return parser.parse_args()


def _run_auto(args: argparse.Namespace) -> None:
  robot = create_camera_robot_from_args(args)
  detector = YoloDetector(
    model_name=args.yolo_model,
    target_labels=args.target_labels,
    confidence_threshold=args.confidence_threshold,
  )
  overlay_manager = CameraOverlayManager(robot, detector=detector)
  auto = AutoExploreController(
    robot,
    detector=detector,
    target_labels=args.target_labels,
    max_depth_m=args.depth_max_m,
    overlay_manager=overlay_manager,
  )

  try:
    with mujoco.viewer.launch_passive(
      robot.model,
      robot.data,
      show_left_ui=False,
      show_right_ui=False,
    ) as viewer:
      configure_viewer(viewer, robot.model)
      robot.set_view(viewer, first_person=True)
      overlay_manager.update(viewer, force=True)
      scene_graph = auto.explore_room(viewer=viewer)
      print(f"Exploration complete. Seen objects: {sorted(scene_graph['object_index'].keys())}")
      print(f"Occupancy map summary: {scene_graph['occupancy_map']}")

      active_target = args.prompt
      while viewer.is_running():
        if active_target:
          success = auto.walk_to_target(active_target, viewer=viewer)
          print(f"walk_to_target('{active_target}') -> {success}")
        next_target = input("Enter another target label (blank to quit): ").strip()
        if not next_target:
          break
        active_target = next_target
  finally:
    robot.close()


def main() -> None:
  args = parse_args()
  if not args.prompt:
    detector = None
    try:
      detector = YoloDetector(
        model_name=args.yolo_model,
        target_labels=args.target_labels,
        confidence_threshold=args.confidence_threshold,
      )
    except Exception as exc:
      print(f"YOLO overlay disabled: {exc}")
    run_manual(args, detector=detector)
    return
  _run_auto(args)


if __name__ == "__main__":
  main()
