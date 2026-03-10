#!/usr/bin/env python3
"""Entry point for manual or automatic camera-robot exploration."""

from __future__ import annotations

import argparse
from pathlib import Path
from queue import Queue
import sys
from threading import Thread

import mujoco.viewer

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from demos.cam_room_explore.cam_controller import AutoExploreController, TARGET_OBJECTS, YoloDetector
from demos.cam_room_explore.cam_room_explore import (
  build_arg_parser,
  configure_viewer,
  create_camera_robot_from_args,
  run_manual,
  DashboardWindow,
)


def parse_args() -> argparse.Namespace:
  parser = build_arg_parser()
  parser.description = "Run the camera robot in manual mode or prompt-driven automatic mode."
  parser.add_argument("--prompt", default=None, help="Target object label, for example 'door'.")
  parser.add_argument("--yolo-model", default="yolov8n.pt")
  parser.add_argument("--target-labels", nargs="*", default=TARGET_OBJECTS)
  parser.add_argument("--confidence-threshold", type=float, default=0.25)
  return parser.parse_args()


def _start_prompt_thread(
  target_label: str,
  scene_graph: dict,
  response_queue: Queue[tuple[str, str | None]],
) -> None:
  found = target_label in scene_graph.get("object_index", {})
  seen_objects = sorted(scene_graph.get("object_index", {}).keys())

  def worker() -> None:
    print(f"Seen objects: {seen_objects}")
    print(f"Target '{target_label}' {'found' if found else 'not found'}.")
    choice = input(
      "Choose: (1) walk to object, (2) search for another object, blank to quit: "
    ).strip()
    if not choice:
      response_queue.put(("quit", None))
      return
    if choice == "1":
      if not found:
        print(f"Cannot walk: '{target_label}' was not found in the scene graph.")
        response_queue.put(("prompt_again", target_label))
        return
      response_queue.put(("walk", target_label))
      return
    if choice == "2":
      next_target = input("Which object should I look for next? ").strip()
      if not next_target:
        response_queue.put(("quit", None))
        return
      response_queue.put(("search", next_target))
      return
    print("Unrecognized choice.")
    response_queue.put(("prompt_again", target_label))

  Thread(target=worker, daemon=True).start()


def _run_auto(args: argparse.Namespace) -> None:
  robot = create_camera_robot_from_args(args)
  detector = YoloDetector(
    model_name=args.yolo_model,
    target_labels=args.target_labels,
    confidence_threshold=args.confidence_threshold,
  )
  dashboard = DashboardWindow(robot, detector=detector)
  auto = AutoExploreController(
    robot,
    detector=detector,
    target_labels=args.target_labels,
    max_depth_m=args.depth_max_m,
    dashboard=dashboard,
  )
  target_label = args.prompt
  response_queue: Queue[tuple[str, str | None]] = Queue()
  prompt_active = False
  phase = "explore"

  try:
    with mujoco.viewer.launch_passive(
      robot.model,
      robot.data,
      show_left_ui=False,
      show_right_ui=False,
    ) as viewer:
      configure_viewer(viewer, robot.model)
      robot.set_view(viewer, first_person=True)

      while viewer.is_running():
        if phase == "explore":
          scene_graph = auto.explore_room(viewer=viewer)
          print(f"Exploration complete. Seen objects: {sorted(scene_graph['object_index'].keys())}")
          print(f"Occupancy map summary: {scene_graph['occupancy_map']}")
          phase = "prompt"
          prompt_active = False
          continue

        if phase == "prompt":
          if not prompt_active:
            _start_prompt_thread(target_label, auto.scene_graph, response_queue)
            prompt_active = True
          if not response_queue.empty():
            action, value = response_queue.get()
            prompt_active = False
            if action == "quit":
              break
            if action == "walk" and value is not None:
              phase = "walk"
            elif action == "search" and value is not None:
              target_label = value
              auto.scene_graph = {"views": [], "object_index": {}, "occupancy_map": {}}
              auto.occupancy_map.cells.clear()
              phase = "explore"
            else:
              phase = "prompt"
          robot.tick(viewer, dashboard=dashboard)
          continue

        if phase == "walk":
          success = auto.walk_to_target(target_label, viewer=viewer)
          print(f"walk_to_target('{target_label}') -> {success}")
          phase = "prompt"
          prompt_active = False
          continue

        robot.tick(viewer, dashboard=dashboard)
  finally:
    dashboard.close()
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
      print(f"YOLO dashboard detections disabled: {exc}")
    run_manual(args, detector=detector)
    return
  _run_auto(args)


if __name__ == "__main__":
  main()
