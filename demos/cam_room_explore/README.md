# Camera Room Explore

MuJoCo demo for driving a simple camera robot around an iTHOR room and optionally running YOLO-based object search.

## Files

- `run.py`: main entry point for manual mode and prompt-driven auto exploration
- `cam_room_explore.py`: scene setup, viewer integration, keyboard controls, RGB/depth capture
- `cam_controller.py`: YOLO detector, occupancy map, room scan, and target approach logic

## Requirements

Base dependencies:

```bash
pip install mujoco numpy
```

Optional object detection:

```bash
pip install ultralytics
```

Notes:

- The demo expects a desktop session with a working OpenGL context.
- The default scene is `scenes/ithor/FloorPlan1_physics_simple.xml`.
- In prompt mode, YOLO is required because exploration and target seeking depend on detections.

## Run

From the repo root:

```bash
python run.py
```

This starts manual mode with the first-person camera view enabled.

Prompt-driven exploration:

```bash
python run.py --prompt oven
```

After the scan finishes, the script prints seen objects and asks whether to walk to the target, search for another object, or quit.

## Manual Controls

- `W` or `Up`: move forward
- `S` or `Down`: move backward
- `A` / `D`: strafe left / right
- `Q` / `E` or `Left` / `Right`: turn left / right
- `Space`: stop
- `R`: reset to the start pose
- `1`: overview camera
- `2`: first-person RGB camera

## Useful Flags

- `--floorplan-xml PATH`: load a different room XML
- `--start-x`, `--start-y`, `--start-z`: start position in meters
- `--start-yaw-deg`: initial robot yaw
- `--move-speed`: translation speed in m/s
- `--turn-speed-deg`: turn speed in deg/s
- `--fovy`: camera field of view in degrees
- `--camera-width`, `--camera-height`: render resolution
- `--depth-max-m`: max processed depth
- `--yolo-model`: YOLO weights file or model name, default `yolov8n.pt`
- `--target-labels`: labels to keep from detections
- `--confidence-threshold`: minimum detection confidence

## Default Detection Labels

The automatic controller defaults to:

- `person`
- `dining table`
- `microwave`
- `oven`
- `toaster`
- `sink`

You can override these with `--target-labels`.

## Behavior Summary

Manual mode:

- Drives the robot with the keyboard

Prompt mode:

- Rotates in place to build a scene graph from detections
- Updates a coarse 2D occupancy map from depth
- If no target label is found during the first scan, moves forward and scans again
- Lets you command the robot to walk toward the best previously observed view of the target
