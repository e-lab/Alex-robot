# cam_room_explore_isaac

Isaac Sim + Rerun equivalent of `demos/cam_room_explore`. A free-flying camera
in the ithor FloorPlan1 kitchen that you drive with the keyboard, streaming
RGB, depth, SAM3 open-vocabulary segmentation, and an optional world-frame
point cloud to a Rerun viewer.

No robot, no walking policy, no joints — just a camera floating in the room.
This is the minimal harness for prototyping perception and autonomous
search/approach logic before porting it onto the Alex ONNX walking policy
(`alex_onnx_walking_policy.py`).

## Run

Configured with **Hydra** (`configs/cam_room_explore.yaml` is the root; see
"Config layout" below). Isaac AppLauncher flags (`--enable_cameras`,
`--headless`, `--device`) still use the `--flag` form; everything else is a
Hydra override (`key=value`).

**Manual teleop in the ithor kitchen (defaults):**
```bash
cd ~/alex/repository-group/IsaacLab
./isaaclab.sh -p /home/sravani/E-Lab/Spring2026/repos/Alex-robot/isaac-sim-rl-bringup/scripts/cam_room_explore/cam_room_explore_isaac.py \
    --enable_cameras
```

**Manual in the hallway, streaming SAM3 + point cloud to Rerun:**
```bash
./isaaclab.sh -p .../cam_room_explore_isaac.py --enable_cameras \
    scene=hallway
```

**Autonomous search + approach:**
```bash
./isaaclab.sh -p .../cam_room_explore_isaac.py --enable_cameras \
    scene=hallway autonomy=approach autonomy.target=sofa
```

**Override any leaf value:**
```bash
# faster keyboard, higher SAM3 threshold, custom spawn
./isaaclab.sh -p .../cam_room_explore_isaac.py --enable_cameras \
    scene=hallway motion.move_speed=1.2 detector.conf=0.5 \
    spawn.x=2.0 spawn.yaw_deg=90
```

## Keyboard

| Key            | Action                                           |
|----------------|--------------------------------------------------|
| ↑ / ↓          | translate forward / back (camera-local)          |
| ← / →          | yaw left / right                                 |
| Q / E          | strafe left / right (camera-local)               |
| R / F          | rise / lower (world Z)                           |
| L              | reset pose to spawn                              |

## Config layout

```
isaac-sim-rl-bringup/configs/
├── cam_room_explore.yaml     ← root; composes all groups below
├── schema.py                 ← dataclass schemas (type-checked)
├── scene/
│   ├── room.yaml             ← ithor FloorPlan1
│   ├── hallway.yaml          ← multi-room hallway with doors
│   └── groundplane.yaml
├── detector/
│   ├── disabled.yaml
│   └── sam3.yaml             ← SAM3 prompts + conf threshold
├── rerun/
│   ├── disabled.yaml
│   └── full.yaml             ← RGB + depth + SAM3 + pointcloud
└── autonomy/
    ├── manual.yaml           ← keyboard only (default)
    └── approach.yaml         ← rotate/scan/walk-to-target FSM
```

**Selecting a group preset** (yaml file name):
```bash
scene=hallway        # loads configs/scene/hallway.yaml
autonomy=approach    # loads configs/autonomy/approach.yaml
detector=disabled    # loads configs/detector/disabled.yaml
```

**Overriding a leaf value** (`group.key=value`):
```bash
scene.doors=closed
detector.conf=0.5
detector.prompts="oven, toaster"
autonomy.target=oven
autonomy.walk_speed=0.6
spawn.x=2.0 spawn.y=-1.0
motion.move_speed=1.2
rerun.pointcloud_stride=4
output.save_every_s=10
```

**See the full composed config** (Hydra prints it at startup) or run:
```bash
./isaaclab.sh -p .../cam_room_explore_isaac.py --enable_cameras --help=hydra
```

## How it works

### Scene layout

Isaac USD stage:

```
/World
├── Light                  dome light
├── Room                   FloorPlan1_physics (walls, floor, objects)
└── Robot                  ← Xform (empty transform node, not a robot)
    └── Camera             ← RTX sensor camera, child of Robot
```

`/World/Robot` is a plain **Xform prim** — a coordinate frame you can translate
and rotate. It has no mesh, no physics, no articulation. The sensor Camera
lives underneath it, so moving Robot moves Camera too.

### Main loop (per tick)

```
1. Read keyboard               _apply_keyboard(...)
2. Update robot pose state     CameraRobot.x/y/z/yaw updated in memory
3. Write pose to USD stage     robot.write_to_stage("/World/Robot")
4. Snap Isaac viewport         sim.set_camera_view(behind-the-robot)
5. Step sim (render)           sim.step(render=True)
6. Update sensor camera        cam.update(SIM_DT)   every 4th tick (~25 Hz)
7. Log to Rerun                _rerun_log(...)
```

Step 3 is the key move: `write_to_stage` pokes the `/World/Robot` Xform's
translate and orient ops. On the next `sim.step`, every child (including
Camera) follows. The RTX sensor renderer produces RGB + depth buffers from
wherever the camera ended up.

### `CameraRobot` class

Tiny state holder — `(x, y, z, yaw)` — with three methods:

- `translate_local(dx_forward, dy_left, dz_world)` — moves along camera-local
  axes so "forward" always means the direction the camera is facing
- `rotate_yaw(dyaw)` — spins about world Z
- `write_to_stage(path)` — updates the translate + orient xform ops on the prim

No physics. Walking into a wall clips through it.

### Keyboard handling

Standalone `KeyboardState` class subscribes to carb's raw keyboard events.
Every tick, `_apply_keyboard` reads which keys are currently held and
converts them to camera deltas. Not using `Se2Keyboard` — that's tied to
articulations.

### What Rerun shows

| Entity path                          | What it is                          |
|--------------------------------------|-------------------------------------|
| `world`                              | root frame, Z-up                    |
| `world/robot` Transform3D + Arrows3D | robot pose + RGB axis triad         |
| `world/cam/points` (optional)        | coloured point cloud from depth     |
| `world/goal` (autonomous)            | green sphere at target 3D position  |
| `controller/state` (autonomous)      | 0=search 1=approach 2=arrived       |
| `controller/forward_dist`            | distance to target (m)              |
| `controller/heading_error_deg`       | yaw error to target (deg)           |
| `world/scene/<label_N>`              | persistent yellow spheres — scene graph objects |
| `camera/rgb`                         | RGB sensor image                    |
| `camera/depth`                       | depth sensor image                  |
| `camera/rgb/sam3_mask`               | SAM3 open-vocab segmentation        |
| `camera/rgb/sam3`                    | SAM3 bounding boxes + labels        |
| `robot/x, /y, /z, /yaw`              | scalar plots of pose over time      |

## Why it's simpler than `alex_onnx_walking_policy.py`

The Alex script has to: load a URDF, build an Articulation, load an ONNX
policy, build an 80-dim observation, run the policy, apply PD control, step
physics, handle foot contacts, attach the camera to a moving articulation
link, keep SE(2) keyboard commands flowing into the policy.

This script throws all of that away. The "robot" is a freely-teleporting
Xform. The camera is its child. Keyboard writes the xform directly.

## What it's good for

- Quickly testing SAM3 prompts against the scene without fighting the walking
  policy
- Building autonomous **search / approach** logic on a predictable actuator —
  write `(x, y, z, yaw)` directly, no drift, no feet
- Later: port that logic onto the ONNX policy by replacing `write_to_stage`
  with `_cmd = [vx, vy, yaw_rate]`

## Autonomous mode (`--prompt`)

Three-state FSM runs once per tick when `--prompt LABEL` is set:

| State       | Trigger                                        | Actuation                                      |
|-------------|------------------------------------------------|------------------------------------------------|
| `search`    | no fresh detection within `--prompt-stale-s`   | yaw continuously; pitch oscillates as a sine wave in ±`--prompt-tilt-deg` with period `--prompt-tilt-period-s` |
| `approach`  | goal known, distance > `--prompt-stop-dist`    | level pitch, P-control yaw toward goal + walk forward |
| `arrived`   | distance ≤ `--prompt-stop-dist`                | freeze (prints once)                           |

**Tilt sweep:** every yaw angle gets scanned at multiple elevations because
yaw and pitch vary simultaneously — no pause cycle. The camera traces a
sinusoidal path across the scene, so objects on shelves and on the floor pass
through view naturally without a dedicated tilt phase.

How the goal is extracted: each SAM3 tick, the highest-score mask whose prompt
equals `--prompt` has its centroid pixel looked up in the depth buffer, then
unprojected to world XYZ using the camera intrinsics + prim pose.

**Goal locking:** the first detection whose SAM3 score ≥ `--prompt-lock-conf`
latches the world XYZ. After locking, the robot walks to that frozen position
regardless of whether SAM3 continues to detect the target — prevents mid-
approach oscillation when the object fills the view and mask quality drops.
Press `L` to unlock and re-search. Set `--prompt-lock-conf -1` to disable
locking (goal updates every frame, as before).

If no detection ever reaches lock confidence, the goal still updates on every
frame above `--sam3-conf`, and goes stale after `--prompt-stale-s` seconds.

Keyboard override: any key press pauses autonomy for 1 s, letting you take
manual control without fighting the controller. `L` resets pose.

**Arrival → next target.** When the robot arrives, the terminal prompts:

```
[autonomous] Arrived at 'oven'. Next target? (blank / 'quit' to stop autonomy):
```

Type a new concept (e.g. `chair`) and press Enter — the goal unlocks, the
target switches, and the robot starts searching from its current position.
Leaving blank or typing `quit` ends autonomy; keyboard takes over. The input
runs in a background thread so the simulation never freezes waiting for you.

## Scene graph

Every SAM3 detection with a valid depth-projected world position is accumulated
into a persistent `SceneGraph` object. Detections within 0.5 m of an existing
entry with the same label are merged (running average of position, keep best
score). New detections beyond that radius create a new entry.

The scene graph is:
- **Logged to Rerun** as persistent yellow spheres at `world/scene/<label_N>`
- **Printed** in the per-second terminal summary (e.g. `scene: 5 objects: 1× chair, 2× door, 1× fridge, 1× oven`)
- **Saved to `scene_graph.json`** on exit (in `isaac-sim-rl-bringup/`)

Example JSON output:
```json
{
  "room_id": "FloorPlan1",
  "scan_complete": true,
  "n_objects": 5,
  "object_labels": ["chair", "door", "fridge", "oven"],
  "objects": {
    "oven_1": {
      "label": "oven",
      "position_xyz": [-0.32, -2.05, 0.78],
      "confidence": 0.73,
      "first_seen_tick": 588,
      "last_seen_tick": 812,
      "bbox_area_px": 48200
    }
  },
  "robot_path": [[1.2, -0.8, 0.0], [1.15, -0.82, 201.9]]
}
```

## Roadmap

- [x] `--prompt oven` autonomous mode: rotate, scan, walk toward best SAM3 detection
- [x] Goal marker in Rerun (`world/goal`)
- [x] Scene graph with persistent object memory and JSON export
- [ ] Accumulated point cloud / top-down occupancy map across frames
- [ ] Obstacle avoidance using the accumulated cloud
- [ ] Port approach FSM onto `alex_onnx_walking_policy.py`
- [ ] Micro-actions as tools + VLM planner
