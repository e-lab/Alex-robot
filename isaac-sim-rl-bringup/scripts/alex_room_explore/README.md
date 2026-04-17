# alex_room_explore

Run the Alex walking ONNX policy in Isaac Sim. Full articulation loaded from
URDF; the script builds an 80-dim observation each tick, runs the ONNX policy,
applies PD joint targets at 50 Hz. Scene options are a flat ground plane, the
ithor FloorPlan1 kitchen, or the multi-room hallway.

Configured with **Hydra** (shares the same `configs/` tree as
[`cam_room_explore`](../cam_room_explore/README.md); the root config is
`configs/alex_room_explore.yaml`). Isaac AppLauncher flags
(`--enable_cameras`, `--headless`, `--device`) still use the `--flag` form;
everything else is a Hydra override (`key=value`).

## Run

```bash
cd ~/pathtoFolder/IsaacLab

# groundplane + default walking cmd (vx=0.3 m/s)
./isaaclab.sh -p \
  /pathtoFolder/Alex-robot/isaac-sim-rl-bringup/scripts/alex_room_explore/alex_onnx_walking_policy.py

# ithor FloorPlan1 kitchen
./isaaclab.sh -p .../alex_onnx_walking_policy.py scene=room

# multi-room hallway, doors open
./isaaclab.sh -p .../alex_onnx_walking_policy.py scene=hallway

# hallway with doors initially closed
./isaaclab.sh -p .../alex_onnx_walking_policy.py scene=hallway scene.doors=closed

# start in standing mode instead of walking
./isaaclab.sh -p .../alex_onnx_walking_policy.py policy.standing=true

# tweak initial velocity command
./isaaclab.sh -p .../alex_onnx_walking_policy.py policy.vx=0.5 policy.yaw=0.3
```

**Head camera + Rerun + SAM3** (same groups as `cam_room_explore`):
```bash
./isaaclab.sh -p .../alex_onnx_walking_policy.py --enable_cameras \
    scene=room rerun=full detector=sam3
```

**`--scene room` prerequisite** — download the USD once:
```bash
cd /path/to/molmospaces/molmo_spaces_isaac
pip install -e .[dev,sim]
pip install -e /path/to/molmospaces/
ms-download --type usd --install-dir /path/to/Alex-robot/assets/usd --scenes ithor
```

## Keyboard

| Key                      | Action                              |
|--------------------------|-------------------------------------|
| ↑ / Numpad 8             | walk forward                        |
| ↓ / Numpad 2             | walk backward                       |
| ← / Z                    | turn left (yaw +)                   |
| → / X                    | turn right (yaw −)                  |
| Q / Numpad 4             | strafe left                         |
| E / Numpad 6             | strafe right                        |
| L                        | stop — zero velocity                |
| S                        | toggle standing mode                |

## Config groups

```
configs/
├── alex_room_explore.yaml     ← root composition (this script)
├── scene/                     ← shared with cam_room_explore
│   ├── groundplane.yaml       (default for alex_room_explore)
│   ├── room.yaml
│   └── hallway.yaml
├── policy/
│   └── flatfeet.yaml          ← 2026-03-17 ONNX + initial vx/vy/yaw
├── detector/                  ← shared SAM3 group
│   ├── disabled.yaml          (default)
│   └── sam3.yaml
├── yolo/
│   ├── disabled.yaml          (default)
│   └── ihmc.yaml              ← IHMC 12-class custom ONNX
└── rerun/                     ← shared
    ├── disabled.yaml          (default)
    └── full.yaml              ← RGB + depth + SAM3 + pointcloud
```

**Common overrides:**
```bash
scene=hallway scene.doors=closed     # pick scene + door state
policy.standing=true                  # start standing, not walking
policy.vx=0.5 policy.yaw=0.3          # initial velocity command
detector=sam3                         # SAM3 open-vocab on head cam
detector.prompts="door, chair, oven"  # SAM3 prompt list
yolo=ihmc                             # IHMC custom YOLO instead of SAM3
rerun=full rerun.pointcloud=false     # rerun on, point cloud off
```

## What the script does

| Component | Detail |
|-----------|--------|
| Policy    | `2026-03-17_23-20-27_flatfeet` — 80-dim obs → 23-dim action |
| Control   | PD position targets at 50 Hz (4 × 5 ms physics substeps)    |
| Keyboard  | `isaaclab.devices.Se2Keyboard` → updates velocity cmd each tick |
| Command   | `[vx, vy, yaw, standing_flag]` — initial value from `policy.*`, live keyboard updates |
| Scene     | `scene=groundplane | room | hallway` (full collision physics in room/hallway) |

Per-tick terminal line prints: mode, commanded velocity, root position,
hip/knee angles, contact forces, action magnitude.

## Model file

The ONNX policy is not in git. Copy `policy.onnx` from the lab machine into
`isaac-sim-rl-bringup/models/2026-03-17_23-20-27_flatfeet/`. Override
`policy.onnx_path=...` if it lives elsewhere.

## Roadmap

- [x] Hydra config system (shared with `cam_room_explore`)
- [x] Hallway scene support
- [ ] Port `cam_room_explore`'s search/approach FSM (replace teleport
      actuation with `_cmd` velocity-command writes so the walking robot
      actually walks to targets)
- [ ] Fall detector + recovery state
- [ ] Forward-depth obstacle check
