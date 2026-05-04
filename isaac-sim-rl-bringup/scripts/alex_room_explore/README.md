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

**Phase 1 autonomy — fixed-XYZ goal on a flat plane** (no perception):
```bash
# Walk to (3, 0, 0) on the ground plane. Acceptance test for Phase 1 of
# PLAN/autonomous_navigation_plan.md.
./isaaclab.sh -p .../alex_onnx_walking_policy.py \
    scene=groundplane autonomy=fixed_xyz

# Override goal:
./isaaclab.sh -p .../alex_onnx_walking_policy.py \
    scene=groundplane autonomy=fixed_xyz 'autonomy.fixed_xyz=[2.0,1.0,0.0]'

# Tighter stop, slower walk:
./isaaclab.sh -p .../alex_onnx_walking_policy.py \
    scene=groundplane autonomy=fixed_xyz \
    autonomy.stop_dist=0.5 autonomy.walk_speed=0.20
```

In autonomy mode, any keyboard press pauses the FSM for ~1 s — useful for
nudging the robot or interrupting before the goal.

**Phase 2 autonomy — SAM3-detected goal in a real scene:**
```bash
# Walk to the oven in the FloorPlan1 kitchen.
./isaaclab.sh -p .../alex_onnx_walking_policy.py --enable_cameras \
    scene=room autonomy=approach autonomy.target=oven \
    detector=sam3 rerun=full

# Looser lock (faster acquisition, more false positives):
./isaaclab.sh -p .../alex_onnx_walking_policy.py --enable_cameras \
    scene=room autonomy=approach autonomy.target=oven \
    detector=sam3 rerun=full \
    autonomy.lock_conf=0.5 autonomy.min_observations=2
```

The FSM starts in SEARCH (rotates yaw at `autonomy.search_yaw` rad/s). On
each camera tick, SAM3 segmentations are projected to world XYZ via the
vendored scene-graph package and accumulated into a per-run `SceneGraph`.
Once one ObjectNode matching `autonomy.target` reaches confidence ≥
`autonomy.lock_conf` *and* has been observed ≥ `autonomy.min_observations`
times, the goal latches and the FSM transitions to APPROACH. The scene
graph is saved to `output.scene_graph_path` (default
`isaac-sim-rl-bringup/scene_graph.json`) on clean exit and on Ctrl+C.

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
├── rerun/                     ← shared
│   ├── disabled.yaml          (default)
│   └── full.yaml              ← RGB + depth + SAM3 + pointcloud
└── autonomy/                  ← FSM controller (this script)
    ├── manual.yaml            (default — keyboard only)
    ├── fixed_xyz.yaml         ← Phase 1: walk to a hardcoded XYZ
    └── approach.yaml          ← Phase 2: SAM3-detected goal
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

## Autonomy module (`autonomy/` sibling package)

Pure-logic Python (plus one Isaac adapter for the head camera) so most of
the controller stays unit-testable without the simulator:

```
scripts/alex_room_explore/autonomy/
├── pose.py           yaw_from_quat(), FallMonitor (height + tilt)
├── translator.py     fsm_mode_to_cmd() + GaitLimits (vx≤0.4, vy≤0.3, yaw≤0.4)
├── goal.py           GoalState — set_fixed (Phase 1) + update_from_object (Phase 2)
├── fsm.py            FSMController: search → approach → arrived → fallen
├── target_picker.py  pick_goal_for_target — graph → ObjectNode by label + lock
├── obstacle.py       forward_cone_distance (emergency-brake only — see docs/phase3_retrospective.md)
├── perception.py     get_head_cam_pose_K, read_rgb_depth   (Isaac-coupled)
└── __init__.py
```

Phase-2 perception substrate (SAM3 → mask → unproject → dedup) is provided
by the **vendored** `scene_graph/` package — see
`isaac-sim-rl-bringup/scene_graph/VENDORED.md`.

The main script's autonomy hooks:
- `_build_autonomy_bundle()` constructs FSM + GoalState + FallMonitor and
  (in `approach` mode) a SceneGraph + target metadata.
- `_step_autonomy(bundle, robot)` runs every policy tick (50 Hz), writes
  `_cmd` in place from the FSM.
- `_step_perception(bundle, head_cam, tick)` runs every camera tick
  (~12.5 Hz), invokes `process_one_frame` from the vendored package, then
  picks the highest-confidence matching ObjectNode and updates the goal.
  Also caches the **forward-cone obstacle distance** on the bundle so
  `_step_autonomy` can fire the emergency brake (zero `_cmd` if a close
  obstacle pops up). Steering around obstacles is the planner's job
  (Phase 3.5), not the cone's.

If `cfg.autonomy.mode == "manual"` the bundle is `None`, both hooks are
no-ops, and the keyboard path is unchanged.

### Tests

```bash
cd isaac-sim-rl-bringup
python -m pytest tests/autonomy/ -v   # autonomy package (Phase 1 + 2)
python -m pytest tests/unit/ -v       # vendored scene_graph package
python -m pytest tests/ -q            # both, ~214 tests
```

Autonomy package: 100 % coverage on every pure-logic module
(pose / translator / goal / fsm / target_picker). The Isaac adapter
(`perception.py`) is intentionally not unit-tested — it's exercised end-
to-end by the sim acceptance trial.

## Roadmap

- [x] Hydra config system (shared with `cam_room_explore`)
- [x] Hallway scene support
- [x] **Phase 1**: FSM (search/approach/arrived/fallen), `_cmd` translator
      with gait-limit clamping, fall-detection stub, `autonomy=fixed_xyz`
      preset, 56 unit tests at 100 % coverage
- [x] **Phase 2**: SAM3 → goal XYZ via the vendored `scene_graph/` package.
      `autonomy=approach autonomy.target=<label>` walks to the highest-
      confidence ObjectNode of that label. Goal latches at
      `score >= lock_conf` after `>= min_observations` sightings. Scene
      graph saved to `output.scene_graph_path` on exit.
- [x] ~~**Phase 3**: reactive forward-cone obstacle avoidance.~~
      **Deprecated 2026-05-04** after failing acceptance against wall-
      shaped obstacles (kitchen counter). Three variants were tried (head-
      cam cone, chest-cam cone, three-zone strafe). Each broke for the
      same fundamental reason: a local sensor cone has no concept of
      "go around the end of the wall." See
      [`docs/phase3_retrospective.md`](../../docs/phase3_retrospective.md)
      for the full story. Salvaged: chest cam stays as an
      **emergency-stop** only (no steering), `forward_cone_distance` stays
      as the brake's distance estimator.
- [ ] **Phase 3.5**: USD-derived 2D occupancy + A* planner + waypoint
      follower. The room USD is the source of truth for static geometry;
      we rasterise it once, plan with A*, and follow the waypoints with
      the existing FSM heading controller. SAM3 stays as the goal-lookup
      mechanism. See `PLAN/autonomous_navigation_plan.md` § Phase 3.5 for
      the design + acceptance criteria.
- [ ] **Phase 4**: full fall-recovery sequence (extend the Phase-1 stub)
- [ ] **Phase 5** (optional): LLM task agent for multi-target chains

### Phase 1 acceptance criterion

(from `PLAN/autonomous_navigation_plan.md`)

> Scene: `groundplane`, no obstacles. Goal: `(3, 0, 0)`. Pass: 4/5 trials reach
> within `stop_dist = 1.0 m` of the goal, no falls, no `vx`/`yaw_rate` clamps
> hit.

```bash
./isaaclab.sh -p .../alex_onnx_walking_policy.py \
    scene=groundplane autonomy=fixed_xyz
```

### Phase 2 acceptance criterion

(from `PLAN/autonomous_navigation_plan.md`)

> Scene: `room` (FloorPlan1 kitchen). Pass: 3/5 trials detect the oven
> within 30 s of scanning, lock the goal, walk to within 1.0 m, no falls.
> Scene-graph JSON contains the oven with reasonable XYZ.

```bash
./isaaclab.sh -p .../alex_onnx_walking_policy.py --enable_cameras \
    scene=room autonomy=approach autonomy.target=oven \
    detector=sam3 rerun=full
```

### Phase 3 (deprecated)

The original reactive plan never met its 3/5 acceptance bar. See
[`docs/phase3_retrospective.md`](../../docs/phase3_retrospective.md) for
the full failure analysis. The replacement is **Phase 3.5** (USD planner,
spec'd in `PLAN/autonomous_navigation_plan.md`). The current code keeps
the chest cam as an emergency brake only — no steering from depth.
