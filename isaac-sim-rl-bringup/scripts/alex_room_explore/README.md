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
    scene=room autonomy=manual detector=sam3 rerun=full
```

**Full demo — autonomous navigation to a SAM3-detected target:**
```bash
./isaaclab.sh -p .../alex_onnx_walking_policy.py --enable_cameras \
    scene=room autonomy=approach autonomy.target=stove \
    detector=sam3 rerun=full \
    autonomy.min_observations=1
```

This is the "all phases working together" command. It exercises:
- SAM3 perception → `goal LOCKED` on the named target.
- USD-derived occupancy grid + A* planner → `[autonomy] planner: N waypoints`.
- Waypoint follower → robot walks each waypoint, then faces the goal.
- Phase-4 stuck monitor + recovery agent → engages automatically if the
  robot wedges or falls (also triggerable manually with **F**).

Targets that work out-of-the-box with the default SAM3 vocabulary:
`stove`, `sink`, `oven`. Set `autonomy.min_observations=1` for fastest
goal lock; bump to 3 for a stricter (but slower) detection.

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

**Phase 2 autonomy — SAM3-detected goal in a real scene** (perception
only, no planner; useful when running scenes without an occupancy grid):
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
times, the goal latches and the FSM transitions to APPROACH. **For
`scene=room`, an A* path is then planned through the USD-derived
occupancy grid (Phase 3.5) and the waypoint follower drives Alex along
it.** The scene graph is saved to `output.scene_graph_path` (default
`isaac-sim-rl-bringup/scene_graph.json`) on clean exit and on Ctrl+C.

**Phase 4 — recovery testing:** press **F** in the Isaac viewport while
the robot is walking to force a synthetic fall. The recovery agent
engages, holds standing for 3 s, re-checks the pose, and re-plans:

```
[keyboard] F pressed → forcing synthetic fall on next autonomy tick
[autonomy] [FALL] root_z=0.93m  proj_grav_xy=(...) — entering recovery (attempt 1/2)
[autonomy] recovery succeeded (attempt 1/2) — re-planning from current pose
[autonomy] planner: 5 waypoints, length 2.74m, inflation 0.45m
```

The stuck monitor engages automatically if the robot makes < 0.10 m of
progress over 5 s while in APPROACH — it rotates 90° in place
(closed-loop on yaw) and re-plans.

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
| F                        | force a synthetic fall (Phase-4 test) |

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
scene=hallway scene.doors=closed              # pick scene + door state
policy.standing=true                           # start standing, not walking
policy.vx=0.5 policy.yaw=0.3                   # initial velocity command
detector=sam3                                  # SAM3 open-vocab on head cam
detector.prompts="door, chair, oven"           # SAM3 prompt list
yolo=ihmc                                      # IHMC custom YOLO instead of SAM3
rerun=full rerun.pointcloud=false              # rerun on, point cloud off

# Phase-2 perception
autonomy.target=sink                           # any label in detector.prompts
autonomy.lock_conf=0.5                         # detection-score threshold to latch
autonomy.min_observations=1                    # frames seen before locking

# Phase-4 recovery + stuck (defaults shown)
autonomy.recovery_stand_s=3.0                  # standing-flag hold per attempt
autonomy.recovery_max_attempts=2               # 2 attempts → FAILED
autonomy.recovery_rotation_yaw=0.4             # rad/s during 90° rotation
autonomy.stuck_window_s=5.0                    # APPROACH-vx>0 window
autonomy.stuck_dist_m=0.1                      # min displacement over window
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
│                       (with arrival-band hysteresis to prevent boundary flapping)
├── target_picker.py  pick_goal_for_target — graph → ObjectNode by label + lock
├── obstacle.py       forward_cone_distance (emergency-brake only — see docs/phase3_retrospective.md)
├── perception.py     get_cam_pose_K, read_rgb_depth, read_depth (Isaac-coupled)
├── usd_occupancy.py  GridFrame + occupancy_from_usd (Phase 3.5a)
├── planner.py        plan_path — A* + inflation + line-of-sight smoothing (Phase 3.5b)
├── recovery.py       RecoveryAgent + StuckMonitor + YawTracker (Phase 4)
├── timing.py         Timings + format_timing_report + memory helpers
└── __init__.py
```

Phase-2 perception substrate (SAM3 → mask → unproject → dedup) is provided
by the **vendored** `scene_graph/` package — see
`isaac-sim-rl-bringup/scene_graph/VENDORED.md`.

The main script's autonomy hooks:
- `_build_autonomy_bundle()` constructs FSM + GoalState + FallMonitor +
  RecoveryAgent + StuckMonitor + YawTracker, and (in `approach` mode) a
  SceneGraph + target metadata. For `scene=room` it also builds (or
  loads from cache) the USD-derived occupancy grid.
- `_step_autonomy(bundle, robot)` runs every policy tick (50 Hz). At the
  top: recovery state machine (FAILED short-circuit → STANDING done →
  ROTATING done → fresh fall trigger). Then the waypoint follower walks
  Alex along the planner's path; the FSM's heading controller drives
  intermediate waypoints, and the final waypoint goes through `fsm.step`
  so arrival fires on the real target. Stuck monitor + emergency brake
  follow.
- `_step_perception(bundle, head_cam, chest_cam, tick)` runs every
  camera tick (~12.5 Hz), invokes `process_one_frame` from the vendored
  scene-graph package, then picks the highest-confidence matching
  ObjectNode and updates the goal. Caches the SAM3 `RawDetection` list
  on the bundle so `_log_sam3` can render the rerun overlay without
  re-running SAM3, and caches the **forward-cone obstacle distance**
  for the emergency brake.
- `_maybe_plan_path_on_lock(bundle)` runs **once** the moment the goal
  latches: A* from current robot pose to goal XY through the inflated
  occupancy grid. The path + index land on the bundle.
- `_replan_from_current_pose(bundle)` re-runs the planner after a
  recovery event (stuck rotation or fall stand-up).

If `cfg.autonomy.mode == "manual"` the bundle is `None`, the hooks are
no-ops, and the keyboard path is unchanged.

### Tests

```bash
cd isaac-sim-rl-bringup
python -m pytest tests/autonomy/ -q   # autonomy package (Phases 1, 2, 3.5, 4 + timing)
python -m pytest tests/unit/ -v       # vendored scene_graph package
python -m pytest tests/ -q            # both
```

Autonomy package: **159 unit tests** across pose / translator / goal /
fsm / target_picker / obstacle / usd_occupancy / planner / recovery /
timing. Pure-logic only (no Isaac, no torch in the test path) — full
suite runs in well under one second. The Isaac adapter (`perception.py`)
and the runtime integration in `alex_onnx_walking_policy.py` are
intentionally not unit-tested — they're exercised end-to-end by the sim
acceptance trial.

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
- [x] **Phase 3.5**: USD-derived 2D occupancy + A* planner + waypoint
      follower. The room USD is the source of truth for static geometry;
      we rasterise it once at startup (cached as
      `<usd_dir>/room.occupancy.npz`), plan with A* on the inflated
      grid (default `inflation=0.45 m`), and the waypoint follower
      drives Alex through each waypoint. The final waypoint hands off
      to `fsm.step` so arrival fires on the real target; once
      `dist < stop_dist`, the robot rotates in place to face the goal
      before declaring ARRIVED. SAM3 is the goal-lookup mechanism.
- [x] **Phase 4**: fall recovery + stuck detection.
      `RecoveryAgent` (IDLE / STANDING / ROTATING / FAILED) holds the
      standing flag for 3 s on a fall, then re-checks pose and
      re-plans on success (up to 2 attempts). `StuckMonitor` latches
      when APPROACH commands `vx > 0` but the robot moves < 0.10 m
      over 5 s → rotates 90° in place (closed-loop on yaw via
      `YawTracker`) → re-plans from the new pose. F-key triggers a
      synthetic fall for testing. End-of-run timing + memory report
      printed via SIGINT / atexit.
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

### Phase 3.5 acceptance criterion

> Scene: `room` with the robot spawned such that the kitchen island and
> L-shaped counter block the direct line to the goal. Pass: 3/5 trials
> reach the goal via a planned path (visualised as a polyline in
> rerun) and ARRIVE without falling, < 30 s end-to-end.

```bash
./isaaclab.sh -p .../alex_onnx_walking_policy.py --enable_cameras \
    scene=room autonomy=approach autonomy.target=stove \
    detector=sam3 rerun=full \
    autonomy.min_observations=1
```

Validated end-to-end: 5-waypoint, 2.82 m path, single plan call,
robot walks every waypoint, faces the goal, ARRIVES at `dist≈0.84 m`,
~175 s wall-clock.

### Phase 4 acceptance criterion

> Recovery (fall): 2/3 induced falls produce a `recovery succeeded`
> log line and the robot resumes APPROACH all the way to ARRIVED.
> Recovery (stuck): when wedged, the rotate-90°-and-replan loop
> eventually unwedges and the robot reaches the goal.

Trigger a fall with the **F** hotkey while in APPROACH; trigger stuck
by walking the robot into a low-clearance gap (or just let the
emergency brake fire near the counter — the stuck monitor will
engage automatically).
