# Phase 3 Retrospective — Reactive Obstacle Avoidance

**Status:** deprecated 2026-05-04
**Replaced by:** Phase 3.5 (USD-derived occupancy + A* planner). See
`PLAN/autonomous_navigation_plan.md` § Phase 3.5 for the design.

This memo documents what was tried in Phase 3, why each variant failed, and
what was learned. The reactive code itself was deleted — keeping it in tree
risked someone reading it as the "right" approach. The lesson lives here.

## Context

Phase 3 of `autonomous_navigation_plan.md` called for "reactive obstacle
avoidance": a forward-cone depth check on the head camera, with `vx`
suppressed and `yaw_rate` biased toward the goal when the cone reads
< 1.5 m. Acceptance: 3/5 trials reach the goal in `scene=room` (FloorPlan1
kitchen) without colliding.

The plan said to add a chest camera "only if" the head cam blind spot
caused failures. It did. We added one. It didn't fix the fundamental
problem.

## What was tried

### Attempt 1 — head-cam cone

Forward cone on head-cam depth, ±20° h × ±10° v. When dist < stop_dist,
zero `vx` and rotate toward the goal heading.

**Failed because:** the head cam sits at ~1.6 m height looking horizontally.
The kitchen counter top at 0.9 m doesn't enter the ±10° vertical cone until
the robot is roughly 0.5 m away (`tan(10°) × 1.6 m / 0.7 m ≈ 0.4 m radial
band` — math worked out by hand, confirmed in sim). 0.5 m is too close to
brake at walk_speed=0.30 m/s with the gait's rotation latency. Robot walked
into the counter and fell, every trial.

### Attempt 2 — head-cam cone with widened vertical FOV

Same as Attempt 1 but `obstacle_cone_v_deg=25`. Idea: widen the vertical
cone so the counter enters earlier.

**Failed because:** geometry doesn't change at the source — the counter is
still below the camera's down-looking horizon line until close. Wider cone
admits more floor pixels and noise, but doesn't see the counter much
earlier. Brake fired at ~0.46 m, same as Attempt 1.

### Attempt 3 — chest camera (TORSO_LINK, 30° down-pitch)

Added a second camera on `TORSO_LINK` pitched 30° downward, depth-only,
320×240. Switched the obstacle-distance source from head_cam to chest_cam.

**Failed because:** the chest cam *did* solve the geometry — counter was
visible from > 1 m, brake fired at 0.98 m on first detection (vs 0.46 m
before), well above the 1.50 m stop threshold. But the rotate-toward-goal
logic was the bug: rotating toward the goal pointed back at the same
counter (since the goal was *behind* the counter), so the brake fired
again, and again. Robot oscillated in place rotating left-right for ~20
minutes before some lucky configuration let it past. *Reached the goal*,
but not credibly — and only when SAM3's goal pose was unstable enough to
shift the heading.

The chest camera itself was the right idea. The "rotate toward goal"
response was wrong.

### Attempt 4 — three-zone clearance + strafe

Split the cone into left/center/right horizontal thirds
(`forward_cone_clearance`). When center < stop_dist, sidestep toward
whichever side has more clearance: set `vy = ±0.18 m/s`, `vx = 0.09 m/s`,
keep yaw control on the goal.

**Failed because:** the FloorPlan1 counter is *wall-shaped*, not point-
shaped. Strafing along it doesn't end — the counter never has an "end"
within reach of a few seconds of lateral travel. The robot strafes for ~5 s,
heading control pulls it back toward goal heading (which is into the
counter), stress on the gait builds, eventually `proj_grav_y = -0.68`
(45°+ tilt) and the robot falls. Saw two clean falls before stopping.

Hysteresis margins (0.30 m) and commitment timers (0.5 s) reduced direction
flips dramatically but couldn't fix the wall-vs-strafe geometry.

## Root cause

Local reactive avoidance only works for **discrete obstacles** you can
sidestep past in 1–2 m of lateral motion: a chair leg, a single pole, a
trash can. It does not work for **wall-shaped obstacles** (counters, table
edges, walls, hallways) because the robot has no concept of "go around the
end of the wall." The end of the wall is not in the cone's view, and there
is no internal model of where the wall ends — the cone is myopic by
construction.

The original plan's directive — *"No global SLAM — local reactive avoidance
only (Phase 3); revisit if room exploration becomes multi-room"* — was
written before we knew the kitchen layout. It was right in spirit (don't
build a SLAM stack), but wrong in detail (the kitchen has wall-shaped
obstacles and needs a global map).

## What was salvaged

- **Chest camera** stays on `TORSO_LINK` pitched 30° down. Demoted from
  primary obstacle sensor to **emergency stop only** — if center-cone
  clearance falls below 0.5 m, zero `_cmd` and hold. No steering from
  depth.
- **`forward_cone_distance(depth, K, h_deg, v_deg)`** stays in
  `autonomy/obstacle.py` — it's the brake's distance estimator. Pure
  numpy, 100 % unit-test coverage.
- **All Phase 3 unit tests** (10 of them) stay green.

## What was deleted

- `forward_cone_clearance` (three-zone variant)
- The sidestep block in `_step_autonomy` (hysteresis margin, commitment
  timer, strafe vy/vx)
- `obstacle_clearance` bundle key + `clearance_left/center/right` rerun
  scalars
- `autonomy.obstacle_stop_dist` config field (replaced by hard-coded
  `emergency_dist=0.5` on the bundle)

The deleted code never reached a clean commit on `sravani-develop`. The
git history shows only the working state we kept; the failed approach is
documented here, not preserved.

## Lessons for Phase 3.5

1. **Use the map you have.** The room is a known sim asset; rasterising
   its USD into a 2D occupancy grid takes < 1 s and is ground truth.
   Going to SAM3 + scan + accumulator was the wrong first step.
2. **Separate "what's the goal?" from "how do I get there?".** SAM3 is
   great for the former, terrible for the latter. Keep it in the goal-
   lookup loop, kick it out of the steering loop.
3. **Pick the obstacle topology first.** Reactive works for discrete;
   deliberative works for wall-shaped. The acceptance criterion should
   have specified what kind of scene the test runs in — "FloorPlan1
   kitchen" was too unspecific.
4. **Wall-shaped obstacles need an end-of-wall waypoint.** A planner
   provides this naturally; a cone never can.

## Files involved (for git archaeology)

If you need to see the deprecated code:
- `git log --all --oneline -- isaac-sim-rl-bringup/scripts/alex_room_explore/autonomy/obstacle.py`
  shows the file's history.
- The reactive sidestep block was at `_step_autonomy` in
  `alex_onnx_walking_policy.py`. It was removed in the same commit that
  introduced this memo.
- The chest camera setup was kept; only the steering code was removed.
