# Loco-Agent0 — Sim Acceptance Runbook

This document tells you how to run each sim-acceptance phase that the
[plan](../PLAN/loco_agent0_plan.md) prescribes (LA-7 through LA-10).
The unit-test suite covers the pure-Python contract; sim acceptance
covers the path that actually runs in Isaac Sim with a live LLM (or
human stand-in) driving the robot.

> All commands assume you are in `isaac-sim-rl-bringup/` and that
> Isaac Lab's `./isaaclab.sh -p` is on your `PATH` (this matches the
> Phase 1-4 demo invocation in `scripts/alex_room_explore/README.md`).

## Quick reference

| Phase | Goal                                      | Client     | Estimated wall-clock |
|------:|-------------------------------------------|------------|----------------------|
| LA-7  | single target + active perception (peek)  | `stdin`    | 5 min, interactive   |
| LA-8  | systematic exploration, no USD prior      | `anthropic`| 3-5 min per trial    |
| LA-9  | dynamic obstacle (B-key blocker)          | `stdin`    | 3 min, scripted      |
| LA-10 | multi-modal (survey + describe_view)      | `anthropic`| 4-6 min per trial    |

## Preflight (once)

```bash
# 1. Confirm the unit-test suite is green before any sim run.
python -m pytest tests/autonomy/ tests/loco_x/ -q
# Expected: 341 passed in ~6 s.

# 2. Confirm the Loco-X package imports standalone.
python -c "from loco_x.agent import AsyncRunner, RunnerConfig, TaskDispatcher; print('ok')"

# 3. For LA-8 / LA-10 (anthropic client) export your API key:
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## LA-7 — Single target with active perception (stdin)

### Goal

Verify the end-to-end loop with a human stand-in for the LLM:

1. agent observation reflects the live SAM3 scene graph,
2. a `peek('left')` skill call rotates the head and triggers SAM3
   re-detection,
3. once the goal appears in the scene graph, `goto('stove')` drives
   the robot to it via the FSM,
4. `finish()` cleanly unwinds.

### Command

```bash
./isaaclab.sh -p scripts/alex_room_explore/alex_onnx_walking_policy.py \
    --enable_cameras \
    scene=room \
    autonomy=approach \
    autonomy.target=stove \
    autonomy.min_observations=1 \
    detector=sam3 \
    rerun=full \
    loco_x=stdin
```

### Recommended: piped stdin script (not interactive)

Isaac Sim takes over the terminal and viewer; typing into the
launching shell is fragile. Use a pre-canned script file and pass
it via ``loco_x.stdin_path``:

```bash
./isaaclab.sh -p .../alex_onnx_walking_policy.py \
    --enable_cameras scene=room autonomy=approach \
    autonomy.target=stove detector=sam3 rerun=full \
    loco_x=stdin \
    loco_x.stdin_path=$PWD/loco_x/runbooks/la7_stove.stdin
```

The shipped ``loco_x/runbooks/la7_stove.stdin`` contains three
turns (peek-if-needed, goto, finish) bounded by ``EOF`` lines —
the same protocol the interactive mode uses. Edit the file to
change behaviour.

### Interactive mode (rare)

If you really want to type at the terminal, launch the script,
wait for the ``[loco_x] agent enabled`` banner, then in **the
same terminal where you launched Isaac** type:

```
```python
goto('stove')
```
EOF
```

Isaac's viewer needs focus for keyboard hotkeys (F-key fall,
etc.) but the *shell* is where stdin lives. If you can't get input
to register, use the piped form above.

### Pass criteria (from plan §LA-7)

- 3/3 trials reach `[autonomy] ARRIVED` with the scripted peek+goto.
- Console shows distinct `[loco_x] turn N` lines and skill timings
  in the end-of-run report.
- Robot finishes at `_cmd = (0, 0, 0, 1)`.

### Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Run ends after 2-3 s with `(no timing data)` | Sim closed before the autonomy loop ran | Re-run *without* `loco_x=stdin` first to confirm the base demo works. If it still exits fast, check Isaac's GPU memory warnings — your `peak GPU allocated: 3301 MB` looks healthy, but the budget-manager warning in the log suggests the carb runtime gave up. |
| Agent never asks for a turn (no `[loco_x] turn 1`) | FSM mode mismatch (Phase 1-4's mode is a string, not an enum) | Fixed in `_step_autonomy` — reads `str(getattr(bundle["fsm"], "mode", "search"))` and maps to "IDLE" for search/idle/arrived. |
| `unknown call: peek` | Sandbox couldn't find the skill | Check `import loco_x.skills` succeeds standalone |
| Robot stands still, no walking | `goto` skill ran but the FSM didn't pick up the new goal | Dispatcher writes `bundle["goal"].set_fixed(...)` via `_seed_phase1_4_goal` — verify `bundle["goal"]` is a `GoalState` instance, not None. |
| Empty scene_graph in observation | SAM3 hasn't fired yet, or scene-graph bridge missing | The autonomy loop's LA-6 hook rebuilds `bundle["scene_nodes"]` from `bundle["scene_graph"].objects` each tick. If your observation always shows `(empty)`, SAM3 simply hasn't grounded anything yet — peek/survey to extend coverage. |
| Stuck on first peek | Head joint not being commanded | `head_yaw_request` is set on the bundle but the autonomy loop doesn't yet read it. Workaround: rely on natural pose drift + chest-cam coverage, or pre-position the head with `face(yaw)` before the run. |

---

## LA-8 — Systematic exploration, heightmap-only (anthropic)

### Goal

The **no-USD** acceptance test. The scene starts with no occupancy
prior; the height map fills in from the chest-cam stream during the
run. The LLM agent must drive frontier exploration until SAM3
detects the goal.

### Command

```bash
export ANTHROPIC_API_KEY=sk-ant-...

./isaaclab.sh -p scripts/alex_room_explore/alex_onnx_walking_policy.py \
    --enable_cameras \
    scene=room \
    autonomy=approach \
    autonomy.target=microwave \
    autonomy.min_observations=1 \
    detector=sam3 \
    rerun=full \
    loco_x=anthropic
```

### Expected sequence

```
[autonomy] occupancy: heightmap provider initialised (empty grid)
[loco_x] agent enabled  tick_hz=2.0 max_turns=20
[loco_x] turn 1: observation 932 chars (~233 tokens)
[loco_x] LLM → ```python
[loco_x]            peek('left')
[loco_x]         ```
[autonomy] head yaw → 'left'
... height map fills in ...
[loco_x] turn 2: scene_graph now has 2 nodes (stove, microwave)
[loco_x] LLM → ```python
[loco_x]            goto('microwave')
[loco_x]         ```
[autonomy] planner: 4 waypoints, length 3.21m, inflation 0.45m
... robot walks ...
[autonomy] ARRIVED at dist=0.84m
[loco_x] LLM → FINISH
[loco_x] agent unwound  status=succeeded  reason="reached microwave"
```

### Pass criteria (from plan §LA-8)

**Success path:**
- 2/3 trials succeed in <15 LLM turns.
- Height map at end of run: `visited_fraction >= 0.35`.
- Zero collisions logged (chest-brake fire count = 0).

**Failure path (required, run separately):**
Remove the microwave from the scene USD before launching. Trial
must:
- Print `[agent] FAILED: <reason>` within 20 turns.
- Reason quotes either `visited_fraction >= 0.9` (Case C) or
  `turn budget exhausted` (Case B).
- Robot ends at `_cmd = (0, 0, 0, 1)`.
- Sim does not crash; timing report fires.
- Phase 4 stuck/recovery NOT triggered by post-fail standstill.

### Cost guardrail

Default `max_turns=20` caps the LLM round-trips. Typical run uses
5-15 turns at ~1-4 k input tokens each on Opus → roughly
$0.05-$0.15 per trial. Stop early with Ctrl+C — the daemon-thread
runner unwinds cleanly.

---

## LA-9 — Dynamic obstacle (stdin)

### Goal

Verify the LA-0b.2 path-invalidation watchdog plus strategic-detour
recovery when the planner's first route is blocked by a newly-
appeared obstacle.

### Command

```bash
./isaaclab.sh -p scripts/alex_room_explore/alex_onnx_walking_policy.py \
    --enable_cameras \
    scene=room \
    autonomy=approach \
    autonomy.target=stove \
    detector=sam3 \
    rerun=full \
    loco_x=stdin
```

### What you do at the terminal

1. Drive the agent to issue a `goto('stove')` so the planner emits
   its first path. Watch for `[autonomy] planner: N waypoints`.
2. **In Isaac viewport**: ~5 s after the planner fires, use the
   **B-key** hotkey to spawn a blocker prim on one of the upcoming
   waypoints. (See `scripts/alex_room_explore/README.md` § Keyboard
   for the B-key handler — to be added in this phase if missing.)
3. Watch for `[autonomy] path invalidated by new obstacle at
   world=(...); replanning`.

### Expected outcomes

| Configuration | LLM expected to | Pass criterion |
|---|---|---|
| Alternate route exists | `goto('stove')` succeeds via the planner's auto-replan; no agent intervention needed | 3/3 trials ARRIVED |
| No alternate route | `goto` returns `error[blocked]`; next agent turn issues `next_frontier()` or `peek` and finds a new path | 2/3 trials reach stove via detour |

### Pass criteria (from plan §LA-9)

- 3/3 alternate-route trials: ARRIVED without manual intervention.
- 2/3 no-alternate trials: agent issues a strategic detour and
  reaches goal.

---

## LA-10 — Multi-modal perception (anthropic)

### Goal

Ablate the D12 multi-modal skills against the geometric baseline.
Three scenario classes; 10 trials total.

### Setup

Same as LA-8 (anthropic client, API key required) but with the
multimodal path enabled. Per D12, `describe_view()` will call the
VLM (default model `claude-sonnet-4-6`) to caption the head-cam frame.

### Scenarios

#### Wide-room (4 trials)

Alex spawns in a large kitchen facing one wall; microwave is at the
periphery of FOV.

```bash
./isaaclab.sh -p scripts/alex_room_explore/alex_onnx_walking_policy.py \
    --enable_cameras scene=room autonomy=approach \
    autonomy.target=microwave detector=sam3 rerun=full \
    loco_x=anthropic
```

Two trials with `survey()` allowed, two restricted to `survey(quick=True)`
via prompt injection. **Pass:** 4/4 ARRIVED, mean turns ≥ 25% lower
than the LA-8 peek-only baseline.

#### Ambiguous-label (3 trials)

`autonomy.target=pot`. SAM3's default vocabulary doesn't ground
"pot" well; agent expected to call `describe_view()` → VLM responds
with something like "a saucepan on the back burner of the stove" →
agent re-issues `find("saucepan")` or `goto("stove")`.

**Pass:** 2/3 ARRIVED. The 1 failure must terminate via `fail()`
(clean), not by turn-budget exhaustion.

#### Wrong-room (3 trials)

Spawn in a non-kitchen room (dining); `autonomy.target=stove`.
Agent must call `describe_view()` to recognise "this isn't the
kitchen" and use `next_frontier()` toward an exit.

**Pass:** 2/3 reach the kitchen and then succeed; clean-fail rule
for the third trial.

### Decision rule (from plan §D12)

If the wide-room scenario doesn't show ≥ 25% turn reduction vs the
LA-8 peek-only baseline on at least 2/4 trials, **D12 stays in
the code but is gated off by default**. Don't ship dead complexity.

### Cost

VLM calls add ~$0.01-0.03 each; cap is `vlm_max_calls_per_task=5`.
Median trial: 2 VLM calls, ~$0.05 extra over the LA-8 baseline cost.

---

## End-of-run artifacts

Every sim-acceptance run produces:

- **stdout log** — pipe through `tee logs/la7_trial1.log` for
  postmortem.
- **rerun.io stream** (`rerun=full`) — pose, scene graph, height
  map, planner path, agent observation. View with `rerun` (or open
  the saved `.rrd` file).
- **end-of-run timing report** — printed at script exit via the
  Phase 4 `Timings` accumulator. Shows per-skill wall-clock budget
  (`agent.tick`, `agent.skill.goto`, ...). Anything > 50% of the
  total flags an optimisation target.

## When something goes wrong

1. **Sim runs but agent is silent.** Check `cfg.loco_x.enabled` is
   `true` in the merged config (`hydra.run.dir`). The default group
   is `disabled`; passing `loco_x=stdin` switches it.
2. **Agent emits code but sandbox rejects.** Read the `last_action`
   line in the next observation — `sandbox_rejected` says exactly
   which AST violation (import / dunder / unknown call). Adjust the
   prompt or skill API; do NOT loosen the sandbox.
3. **LLM hangs > exec_timeout_s.** The runner catches `SandboxTimeout`
   and feeds it back. Check the model isn't generating an absurdly
   long block — the system prompt explicitly bounds responses.
4. **Sim crashes.** Phase 4's emergency brake should never reach
   crash state; if it does, look for a `path_invalidated` storm in
   the log (D10 watchdog firing repeatedly). The fix is in
   `loco_x/occupancy/heightmap_provider.py` — the consistency-gate
   thresholds may need re-tuning for the specific scene.
