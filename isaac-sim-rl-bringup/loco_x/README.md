# Loco-X — Agentic Locomotion Framework

Loco-X is the agentic layer that sits on top of Phase 1-4's locomotion
primitives. It adapts the CaP-Agent0 pattern (arXiv 2603.22435,
github.com/capgym/cap-x) — "an LLM emits Python that composes
perception and control primitives, with structured multi-turn
feedback" — to Alex's locomotion + scene-graph stack.

See [`PLAN/loco_agent0_plan.md`](../PLAN/loco_agent0_plan.md) for the
full design (13 design decisions, 12 implementation phases).

## Layout

```
loco_x/
├── cli/                    Hydra @main entry points
│   ├── run.py              live runtime (replaces _build_autonomy_bundle
│   │                       glue when the agent is enabled)
│   └── eval.py             trial runner (analog of capx/envs/runner.py)
├── occupancy/              D10: runtime occupancy provider
│   ├── base.py             OccupancyProvider Protocol + CellState
│   ├── usd_provider.py     LA-0a — wraps autonomy.usd_occupancy
│   ├── heightmap_provider.py   LA-0b — online RGBD-built map
│   ├── frontier.py         LA-0c — info-gain + D14.1 semantic anchors
│   └── synthetic.py        test-only point cloud factory
├── perception/             D2: agent-facing observation
│   ├── observation.py
│   └── scene_filter.py     spatial-temporal scene-graph filter
├── skills/                 D4 + D12: the LLM-callable API
│   ├── registry.py
│   ├── locomotion.py       goto / goto_xy / face / stop
│   ├── perception.py       find / peek / survey / list_scene / describe_view
│   ├── exploration.py      next_frontier / visited_fraction
│   └── meta.py             finish / fail
├── llm/                    D6 + D12: text + multimodal client
│   ├── client.py
│   ├── prompts.py          system prompt + decision table
│   └── parsers.py          REGENERATE / FINISH / fenced-block extraction
├── agent/                  D1: sandbox + runner
│   ├── runner.py           the loop (analog of capx/envs/trial.py)
│   └── sandbox.py          AST whitelist + thread-timeout exec
├── conf/                   Hydra config tree (composed at startup)
│   ├── config.yaml         root composition
│   ├── scene/              room | hallway | groundplane
│   ├── occupancy/          usd | heightmap
│   ├── perception/         sam3 | disabled
│   ├── agent/              disabled | stdin | anthropic | openrouter
│   └── rerun/              disabled | full
└── README.md               (this file)
```

## Boundary rule

Loco-X **imports from** but never modifies:

* `scripts/alex_room_explore/autonomy/` — Phase 1-4 primitives
  (planner, FSM, recovery, goal lock, USD occupancy rasteriser).
* `scene_graph/` — SAM3 + scene-graph maintenance.

Phase 1-4 stays frozen as the safety net we benchmark against
(LA-0a's "bit-identical" canary). When you change a primitive, change
it under `autonomy/` and Loco-X picks it up for free.

## Running

For the agent embedded in the existing Phase 1-4 demo (the only path
that actually moves Alex today), pass the new `loco_x` Hydra group:

```bash
# Default: agent off — identical to Phase 1-4.
./isaaclab.sh -p scripts/alex_room_explore/alex_onnx_walking_policy.py \
    --enable_cameras scene=room autonomy=approach \
    detector=sam3 rerun=full

# Loco-X agent on with a human-in-the-loop stdin client:
./isaaclab.sh -p scripts/alex_room_explore/alex_onnx_walking_policy.py \
    --enable_cameras scene=room autonomy=approach \
    autonomy.target=stove detector=sam3 rerun=full \
    loco_x=stdin

# Loco-X agent on with Claude:
export ANTHROPIC_API_KEY=sk-ant-...
./isaaclab.sh -p scripts/alex_room_explore/alex_onnx_walking_policy.py \
    --enable_cameras scene=room autonomy=approach \
    autonomy.target=microwave detector=sam3 rerun=full \
    loco_x=anthropic
```

See [`SIM_ACCEPTANCE.md`](./SIM_ACCEPTANCE.md) for the LA-7 through
LA-10 acceptance scenarios with pass criteria.

## Tests

```bash
python -m pytest tests/loco_x/ -q
```

The full sweep (`tests/autonomy/ tests/loco_x/`) must stay green on
every commit. LA-0a's `test_planner_paths_identical_to_phase_1_4_demo`
is the canary: if it drifts, the safety net broke.

## Phase status

* [x] LA-0a — `OccupancyProvider` interface + USD-provider refactor
* [x] LA-0b.1 — synthetic point cloud harness (10 tests, no Isaac, no GPU)
* [x] LA-0b.2 — head-cam height-map integration (backproject + backend facade)
* [x] LA-0b.3 — chest-cam as primary occupancy stream; head-cam opt-in
      via peek/survey. `step_perception(chest_depth, chest_pose,
      head_depth=, head_pose=, ...)` two-stream signature.
* [x] LA-0c — frontier + info-gain + D14 semantic/variance costs
* [x] LA-1 — skill registry + AST/exec-timeout sandbox (D1, D4)
* [x] LA-2 — observation builder (D2 spatial-temporal filter + D9 dual coords + D13 snapshot)
* [x] LA-3 (folded into LA-0c) — see plan
* [x] LA-4 — LLM client + parsers (Scripted / Stdin / Anthropic; multimodal for D12)
* [x] LA-5 — agent runner (closed loop: observation → LLM → sandbox → skills)
* [x] LA-6 — wire AsyncRunner + TaskDispatcher into `_step_autonomy`
       (opt-in via `loco_x=scripted` Hydra group)
* [x] LA-7 — sim acceptance: single target with active perception.
       Validated across 2 targets × 2 backends in FloorPlan1 kitchen:
         * stove + stdin:      3 turns, dist=0.90m (commit da716ba)
         * stove + anthropic:  3 turns, dist=0.92m, $0.05 (f9b4a49)
         * sink + anthropic:   3 turns, dist=1.00m, $0.05.
           Recovered from a planner NO-PATH via Phase 4's
           stuck-rotation + the agent wake-up mechanism (401bea5).
* [ ] LA-8 through LA-10 — sim acceptance phases
