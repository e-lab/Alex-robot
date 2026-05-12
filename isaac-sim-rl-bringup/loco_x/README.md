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

```bash
# Default: USD-prior demo, agent off — identical to Phase 1-4.
python -m loco_x.cli.run scene=room

# No-prior height-map exploration:
python -m loco_x.cli.run scene=room occupancy=heightmap

# With the LLM agent driving:
python -m loco_x.cli.run \
    scene=room \
    occupancy=heightmap \
    agent=anthropic \
    rerun=full
```

## Tests

```bash
python -m pytest tests/loco_x/ -q
```

The full sweep (`tests/autonomy/ tests/loco_x/`) must stay green on
every commit. LA-0a's `test_planner_paths_identical_to_phase_1_4_demo`
is the canary: if it drifts, the safety net broke.

## Phase status

* [x] LA-0a — `OccupancyProvider` interface + USD-provider refactor
* [ ] LA-0b.1 — synthetic point cloud harness
* [ ] LA-0b.2 — head-cam height-map integration
* [ ] LA-0c — frontier + info-gain + D14 semantic/variance costs
* [ ] LA-1 through LA-10 — see plan
