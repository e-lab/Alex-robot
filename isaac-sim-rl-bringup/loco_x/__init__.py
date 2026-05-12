"""Loco-Agent0 — agentic locomotion framework on top of Phase 1-4 primitives.

Adapts the CaP-Agent0 pattern (arXiv 2603.22435, github.com/capgym/cap-x)
to Alex's locomotion + scene-graph stack. See PLAN/loco_agent0_plan.md.

Module map (subpackages mirror cap-x where the analogy holds; see plan
§ Critical files for the canonical list):

    loco_x.occupancy     —  runtime occupancy provider interface (D10).
                            USD wrapper for sim-with-prior; height map
                            for the no-prior real-world / no-USD case.
    loco_x.perception    —  agent-facing observation builder (D2) +
                            spatial-temporal scene-graph filter.
    loco_x.skills        —  the skill library the LLM is allowed to
                            call (D4). One module per category.
    loco_x.llm           —  text + multimodal client (D6, D12), prompt
                            templates (D2, D12), code-block parser.
    loco_x.agent         —  the agent loop (runner) + AST/exec-timeout
                            sandbox (D1).
    loco_x.cli           —  Hydra @main entry points.
    loco_x.conf          —  Hydra config tree (composed at startup).

Loco-X imports from but never modifies:

    scripts.alex_room_explore.autonomy  —  Phase 1-4 primitives
                                           (planner, FSM, recovery,
                                           goal lock, USD occupancy).
    scene_graph                         —  SAM3 + scene-graph maintenance.

This boundary is deliberate: Phase 1-4 stays frozen as the safety net
we benchmark against.
"""
