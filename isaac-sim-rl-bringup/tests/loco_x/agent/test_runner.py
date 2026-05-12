"""Tests for the agent runner — the closed loop (LA-5).

The runner is the small piece that ties LA-2 (observation), LA-4
(LLM client), and LA-1 (sandbox + skills) together. Each tick:

* gating decides whether to run at all (agent.enabled, stop flag,
  FSM idle, tick_hz throttle),
* the observation is built from the bundle snapshot,
* the LLM is queried,
* the returned code goes through the sandbox,
* skill results land in ``bundle["last_action"]`` for the next turn,
* D11 watchdogs check progress and turn budget.

These tests exercise every gating branch and every termination path
with a :class:`ScriptedClient` so no network or LLM mocks are needed.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import pytest

from loco_x.agent.runner import AgentRunner, RunnerConfig
from loco_x.llm.client import ScriptedClient
from loco_x.occupancy import CellState
from loco_x.occupancy import FrontierCandidate


# ── Bundle / provider stubs ────────────────────────────────────────────────
class _StubProvider:
    def __init__(self, *, visited=0.05, frontier=(1.0, 0.0)):
        self._vf = visited
        self._frontier = frontier

    def visited_fraction(self): return self._vf
    def frontier_cells(self, *, from_xy=None, k=1, prefer_near=None):
        if self._frontier is None:
            return []
        return [FrontierCandidate(
            world_xy=self._frontier, info_gain=20.0,
            travel_distance=1.0, score=10.0,
        )]
    def max_path_staleness(self, path): return (0.0, None)
    def path_invalidated_by_new_obstacle(self, path): return None
    def query(self, xy):
        if abs(xy[0]) > 100 or abs(xy[1]) > 100:
            return CellState.UNKNOWN
        return CellState.FREE
    def origin_xy(self): return (-10.0, -10.0)
    def resolution_m(self): return 0.1


def _bundle(
    *,
    fsm="IDLE",
    scene_nodes=None,
    occ_provider=None,
    task_queue=None,
):
    return {
        "robot_pose": {"xy": (0.0, 0.0), "yaw_rad": 0.0},
        "fsm_mode": fsm,
        "scene_nodes": scene_nodes or [{"label": "stove",
                                         "world_xy": (2.0, 0.0),
                                         "last_seen": 1.0,
                                         "confidence": 0.81}],
        "occ_provider": occ_provider or _StubProvider(),
        "goal_label": None,
        "last_action": None,
        "path": None,
        "path_index": 0,
        "agent_should_stop": False,
        "task_result_status": None,
        "task_result_reason": None,
        "task_queue": task_queue if task_queue is not None else [],
        "fsm_mode": fsm,
    }


def _runner(bundle, *, responses, multimodal_responses=None, cfg=None):
    """Build a test runner. Test default config disables the startup
    delay (which is meant for real-world Isaac startup) so tests can
    fire turns at now=0 without a 2 s warm-up."""
    client = ScriptedClient(
        responses=responses,
        multimodal_responses=multimodal_responses,
    )
    return AgentRunner(
        bundle=bundle,
        client=client,
        config=cfg or RunnerConfig(startup_delay_s=0.0, verbose=False),
    )


# ── Gating ─────────────────────────────────────────────────────────────────
def test_runner_skips_tick_when_disabled() -> None:
    """If agent.enabled=false, ``maybe_tick`` is a no-op — no LLM call,
    no observation built, no state mutation."""
    bundle = _bundle()
    runner = _runner(bundle, responses=["``` ``` `python\nstop()\n```"])
    runner.config = RunnerConfig(enabled=False)
    runner.maybe_tick(now=0.0)
    assert runner.turn_count == 0
    assert bundle["last_action"] is None


def test_runner_skips_tick_when_stopped() -> None:
    """Once a skill has set ``agent_should_stop``, further ticks are
    no-ops. The autonomy script may keep calling ``maybe_tick`` but
    the runner stays quiescent."""
    bundle = _bundle()
    bundle["agent_should_stop"] = True
    runner = _runner(bundle, responses=["```python\nstop()\n```"])
    runner.maybe_tick(now=0.0)
    assert runner.turn_count == 0


def test_runner_skips_tick_when_fsm_not_idle() -> None:
    """The LLM only thinks while the FSM is idle. While the autonomy
    loop is mid-action (APPROACH / EXPLORE / RECOVERY), the runner
    waits."""
    bundle = _bundle(fsm="APPROACH")
    runner = _runner(bundle, responses=["```python\nstop()\n```"])
    runner.maybe_tick(now=0.0)
    assert runner.turn_count == 0


def test_runner_skips_tick_when_queue_non_empty() -> None:
    """If the task queue still has work to dispatch from a prior turn
    the agent doesn't think yet — give the autonomy loop a chance to
    drain. (Tested explicitly because empty-queue and idle-FSM are
    related but distinct signals.)"""
    bundle = _bundle(task_queue=[{"kind": "goto", "label": "stove"}])
    runner = _runner(bundle, responses=["```python\nstop()\n```"])
    runner.maybe_tick(now=0.0)
    assert runner.turn_count == 0


def test_runner_throttles_to_tick_hz() -> None:
    """``tick_hz`` caps how often ``maybe_tick`` actually fires —
    1/tick_hz seconds between LLM calls minimum. The autonomy loop is
    expected to drain ``task_queue`` between turns; we simulate that
    explicitly here."""
    bundle = _bundle()
    runner = _runner(bundle, responses=[
        "```python\nstop()\n```",
        "```python\nstop()\n```",
    ], cfg=RunnerConfig(tick_hz=2.0, startup_delay_s=0.0, verbose=False))
    runner.maybe_tick(now=0.0)        # first tick fires
    assert runner.turn_count == 1
    bundle["task_queue"] = []         # autonomy loop drained it
    runner.maybe_tick(now=0.1)        # 100 ms later — throttled by tick_hz
    assert runner.turn_count == 1
    runner.maybe_tick(now=0.6)        # 600 ms later — fires again
    assert runner.turn_count == 2


# ── End-to-end normal path ─────────────────────────────────────────────────
def test_runner_normal_turn_executes_code_and_records_last_action() -> None:
    """Standard turn: LLM emits a goto, sandbox runs it, the result
    lands in bundle["last_action"] for the next observation."""
    bundle = _bundle()
    runner = _runner(bundle, responses=[
        "```python\ngoto('stove')\n```",
    ])
    runner.maybe_tick(now=0.0)
    assert runner.turn_count == 1
    # The goto skill pushed a task.
    assert any(t["kind"] == "goto" for t in bundle["task_queue"])
    # last_action records the skill result so it surfaces next obs.
    last = bundle["last_action"]
    assert last is not None
    assert last["status"] == "queued"


def test_runner_executes_multi_target_chain_then_finishes() -> None:
    """Canonical multi-target run: goto stove, goto sink, finish.
    Three turns; finish() sets the stop flag and the runner unwinds."""
    bundle = _bundle(scene_nodes=[
        {"label": "stove", "world_xy": (2.0, 0.0),
         "last_seen": 1.0, "confidence": 0.81},
        {"label": "sink", "world_xy": (-1.5, 2.0),
         "last_seen": 1.0, "confidence": 0.62},
    ])
    runner = _runner(bundle, responses=[
        "```python\ngoto('stove')\n```",
        "```python\ngoto('sink')\n```",
        "```python\nfinish('visited both')\n```",
    ])
    # Three ticks with enough gap to clear tick_hz throttle.
    for i in range(3):
        runner.maybe_tick(now=float(i) * 10.0)
        # Re-clear FSM + task queue between turns so the gate re-opens.
        bundle["fsm_mode"] = "IDLE"
        bundle["task_queue"] = []
    assert runner.turn_count == 3
    assert bundle["agent_should_stop"] is True
    assert bundle["task_result_status"] == "succeeded"


def test_runner_handles_finish_signal_alone() -> None:
    """If the LLM emits bare FINISH with no code (parser
    signal=='finish'), the runner sets succeeded and unwinds."""
    bundle = _bundle()
    runner = _runner(bundle, responses=["task done.\n\nFINISH\n"])
    runner.maybe_tick(now=0.0)
    assert bundle["agent_should_stop"] is True
    assert bundle["task_result_status"] == "succeeded"


# ── Sandbox / skill failure paths ──────────────────────────────────────────
def test_runner_records_ast_rejection_as_last_action() -> None:
    """If the LLM emits forbidden Python (import os), the sandbox
    rejects it; the runner records an error feedback in
    bundle["last_action"] so the next observation tells the LLM what
    happened. The runner does NOT stop — the LLM gets another turn."""
    bundle = _bundle()
    runner = _runner(bundle, responses=[
        "```python\nimport os\ngoto('stove')\n```",
        "```python\nfinish('recovered')\n```",
    ])
    runner.maybe_tick(now=0.0)
    last = bundle["last_action"]
    assert last is not None
    assert last["status"] == "error"
    assert last["error_kind"] == "sandbox_rejected"
    # The runner did NOT terminate; it's ready for the next turn.
    assert bundle["agent_should_stop"] is False


def test_runner_records_sandbox_timeout_as_last_action() -> None:
    """An infinite loop kills via the wall-clock timeout. Same
    treatment as AST rejection: report and continue."""
    bundle = _bundle()
    cfg = RunnerConfig(exec_timeout_s=0.1, startup_delay_s=0.0, verbose=False)
    runner = _runner(bundle, responses=[
        "```python\nwhile True:\n    pass\n```",
    ], cfg=cfg)
    runner.maybe_tick(now=0.0)
    last = bundle["last_action"]
    assert last is not None
    assert last["error_kind"] == "sandbox_timeout"
    assert bundle["agent_should_stop"] is False


def test_runner_records_skill_exception_as_last_action() -> None:
    """A skill raising (TypeError on bad argument type) → captured as
    error feedback, not propagated as a crash."""
    bundle = _bundle()
    runner = _runner(bundle, responses=[
        "```python\nface('not a number')\n```",  # face expects float
    ])
    runner.maybe_tick(now=0.0)
    last = bundle["last_action"]
    assert last is not None
    assert last["status"] == "error"
    assert "skill" in last["error_kind"] or "exception" in last["error_kind"]
    assert bundle["agent_should_stop"] is False


def test_runner_records_parse_error_as_last_action() -> None:
    """The LLM emits text with no code and no signal → LLMParseError.
    The runner records it and asks for another turn."""
    bundle = _bundle()
    runner = _runner(bundle, responses=[
        "I'm thinking about it.",
        "```python\nfinish('recovered')\n```",
    ])
    runner.maybe_tick(now=0.0)
    last = bundle["last_action"]
    assert last is not None
    assert last["error_kind"] == "parse_error"
    assert bundle["agent_should_stop"] is False


# ── D11 termination paths ─────────────────────────────────────────────────
def test_runner_force_fails_on_max_turns() -> None:
    """D11 Case B — hard backstop. After ``max_turns`` ticks the
    runner calls fail() on the agent's behalf, even if the LLM hasn't.
    Reason quotes 'turn budget exhausted'."""
    bundle = _bundle()
    # 4 turns budgeted but the LLM never calls finish/fail.
    cfg = RunnerConfig(max_turns=3, startup_delay_s=0.0, verbose=False)
    runner = _runner(bundle, responses=[
        "```python\ngoto('stove')\n```",
        "```python\ngoto('stove')\n```",
        "```python\ngoto('stove')\n```",
    ], cfg=cfg)
    for i in range(4):
        runner.maybe_tick(now=float(i) * 10.0)
        bundle["fsm_mode"] = "IDLE"
        bundle["task_queue"] = []
    assert bundle["agent_should_stop"] is True
    assert bundle["task_result_status"] == "failed"
    assert "turn budget" in bundle["task_result_reason"].lower()


def test_runner_progress_stall_warning_appended_to_observation() -> None:
    """D11 Case D — when visited_fraction hasn't grown by >0.02 over
    the last 5 turns, the runner injects a 'warning: no exploration
    progress for 5 turns' line into the *next* observation.

    We can't easily inspect the observation string the runner gives
    to the LLM (it's an internal call), but the runner exposes
    ``stall_warning_active`` so a test can verify."""
    bundle = _bundle(occ_provider=_StubProvider(visited=0.05))
    cfg = RunnerConfig(progress_stall_window_turns=3,
                       progress_stall_threshold=0.02,
                       startup_delay_s=0.0, verbose=False)
    runner = _runner(bundle, responses=[
        "```python\nstop()\n```",
        "```python\nstop()\n```",
        "```python\nstop()\n```",
        "```python\nstop()\n```",
    ], cfg=cfg)
    for i in range(4):
        runner.maybe_tick(now=float(i) * 10.0)
        bundle["fsm_mode"] = "IDLE"
        bundle["task_queue"] = []
    # After 3 stalled turns the warning should be active.
    assert runner.stall_warning_active is True


def test_runner_clears_stall_warning_when_progress_resumes() -> None:
    """If visited_fraction jumps, the stall warning clears so the
    LLM doesn't see it on subsequent turns."""
    provider = _StubProvider(visited=0.05)
    bundle = _bundle(occ_provider=provider)
    cfg = RunnerConfig(progress_stall_window_turns=3,
                       progress_stall_threshold=0.02,
                       startup_delay_s=0.0, verbose=False)
    runner = _runner(bundle, responses=[
        "```python\nstop()\n```",
        "```python\nstop()\n```",
        "```python\nstop()\n```",
        "```python\nstop()\n```",
    ], cfg=cfg)
    for i in range(3):
        runner.maybe_tick(now=float(i) * 10.0)
        bundle["fsm_mode"] = "IDLE"
        bundle["task_queue"] = []
    assert runner.stall_warning_active is True
    # Progress jumps.
    provider._vf = 0.30
    runner.maybe_tick(now=30.0)
    assert runner.stall_warning_active is False


# ── Multi-target failure: no silent retarget ──────────────────────────────
def test_runner_does_not_auto_retarget_on_skill_failure() -> None:
    """D11 contract: when a skill returns an error dict, the runner
    does NOT silently retry or pick a related target. The LLM gets
    the error in last_action and decides what to do next."""
    bundle = _bundle()  # no 'kettle' in scene_nodes
    runner = _runner(bundle, responses=[
        "```python\ngoto('kettle')\n```",   # unknown_label
        "```python\nfinish('giving up')\n```",
    ])
    runner.maybe_tick(now=0.0)
    last = bundle["last_action"]
    assert last is not None
    assert last["status"] == "error"
    assert last["error_kind"] == "unknown_label"
    # Crucially: bundle["task_queue"] is empty — no silent retarget.
    assert bundle["task_queue"] == []
    # And the runner is ready for the next turn.
    assert bundle["agent_should_stop"] is False
