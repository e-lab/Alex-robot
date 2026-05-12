"""Tests for the LA-6 task dispatcher.

LA-1 skills push tasks onto ``bundle["task_queue"]`` rather than
acting directly. The dispatcher is the autonomy-side consumer: each
autonomy tick (60 Hz), it drains the queue and translates each task
into the Phase 1-4 control surface.

Translation table (lives in the dispatcher):

* ``{kind: goto, label, world_xy}``     → set goal_label + goal_lock_xyz
* ``{kind: goto_xy, xy}``                → seed goal_lock_xyz with no label
* ``{kind: face, yaw_rad}``              → set face_yaw_rad
* ``{kind: stop}``                       → set safe_stop_requested
* ``{kind: peek, direction}``            → set head_yaw_request
* ``{kind: survey, angles_deg}``         → set head_sweep_queue

The dispatcher is a pure function over the bundle — no Isaac, no
threading. The Isaac glue (read these fields and act on them) lives
in the existing autonomy loop; LA-6's Isaac-side edit is one line
that calls ``dispatch(bundle)`` once per tick.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

import pytest

from loco_x.agent.dispatcher import (
    TaskDispatcher,
    UnknownTaskKind,
)


def _bundle() -> Dict[str, Any]:
    return {
        "task_queue": [],
        # Phase 1-4 control-surface fields the dispatcher writes into.
        "goal_label": None,
        "goal_lock_xyz": None,
        "face_yaw_rad": None,
        "safe_stop_requested": False,
        "head_yaw_request": None,
        "head_sweep_queue": None,
    }


# ── Locomotion task dispatch ───────────────────────────────────────────────
def test_dispatch_goto_sets_goal_label_and_xy() -> None:
    """``goto(label)`` skill pushed
    ``{kind: goto, label, world_xy}``; dispatcher seeds goal_label
    and goal_lock_xyz so the existing FSM picks it up next tick."""
    bundle = _bundle()
    bundle["task_queue"].append({
        "kind": "goto", "label": "stove", "world_xy": (2.0, 0.0),
    })
    d = TaskDispatcher()
    d.drain(bundle)
    assert bundle["goal_label"] == "stove"
    assert bundle["goal_lock_xyz"] == (2.0, 0.0)
    assert bundle["task_queue"] == []


def test_dispatch_goto_xy_seeds_goal_lock_with_no_label() -> None:
    """``goto_xy(x, y)`` is unlabeled; the FSM uses the coords
    directly. goal_label stays None."""
    bundle = _bundle()
    bundle["task_queue"].append({"kind": "goto_xy", "xy": (1.5, 0.7)})
    TaskDispatcher().drain(bundle)
    assert bundle["goal_label"] is None
    assert bundle["goal_lock_xyz"] == (1.5, 0.7)


def test_dispatch_face_sets_yaw() -> None:
    bundle = _bundle()
    bundle["task_queue"].append({"kind": "face", "yaw_rad": math.pi / 2})
    TaskDispatcher().drain(bundle)
    assert bundle["face_yaw_rad"] == math.pi / 2


def test_dispatch_stop_sets_safe_stop_flag() -> None:
    bundle = _bundle()
    bundle["task_queue"].append({"kind": "stop"})
    TaskDispatcher().drain(bundle)
    assert bundle["safe_stop_requested"] is True


# ── Perception task dispatch ───────────────────────────────────────────────
def test_dispatch_peek_sets_head_yaw_request() -> None:
    """The autonomy loop reads ``head_yaw_request`` next tick and
    issues the neck joint command."""
    bundle = _bundle()
    bundle["task_queue"].append({"kind": "peek", "direction": "left"})
    TaskDispatcher().drain(bundle)
    assert bundle["head_yaw_request"] == "left"


def test_dispatch_survey_sets_head_sweep_queue() -> None:
    bundle = _bundle()
    bundle["task_queue"].append({
        "kind": "survey", "angles_deg": [-45, 0, 45],
    })
    TaskDispatcher().drain(bundle)
    assert bundle["head_sweep_queue"] == [-45, 0, 45]


# ── Queue drain semantics ──────────────────────────────────────────────────
def test_dispatch_drains_queue_in_order() -> None:
    """Multi-task queue: dispatcher processes in FIFO order so a
    chain like goto+face+stop applies cleanly."""
    bundle = _bundle()
    bundle["task_queue"].extend([
        {"kind": "goto", "label": "stove", "world_xy": (2.0, 0.0)},
        {"kind": "face", "yaw_rad": 0.0},
        {"kind": "stop"},
    ])
    TaskDispatcher().drain(bundle)
    assert bundle["goal_label"] == "stove"
    assert bundle["face_yaw_rad"] == 0.0
    assert bundle["safe_stop_requested"] is True
    assert bundle["task_queue"] == []


def test_dispatch_empty_queue_is_noop() -> None:
    bundle = _bundle()
    snapshot = dict(bundle)
    TaskDispatcher().drain(bundle)
    # Nothing changed except the queue stayed empty.
    assert bundle["task_queue"] == []
    assert bundle["goal_label"] == snapshot["goal_label"]


# ── Unknown-task handling ──────────────────────────────────────────────────
def test_dispatch_unknown_task_kind_raises() -> None:
    """An unknown task kind is a programmer bug (skill author drift),
    not LLM hallucination — the LLM can only produce kinds the
    skills push. Raise so the autonomy loop's logger catches it."""
    bundle = _bundle()
    bundle["task_queue"].append({"kind": "teleport", "xy": (1, 1)})
    with pytest.raises(UnknownTaskKind):
        TaskDispatcher().drain(bundle)


def test_dispatch_logs_task_history_for_postmortem() -> None:
    """Each drained task is appended to ``bundle["task_history"]`` so
    the end-of-run timing report can show what the agent did."""
    bundle = _bundle()
    bundle["task_history"] = []
    bundle["task_queue"].extend([
        {"kind": "goto", "label": "stove", "world_xy": (2.0, 0.0)},
        {"kind": "stop"},
    ])
    TaskDispatcher().drain(bundle)
    assert len(bundle["task_history"]) == 2
    assert bundle["task_history"][0]["kind"] == "goto"
    assert bundle["task_history"][1]["kind"] == "stop"
