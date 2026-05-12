"""Tests for AsyncRunner — the daemon-thread wrapper (LA-6).

The synchronous :class:`AgentRunner` from LA-5 blocks while the LLM
client is mid-call. In production that's a 1-3 s wait on a network
round-trip; the autonomy loop runs at 60 Hz and cannot afford it.

:class:`AsyncRunner` wraps the sync runner on a daemon thread. The
autonomy loop calls ``poll(now)`` once per tick — non-blocking — and
the wrapper fires off a new turn only when the previous one has
returned. While the LLM is in flight, ``poll`` returns immediately.

Tests use a deliberately slow stub client to confirm the wrapper
doesn't block the calling thread.
"""
from __future__ import annotations

import threading
import time
from typing import List

import pytest

from loco_x.agent.async_runner import AsyncRunner
from loco_x.agent.runner import RunnerConfig
from loco_x.llm.client import LLMClient, Message
from loco_x.llm.parsers import LLMResponse


class _SlowScriptedClient:
    """Like ScriptedClient but each query() takes ``delay_s``
    seconds — proxies a slow LLM. Tests use 0.1-0.2 s so the suite
    still completes in <1 s per test."""

    def __init__(self, responses: List[str], *, delay_s: float = 0.1):
        self._responses = list(responses)
        self._idx = 0
        self._delay_s = delay_s

    def query(self, messages):
        time.sleep(self._delay_s)
        if self._idx >= len(self._responses):
            raise IndexError("scripted exhausted")
        from loco_x.llm.parsers import parse_response
        raw = self._responses[self._idx]
        self._idx += 1
        return parse_response(raw)

    def query_multimodal(self, *, messages, image_bytes):
        return "stub caption"


def _bundle(*, scene_nodes=None):
    return {
        "robot_pose": {"xy": (0.0, 0.0), "yaw_rad": 0.0},
        "fsm_mode": "IDLE",
        "scene_nodes": scene_nodes or [{"label": "stove",
                                         "world_xy": (2.0, 0.0),
                                         "last_seen": 1.0, "confidence": 0.81}],
        "occ_provider": None,
        "goal_label": None,
        "last_action": None,
        "path": None,
        "path_index": 0,
        "agent_should_stop": False,
        "task_result_status": None,
        "task_result_reason": None,
        "task_queue": [],
    }


# ── Non-blocking poll ──────────────────────────────────────────────────────
def test_async_poll_does_not_block_on_slow_client() -> None:
    """The whole point of the async wrapper: ``poll`` returns
    immediately even when the LLM call takes ~0.2 s."""
    bundle = _bundle()
    client = _SlowScriptedClient(
        responses=["```python\nfinish('done')\n```"],
        delay_s=0.2,
    )
    runner = AsyncRunner(bundle=bundle, client=client,
                         config=RunnerConfig(tick_hz=10.0, startup_delay_s=0.0, verbose=False))
    t0 = time.monotonic()
    runner.poll(now=0.0)
    dt = time.monotonic() - t0
    # Poll itself must be quick — well under the client's delay.
    assert dt < 0.05, f"poll blocked for {dt:.3f}s"
    runner.shutdown(timeout_s=1.0)


def test_async_poll_eventually_completes_turn() -> None:
    """After enough polls (long enough for the slow client to finish),
    the runner has executed the canned response — bundle reflects it."""
    bundle = _bundle()
    client = _SlowScriptedClient(
        responses=["```python\ngoto('stove')\n```"],
        delay_s=0.1,
    )
    runner = AsyncRunner(bundle=bundle, client=client,
                         config=RunnerConfig(tick_hz=10.0, startup_delay_s=0.0, verbose=False))
    # Fire a poll, wait for the turn to finish, then poll again to
    # confirm state.
    runner.poll(now=0.0)
    # Wait up to 1 s for the slow turn to complete.
    deadline = time.monotonic() + 1.0
    while runner.turn_count == 0 and time.monotonic() < deadline:
        time.sleep(0.02)
    runner.shutdown(timeout_s=1.0)
    assert runner.turn_count == 1
    # The skill enqueued a goto.
    assert any(t["kind"] == "goto" for t in bundle["task_queue"])


def test_async_poll_does_not_fire_concurrent_turns() -> None:
    """If the slow turn is still in flight, subsequent polls must
    NOT start another one. Otherwise we'd race on bundle writes."""
    bundle = _bundle()
    client = _SlowScriptedClient(
        responses=["```python\nfinish('a')\n```",
                   "```python\nfinish('b')\n```"],
        delay_s=0.2,
    )
    runner = AsyncRunner(bundle=bundle, client=client,
                         config=RunnerConfig(tick_hz=100.0, startup_delay_s=0.0, verbose=False))
    runner.poll(now=0.0)
    # Hammer poll() — none of these should kick off a second turn
    # while the first is in flight.
    for k in range(10):
        runner.poll(now=k * 0.001)
        time.sleep(0.005)
    # Now wait for the first turn to settle.
    deadline = time.monotonic() + 1.0
    while runner.turn_count == 0 and time.monotonic() < deadline:
        time.sleep(0.02)
    runner.shutdown(timeout_s=1.0)
    # Exactly one turn executed.
    assert runner.turn_count == 1


# ── Lifecycle ─────────────────────────────────────────────────────────────
def test_async_shutdown_terminates_thread() -> None:
    """``shutdown`` joins the daemon thread cleanly so the autonomy
    script can end without leaving orphans."""
    bundle = _bundle()
    client = _SlowScriptedClient(
        responses=["```python\nfinish('done')\n```"],
        delay_s=0.05,
    )
    runner = AsyncRunner(bundle=bundle, client=client,
                         config=RunnerConfig(tick_hz=10.0, startup_delay_s=0.0, verbose=False))
    runner.poll(now=0.0)
    # Let the turn complete.
    deadline = time.monotonic() + 1.0
    while runner.turn_count == 0 and time.monotonic() < deadline:
        time.sleep(0.02)
    runner.shutdown(timeout_s=1.0)
    assert not runner.is_alive()


def test_async_shutdown_handles_idle_runner() -> None:
    """A runner that never fired (no poll() yet) shuts down cleanly
    without raising."""
    bundle = _bundle()
    client = _SlowScriptedClient(responses=[], delay_s=0.05)
    runner = AsyncRunner(bundle=bundle, client=client)
    runner.shutdown(timeout_s=1.0)
    assert not runner.is_alive()
