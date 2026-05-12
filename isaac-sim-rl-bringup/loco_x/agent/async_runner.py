"""Daemon-thread wrapper around :class:`AgentRunner` (LA-6).

The synchronous :class:`AgentRunner` blocks while the LLM client is
mid-call. In production the autonomy loop runs at 60 Hz and the LLM
round-trip is 1-3 s; blocking that loop would freeze the gait.

:class:`AsyncRunner` runs the sync runner's ``_tick`` on a daemon
thread. The autonomy loop calls ``poll(now)`` once per tick:

* If no turn is in flight and the gate permits, start one.
* Otherwise return immediately — non-blocking.

When the daemon-thread turn returns, ``turn_count`` and bundle state
reflect the result. The autonomy loop's next ``poll`` sees the
runner as idle again and may kick off a new turn.

The wrapper is deliberately minimal: one thread at a time, no
queueing, no asyncio. The runner's gating already handles "tick_hz
throttle" and "agent_should_stop", so the wrapper only needs to
prevent concurrent turns.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, Optional

from loco_x.llm.client import LLMClient

from .runner import AgentRunner, RunnerConfig


class AsyncRunner:
    """Daemon-thread wrapper around :class:`AgentRunner`.

    The wrapper owns the inner sync runner and forwards
    ``turn_count`` / ``stall_warning_active`` as live properties so
    tests can inspect state without reaching into internals.
    """

    def __init__(
        self,
        *,
        bundle: Dict[str, Any],
        client: LLMClient,
        config: Optional[RunnerConfig] = None,
    ) -> None:
        self._inner = AgentRunner(
            bundle=bundle, client=client, config=config,
        )
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._shutting_down = False

    # ── Public per-tick entrypoint ─────────────────────────────────
    def poll(self, now: float) -> None:
        """Non-blocking. Fire a turn on a daemon thread if the
        inner runner's gate permits AND no turn is currently in
        flight. Otherwise return immediately."""
        if self._shutting_down:
            return
        # Cheap check first — don't grab the lock just to bail.
        if self._thread is not None and self._thread.is_alive():
            return
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            # Pre-gate via inner — if it would deny, don't spawn.
            if not self._inner._gate(now):
                return
            self._thread = threading.Thread(
                target=self._inner._tick,
                args=(now,),
                daemon=True,
                name="loco_x-agent",
            )
            self._thread.start()

    # ── Lifecycle ──────────────────────────────────────────────────
    def shutdown(self, *, timeout_s: float = 5.0) -> None:
        """Wait for any in-flight turn to finish, then mark the
        runner shut down. Idempotent. Safe to call from atexit
        hooks."""
        self._shutting_down = True
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout_s)

    def is_alive(self) -> bool:
        """True iff a turn is currently in flight on the daemon
        thread."""
        return self._thread is not None and self._thread.is_alive()

    # ── Forwarded state (read-only) ────────────────────────────────
    @property
    def turn_count(self) -> int:
        return self._inner.turn_count

    @property
    def stall_warning_active(self) -> bool:
        return self._inner.stall_warning_active

    @property
    def config(self) -> RunnerConfig:
        return self._inner.config


__all__ = ["AsyncRunner"]
