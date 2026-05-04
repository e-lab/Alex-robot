"""Phase 4 — recovery agent + stuck monitor + closed-loop yaw tracker.

Three pure-logic state machines invoked once per autonomy tick (50 Hz).

- ``RecoveryAgent``  : owns IDLE / STANDING / ROTATING / FAILED states.
                       Outputs the appropriate ``_cmd`` tuple per state.
- ``YawTracker``     : closed-loop angular delta tracker, used to know
                       when a 90° rotation has actually completed.
- ``StuckMonitor``   : rolling-window detector that latches when the
                       robot has commanded forward velocity but failed
                       to translate by ``min_disp_m`` over ``window_s``.

Wall-clock is injected via the optional ``now`` parameter on every
method (mirrors ``GoalState.is_fresh(now=...)``). Tests run in
milliseconds without ``time.sleep``.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Optional, Tuple

from .translator import wrap_to_pi


# ── RecoveryAgent ────────────────────────────────────────────────────────────
class RecoveryState(str, Enum):
    """High-level recovery phase. ``str`` mixin lets enum values print
    cleanly in log lines and persist as JSON if we ever serialise."""
    IDLE     = "idle"        # not currently recovering
    STANDING = "standing"    # commanding (0,0,0,1) for stand_duration_s
    ROTATING = "rotating"    # commanding (0,0,sign*yaw_rate,0) until YawTracker says done
    FAILED   = "failed"      # max attempts exhausted; do nothing


@dataclass
class RecoveryAgent:
    """Phase-4 recovery state machine.

    Two concurrent goals: pick the robot back up after a fall (STANDING),
    or unstick it when it's pinned against unmodelled geometry
    (ROTATING). Only the fall path consumes the ``attempts_used`` budget;
    rotation is always free because the cause is geometry, not gait.
    """

    # Tunables — sourced from configs/autonomy/*.yaml at instantiation.
    stand_duration_s:    float = 3.0
    rotation_yaw_rate:   float = 0.4   # rad/s, gait-limited
    max_attempts:        int   = 2

    # Mutable state.
    state:               RecoveryState = RecoveryState.IDLE
    attempts_used:       int   = 0
    _phase_start_t:      float = 0.0   # wall-clock when current phase began

    # ── Fall path ────────────────────────────────────────────────────────
    def begin_standing(self, *, now: Optional[float] = None) -> None:
        """Enter the standing phase. Caller is responsible for first
        verifying the robot is fallen and that ``state`` is IDLE."""
        self.state = RecoveryState.STANDING
        self._phase_start_t = time.time() if now is None else float(now)

    def is_standing_done(self, *, now: Optional[float] = None) -> bool:
        """``True`` once the standing window has elapsed. ``False`` if
        we aren't currently in the standing phase (defensive)."""
        if self.state is not RecoveryState.STANDING:
            return False
        t = time.time() if now is None else float(now)
        return (t - self._phase_start_t) >= self.stand_duration_s

    # ── Stuck path ───────────────────────────────────────────────────────
    def begin_rotation(self) -> None:
        """Enter the rotation phase. Pairs with a freshly-started
        ``YawTracker``; "done" is determined by the tracker, not a
        timer, so we don't even need ``now`` here."""
        self.state = RecoveryState.ROTATING

    # ── Outcomes ─────────────────────────────────────────────────────────
    def succeed(self) -> None:
        """Caller has confirmed recovery (re-checked fall, or yaw tracker
        reached its target). Reset to IDLE and zero the attempt budget."""
        self.state = RecoveryState.IDLE
        self.attempts_used = 0

    def fail_attempt(self) -> None:
        """Caller has confirmed the standing window ended with the robot
        still fallen. Bump the counter; transition to FAILED if the
        budget is exhausted, else return to IDLE so the caller can
        re-trigger ``begin_standing`` for another shot.
        """
        self.attempts_used += 1
        if self.attempts_used >= self.max_attempts:
            self.state = RecoveryState.FAILED
        else:
            self.state = RecoveryState.IDLE

    # ── Output ───────────────────────────────────────────────────────────
    def cmd(self, *, rotation_sign: float = 1.0) -> Tuple[float, float, float, float]:
        """Return the ``(vx, vy, yaw_rate, standing)`` tuple appropriate
        for the current state. Caller writes into the global ``_cmd``.

        ``rotation_sign`` is supplied by the caller (positive for CCW,
        negative for CW) and matches the sign of ``YawTracker.target_rad``.
        """
        if self.state is RecoveryState.STANDING:
            return (0.0, 0.0, 0.0, 1.0)
        if self.state is RecoveryState.ROTATING:
            sign = 1.0 if rotation_sign >= 0.0 else -1.0
            return (0.0, 0.0, sign * self.rotation_yaw_rate, 0.0)
        if self.state is RecoveryState.FAILED:
            return (0.0, 0.0, 0.0, 1.0)
        # IDLE — defensive safe-stop. Caller shouldn't be invoking cmd()
        # from IDLE, but if they do, don't actuate.
        return (0.0, 0.0, 0.0, 1.0)


# ── YawTracker ───────────────────────────────────────────────────────────────
@dataclass
class YawTracker:
    """Closed-loop accumulator that tracks how far the robot has actually
    rotated since ``start()``, handling the ±π wrap-around.

    Each tick the caller passes ``yaw_now`` (the robot's current yaw in
    [-π, π]) and we add ``wrap_to_pi(yaw_now - prev_yaw)`` to ``delta``.
    Once ``|delta| >= |target_rad|``, ``update`` returns ``True``.

    Why this beats an open-loop timer: the walking policy doesn't track
    commanded yaw_rate exactly — gait limits, ground friction, and the
    occasional micro-step drift mean a 4-second open-loop rotation at
    0.4 rad/s could end up anywhere from 50° to 110°. Closed-loop on the
    actual ``root_quat_w`` yaw gives us a deterministic 90° regardless.
    """

    target_rad: float = math.pi / 2
    _prev_yaw:  float = 0.0
    delta:      float = 0.0
    active:     bool  = False

    def start(self, yaw_now: float) -> None:
        """Begin tracking from the current yaw. Resets accumulated delta."""
        self._prev_yaw = float(yaw_now)
        self.delta = 0.0
        self.active = True

    def update(self, yaw_now: float) -> bool:
        """Add the yaw step since last tick to ``delta``; return ``True``
        once |delta| has reached |target_rad|. Safe to call before
        ``start()`` (returns False)."""
        if not self.active:
            return False
        step = wrap_to_pi(float(yaw_now) - self._prev_yaw)
        self.delta += step
        self._prev_yaw = float(yaw_now)
        return abs(self.delta) >= abs(self.target_rad)


# ── StuckMonitor ─────────────────────────────────────────────────────────────
@dataclass
class StuckMonitor:
    """Latch ``True`` when the robot has been actively *trying* to move
    forward for at least ``window_s``, but the total XY displacement
    over that window is less than ``min_disp_m``.

    Sample buffer is dropped whenever the caller flags ``active=False``
    (FSM not in APPROACH, or ``vx == 0``). This means we can't accumulate
    stuck-time during emergency-brake holds, recovery rotations, or
    SEARCH spins — exactly the desired behaviour.

    Mirrors the shape of ``FallMonitor``: latches True until ``reset()``.
    """

    window_s:    float = 5.0
    min_disp_m:  float = 0.2
    samples:     Deque[Tuple[float, float, float]] = field(default_factory=deque)
    stuck:       bool  = False

    def update(
        self,
        x: float,
        y: float,
        *,
        active: bool,
        now: Optional[float] = None,
    ) -> bool:
        """Returns the (possibly newly-latched) stuck status.

        Caller is responsible for invoking ``reset()`` after corrective
        action, or the latch persists.
        """
        t = time.time() if now is None else float(now)

        if not active:
            self.samples.clear()
            return self.stuck

        # Append, then evict the second-oldest sample whenever the
        # *next-oldest* would still cover the window. This keeps exactly
        # one "anchor" sample older than the cutoff so the displacement
        # comparison always has a window-spanning reference.
        self.samples.append((t, float(x), float(y)))
        cutoff = t - self.window_s
        while len(self.samples) >= 2 and self.samples[1][0] <= cutoff:
            self.samples.popleft()

        # Need ≥ 2 samples spanning at least the full window before we
        # can decide "stuck" — otherwise we'd false-positive on the
        # first sample after activation.
        if len(self.samples) < 2:
            return self.stuck
        oldest_t, ox, oy = self.samples[0]
        if (t - oldest_t) < self.window_s:
            return self.stuck

        disp = math.hypot(x - ox, y - oy)
        if disp < self.min_disp_m:
            self.stuck = True
        return self.stuck

    def reset(self) -> None:
        """Clear the latch and the rolling buffer. Subsequent updates
        require a fresh full window before the monitor can re-latch."""
        self.stuck = False
        self.samples.clear()


__all__ = [
    "RecoveryAgent",
    "RecoveryState",
    "StuckMonitor",
    "YawTracker",
]
