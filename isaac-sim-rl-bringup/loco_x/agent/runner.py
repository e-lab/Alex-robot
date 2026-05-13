"""Agent runner — the closed loop (LA-5).

Threads together :mod:`loco_x.perception.observation` (LA-2),
:mod:`loco_x.llm.client` (LA-4), :mod:`loco_x.agent.sandbox` (LA-1),
and :mod:`loco_x.skills` (LA-1). One ``maybe_tick(now)`` per autonomy
tick; the runner gates on FSM idle + tick_hz throttle + stop flag and
only fires when all three permit.

Each fired tick:

1. Build the per-turn observation (D2 + D9 + D13 snapshot).
2. Compose a message list with the system prompt + observation.
3. Call ``client.query(messages)`` → :class:`LLMResponse`.
4. Handle the parsed ``signal`` (``finish`` → succeed-unwind;
   ``regenerate`` → execute code and loop again next tick).
5. Run ``response.code`` through :class:`Sandbox`. Skill failures /
   AST rejection / wall-clock timeout / parse errors all land in
   ``bundle["last_action"]`` as D4-shaped error dicts so the next
   observation surfaces them.
6. Update D11 watchdogs: progress-stall warning (Case D) and
   max-turns backstop (Case B).

The runner is **synchronous** in LA-5: ``maybe_tick`` calls the LLM
client inline. The autonomy script that hosts the runner runs at
60&nbsp;Hz and the LLM round-trip is ~1-3&nbsp;s, so in production we
will wrap this in a daemon-thread Future poll (LA-6) to keep the
60&nbsp;Hz loop non-blocking. The synchronous core is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from collections import deque

from loco_x.agent.sandbox import (
    Sandbox,
    SandboxRejected,
    SandboxResult,
    SandboxTimeout,
)
from loco_x.llm.client import LLMClient, Message
from loco_x.llm.parsers import LLMParseError
from loco_x.perception.observation import build_observation
from loco_x.skills import make_skills


# ── Config ─────────────────────────────────────────────────────────────────
@dataclass
class RunnerConfig:
    """All knobs that affect runner behavior in one struct.

    Matches the ``AgentCfg`` Hydra group (``loco_x/conf/agent/*.yaml``).
    Defaults mirror D7 / D11 / D13 conventions.
    """

    enabled: bool = True
    tick_hz: float = 2.0
    max_turns: int = 20
    exec_timeout_s: float = 5.0
    # D11 Case D — progress watchdog.
    progress_stall_window_turns: int = 5
    progress_stall_threshold: float = 0.02
    # System prompt; the runner injects it as the first message. LA-1
    # ships an empty default; LA-6 will inject the real decision-table
    # prompt from loco_x.llm.prompts.
    system_prompt: str = ""
    # LA-7 diagnostic: print one line per turn so the sim log shows
    # the agent at work. Default on for the integrated runtime.
    verbose: bool = True
    # LA-7: give perception (SAM3 + Phase-1-4 goal lock) time to
    # produce enough observations before the agent fires its first
    # turn. SAM3's min_observations gate (default 3) plus the
    # goal-lock confidence threshold typically take ~5 s from a
    # clean start; firing the agent earlier produces empty-graph
    # observations and burns turns on stop()/find() that can't
    # succeed yet.
    startup_delay_s: float = 5.0


# ── Runner ─────────────────────────────────────────────────────────────────
class AgentRunner:
    """Per-task agent loop wrapper. One runner per agent task; the
    autonomy script holds it on the bundle as ``bundle["agent"]``."""

    def __init__(
        self,
        *,
        bundle: Dict[str, Any],
        client: LLMClient,
        config: Optional[RunnerConfig] = None,
    ) -> None:
        self.bundle = bundle
        self.client = client
        self.config = config or RunnerConfig()
        self.turn_count: int = 0
        self._last_tick_t: Optional[float] = None
        # Tracks the first ``now`` we ever see; used to enforce
        # ``startup_delay_s`` so the agent doesn't tick before
        # perception has had a chance to run.
        self._first_poll_t: Optional[float] = None
        # Rolling buffer of recent visited_fraction values for D11.D.
        self._visited_history: Deque[float] = deque(
            maxlen=max(1, self.config.progress_stall_window_turns + 1)
        )
        self.stall_warning_active: bool = False
        # Conversation history kept short — D2 says the observation
        # carries enough context per turn, so the LLM gets just the
        # current observation each call. We keep system + last user
        # for shape; richer history can land later if it earns it.
        self._system_msg = Message(role="system", content=config.system_prompt) if config else None

    # ── Public entrypoint ──────────────────────────────────────────
    def maybe_tick(self, now: float) -> None:
        """Per-autonomy-tick gate. Cheap when gating denies (no LLM
        call); does the full turn when it allows."""
        if not self._gate(now):
            return
        self._tick(now)

    def reason_for_skip(self, now: float) -> Optional[str]:
        """Diagnostic helper. Returns a short string describing why
        the gate would deny ``maybe_tick(now)``, or ``None`` if the
        gate would permit. Used by the autonomy-script integration
        to print agent status when the agent isn't ticking."""
        cfg = self.config
        if not cfg.enabled:
            return "agent.enabled=false"
        if self.bundle.get("agent_should_stop"):
            return "agent_should_stop=true"
        fsm = self.bundle.get("fsm_mode", "IDLE")
        if fsm not in ("IDLE", "ARRIVED"):
            return f"fsm={fsm} (waiting for IDLE/ARRIVED)"
        if self.bundle.get("task_queue"):
            return f"task_queue not empty ({len(self.bundle['task_queue'])} pending)"
        if self.bundle.get("face_yaw_rad") is not None:
            return f"face rotation in progress (target={self.bundle['face_yaw_rad']:.2f} rad)"
        if cfg.startup_delay_s > 0.0 and self._first_poll_t is not None:
            elapsed = now - self._first_poll_t
            if elapsed < cfg.startup_delay_s:
                return (f"startup delay ({elapsed:.1f}s/"
                        f"{cfg.startup_delay_s:.1f}s; waiting for perception)")
        if cfg.tick_hz > 0.0 and self._last_tick_t is not None:
            min_dt = 1.0 / cfg.tick_hz
            dt = now - self._last_tick_t
            if dt < min_dt:
                return f"tick_hz throttle ({dt:.2f}s < {min_dt:.2f}s since last)"
        return None

    # ── Gating ─────────────────────────────────────────────────────
    def _gate(self, now: float) -> bool:
        cfg = self.config
        if not cfg.enabled:
            return False
        if self.bundle.get("agent_should_stop"):
            return False
        # FSM idle gate (D3).
        fsm = self.bundle.get("fsm_mode", "IDLE")
        if fsm not in ("IDLE", "ARRIVED"):
            return False
        # Queue empty gate — give the autonomy loop a tick to drain.
        if self.bundle.get("task_queue"):
            return False
        # LA-7 rotation-in-flight gate: when bundle["face_yaw_rad"] is
        # set, the autonomy loop's face handler is mid-rotation. The
        # FSM stays in IDLE during this (face short-circuits before the
        # FSM runs), but the agent should not fire another turn — it
        # has no information yet and will re-issue the same face call,
        # burning turns. Wait for the handler to clear face_yaw_rad.
        if self.bundle.get("face_yaw_rad") is not None:
            return False
        # Startup-delay gate: wait ``startup_delay_s`` from the first
        # poll we ever see so SAM3 / heightmap have a chance to fire
        # before the LLM sees an empty observation.
        if self._first_poll_t is None:
            self._first_poll_t = now
        if cfg.startup_delay_s > 0.0:
            elapsed = now - self._first_poll_t
            if elapsed < cfg.startup_delay_s:
                return False
        # tick_hz throttle.
        if cfg.tick_hz > 0.0 and self._last_tick_t is not None:
            min_dt = 1.0 / cfg.tick_hz
            if now - self._last_tick_t < min_dt:
                return False
        return True

    # ── One full turn ──────────────────────────────────────────────
    def _tick(self, now: float) -> None:
        cfg = self.config
        self._last_tick_t = now

        # D11.B — max turns reached. Don't even build the observation.
        if self.turn_count >= cfg.max_turns:
            self._force_fail(
                f"turn budget exhausted ({cfg.max_turns} turns)"
            )
            if cfg.verbose:
                print(f"[loco_x] turn {self.turn_count + 1}: FORCE-FAIL "
                      f"(max_turns={cfg.max_turns})")
            return

        # Build observation + record visited_fraction for stall watchdog.
        observation = build_observation(self.bundle, now=now)
        self._update_progress_history()
        if cfg.verbose:
            nodes = self.bundle.get("scene_nodes") or []
            labels = ", ".join(n.get("label", "?") for n in nodes[:8]) or "—"
            print(f"[loco_x] turn {self.turn_count + 1}/{cfg.max_turns}: "
                  f"observation built ({len(observation)} chars, "
                  f"{len(nodes)} scene nodes: [{labels}])")

        # Optionally annotate the observation with the stall warning.
        if self.stall_warning_active:
            observation = (
                observation
                + f"\nwarning: no exploration progress for "
                  f"{cfg.progress_stall_window_turns} turns "
                  f"(visited_fraction barely changed)"
            )

        # Query the LLM. Any client-side exception propagates as an
        # error_dict feedback rather than crashing the runner.
        messages: List[Message] = []
        if self._system_msg and self._system_msg.content:
            messages.append(self._system_msg)
        messages.append(Message(role="user", content=observation))
        try:
            response = self.client.query(messages)
        except LLMParseError as e:
            self._record_last_action(
                status="error", error_kind="parse_error",
                message=str(e),
            )
            if cfg.verbose:
                print(f"[loco_x]   LLM response parse_error: {e}")
            self.turn_count += 1
            return
        except Exception as e:                      # network / auth / ...
            self._record_last_action(
                status="error", error_kind="llm_failed",
                message=str(e),
            )
            if cfg.verbose:
                print(f"[loco_x]   LLM call failed: {type(e).__name__}: {e}")
            self.turn_count += 1
            return

        if cfg.verbose:
            # Pick the first non-comment, non-empty line so the print
            # is informative even when the LLM (or a runbook script)
            # leads with a comment header.
            preview = "(no code)"
            for ln in response.code.splitlines():
                s = ln.strip()
                if s and not s.startswith("#"):
                    preview = s
                    break
            sig = f"  signal={response.signal}" if response.signal else ""
            print(f"[loco_x]   LLM → {preview!r}{sig}")

        # Signal handling.
        if response.signal == "finish":
            # Optionally execute the code first (LLM may emit "one
            # last block then stop"). Then succeed-unwind.
            if response.code:
                self._run_code(response.code)
            self.bundle["agent_should_stop"] = True
            self.bundle["task_result_status"] = "succeeded"
            if not self.bundle.get("task_result_reason"):
                self.bundle["task_result_reason"] = (
                    "LLM emitted FINISH signal"
                )
            self.turn_count += 1
            if cfg.verbose:
                print(f"[loco_x]   FINISH — task succeeded "
                      f"after {self.turn_count} turns")
            return

        # Normal turn — execute the code (REGENERATE and plain
        # responses both run the new code).
        self._run_code(response.code)
        self.turn_count += 1
        if cfg.verbose:
            last = self.bundle.get("last_action") or {}
            status = last.get("status", "?")
            # Skill kind isn't always in the result dict (e.g.
            # ``finish()`` returns a plain ok). Fall back to the last
            # queued task's kind so the print reads as "status=ok
            # kind=goto" when a goto was enqueued this turn.
            kind = (
                last.get("kind")
                or last.get("error_kind")
                or (self.bundle.get("task_queue") or [{}])[-1].get("kind")
                or "ok"
            )
            extra = ""
            if status == "error":
                msg = last.get("message", "")
                if msg:
                    extra = f"  msg={msg!r}"
            print(f"[loco_x]   skill result: status={status} kind={kind}{extra}")

    # ── Code execution + last-action capture ──────────────────────
    def _run_code(self, code: str) -> None:
        """Run ``code`` through the sandbox; capture the result as
        ``bundle["last_action"]`` so the next observation surfaces it.

        Skills mutate ``bundle["task_queue"]`` directly. The
        last-action dict the runner stores follows the
        "probe-then-act" rule:

        * If any skill returned ``status="queued"`` or
          ``status="ok"``, that's the result we surface — the
          agent's *final* decision wins, not its first probe.
        * If every skill in the block errored, surface the first
          error so the LLM sees the most informative failure.

        Rationale: the LLM commonly writes code like
        ``for label in (...): r = find(label); if ok: goto(label);
        break; else: face(45°)``. The find() calls all error, then
        face() queues. Without the probe-then-act rule, the runner
        would report unknown_label as last_action even though the
        agent recovered correctly and a face task is now in the
        queue — a confusing signal for the next observation.
        """
        captured: Dict[str, Any] = {
            "first_error": None,
            "last_action": None,
        }
        ns = make_skills(self.bundle)

        def _wrap(name, fn):
            def _w(*args, **kwargs):
                r = fn(*args, **kwargs)
                if isinstance(r, dict):
                    status = r.get("status")
                    if status == "error":
                        if captured["first_error"] is None:
                            captured["first_error"] = r
                    elif status in ("queued", "ok"):
                        # Final acted-on skill wins.
                        captured["last_action"] = r
                return r
            return _w
        wrapped_ns = {k: _wrap(k, v) for k, v in ns.items() if callable(v)}
        # Stdlib helpers come through untouched.
        for k, v in ns.items():
            if k not in wrapped_ns:
                wrapped_ns[k] = v

        sandbox = Sandbox(
            wrapped_ns,
            timeout_s=self.config.exec_timeout_s,
        )
        try:
            sandbox.run(code)
        except SandboxRejected as e:
            self._record_last_action(
                status="error", error_kind="sandbox_rejected",
                message=str(e),
            )
            return
        except SandboxTimeout as e:
            self._record_last_action(
                status="error", error_kind="sandbox_timeout",
                message=str(e),
            )
            return
        except BaseException as e:
            # Skill raised — e.g. TypeError on bad arg type. Surface
            # cleanly rather than crash the runner.
            self._record_last_action(
                status="error", error_kind="skill_exception",
                message=f"{type(e).__name__}: {e}",
            )
            return

        # No exception — pick the captured payload to surface,
        # following the probe-then-act rule.
        if captured["last_action"] is not None:
            self.bundle["last_action"] = captured["last_action"]
        elif captured["first_error"] is not None:
            self.bundle["last_action"] = captured["first_error"]
        else:
            # Code ran but called no skills (rare); record a minimal
            # ok so the LLM doesn't see a stale prior turn.
            self._record_last_action(
                status="ok", message="code executed; no skills called"
            )

    def _record_last_action(
        self, *, status: str, error_kind: Optional[str] = None,
        message: str = "",
    ) -> None:
        last: Dict[str, Any] = {"status": status, "message": message}
        if error_kind is not None:
            last["error_kind"] = error_kind
        self.bundle["last_action"] = last

    def _force_fail(self, reason: str) -> None:
        """D11.B — runner-side fail() that doesn't go through the
        meta skill (the LLM never got a chance to call it)."""
        self.bundle["agent_should_stop"] = True
        self.bundle["task_result_status"] = "failed"
        self.bundle["task_result_reason"] = reason

    # ── D11.D progress-stall watchdog ──────────────────────────────
    def _update_progress_history(self) -> None:
        """Read current visited_fraction and update the rolling
        window. The stall warning flips on when the window is full
        and the delta from oldest to newest stays below threshold."""
        provider = self.bundle.get("occ_provider")
        if provider is None:
            return
        try:
            vf = float(provider.visited_fraction())
        except Exception:                              # pragma: no cover
            return
        self._visited_history.append(vf)
        cfg = self.config
        window = cfg.progress_stall_window_turns
        if len(self._visited_history) < window:
            self.stall_warning_active = False
            return
        # Compare oldest within the active window to newest.
        oldest = self._visited_history[-window]
        newest = self._visited_history[-1]
        self.stall_warning_active = (newest - oldest) < cfg.progress_stall_threshold


__all__ = ["AgentRunner", "RunnerConfig"]
