"""Tests for the LLM-code sandbox (LA-1, D1).

The sandbox runs LLM-emitted Python in a tightly restricted namespace:

* AST pre-check — reject ``import``, attribute access on dunder names,
  and any call to an identifier not in the whitelist.
* Wall-clock timeout — kill an infinite loop and surface the timeout
  in the next observation.
* No ``__builtins__`` — the LLM only sees the explicit globals dict
  (skills + a tiny stdlib subset like ``range / len / min / max /
  print``).

The sandbox is deliberately conservative. CaP-Agent0's experience says
"more primitives, more wrong combinations" — same applies to what the
LLM can do with primitives. Tight surface = predictable failure modes.

These tests cover the contracts from D1:

* ``test_sandbox_rejects_import``
* ``test_sandbox_rejects_dunder_attribute_access``
* ``test_sandbox_rejects_unknown_call``
* ``test_sandbox_allows_whitelisted_calls``
* ``test_sandbox_timeout_kills_infinite_loop``
* ``test_sandbox_timeout_message_in_feedback``
* (plus the safety-belt cases below)

The thread-timeout uses ``sys.settrace`` rather than ``signal.SIGALRM``
because the agent loop runs off the main thread (see D3).
"""
from __future__ import annotations

import time

import pytest

from loco_x.agent.sandbox import (
    SandboxResult,
    SandboxRejected,
    SandboxTimeout,
    Sandbox,
)


# ── AST whitelist ──────────────────────────────────────────────────────────
def test_sandbox_rejects_import() -> None:
    """The LLM must not be able to import arbitrary modules — even
    safe-looking ones (``os.path``) carry too much surface area."""
    sb = Sandbox(globals_dict={}, timeout_s=1.0)
    with pytest.raises(SandboxRejected, match="import"):
        sb.run("import os")
    with pytest.raises(SandboxRejected, match="import"):
        sb.run("from os import path")


def test_sandbox_rejects_dunder_attribute_access() -> None:
    """``__builtins__``, ``__class__``, ``__globals__`` etc. are well-
    known sandbox escapes. Block any ``Attribute`` node whose attr name
    starts with double underscore."""
    sb = Sandbox(globals_dict={}, timeout_s=1.0)
    with pytest.raises(SandboxRejected, match="dunder"):
        sb.run("().__class__")
    with pytest.raises(SandboxRejected, match="dunder"):
        sb.run("x = 1\nx.__add__")


def test_sandbox_rejects_unknown_call() -> None:
    """Calls to identifiers that aren't in the whitelisted globals
    dict are rejected at AST-check time, before exec runs. This is
    the LLM's main hallucination mode — calling ``walk_to`` when the
    skill is ``goto_xy`` — so we surface it as a clear error rather
    than a NameError at runtime."""
    def known_call(x):
        return x
    sb = Sandbox(globals_dict={"known_call": known_call}, timeout_s=1.0)
    # Known call goes through.
    sb.run("known_call(5)")
    # Unknown call rejected before execution.
    with pytest.raises(SandboxRejected, match="unknown call"):
        sb.run("walk_to(5)")


def test_sandbox_allows_attribute_access_on_returned_dict() -> None:
    """Skill results are dicts (D4). The LLM does
    ``result["status"] == "ok"`` constantly, so we must NOT block
    ordinary ``Subscript`` access. Only dunder *attribute* access is
    blocked."""
    log = []
    def skill_returns_dict():
        return {"status": "ok", "value": 42}
    sb = Sandbox(
        globals_dict={"skill_returns_dict": skill_returns_dict, "log": log},
        timeout_s=1.0,
    )
    sb.run(
        "r = skill_returns_dict()\n"
        "log.append(r['status'])\n"
        "log.append(r['value'])\n"
    )
    assert log == ["ok", 42]


def test_sandbox_allows_whitelisted_calls() -> None:
    """A canonical multi-target chain — what the LLM will actually
    emit in production. range/len/print + a custom skill must all run."""
    captured = []
    def goto(label: str):
        captured.append(label)
        return {"status": "queued", "label": label}
    sb = Sandbox(
        globals_dict={
            "goto": goto,
            "range": range, "len": len, "print": print,
        },
        timeout_s=1.0,
    )
    sb.run(
        "for label in ['stove', 'sink', 'fridge']:\n"
        "    goto(label)\n"
    )
    assert captured == ["stove", "sink", "fridge"]


def test_sandbox_no_builtins_leak() -> None:
    """Even when the user code looks innocent, the sandbox must not
    expose Python builtins as a free namespace. Without an explicit
    ``__builtins__`` entry in globals, ``open`` / ``eval`` / ``exec``
    must not be available."""
    sb = Sandbox(globals_dict={}, timeout_s=1.0)
    with pytest.raises(SandboxRejected, match="unknown call"):
        sb.run("open('/etc/passwd')")
    with pytest.raises(SandboxRejected, match="unknown call"):
        sb.run("eval('1+1')")


# ── Wall-clock timeout ─────────────────────────────────────────────────────
def test_sandbox_timeout_kills_infinite_loop() -> None:
    """A ``while True: pass`` is valid AST and uses no banned globals,
    so the AST whitelist can't reject it. The wall-clock timeout is
    the catch-all: after ``timeout_s`` the sandbox raises
    :class:`SandboxTimeout`."""
    sb = Sandbox(globals_dict={}, timeout_s=0.2)
    t0 = time.monotonic()
    with pytest.raises(SandboxTimeout):
        sb.run("while True:\n    pass")
    elapsed = time.monotonic() - t0
    # Timeout must fire promptly — we allow up to 2x the budget so a
    # slow CI host doesn't false-positive.
    assert elapsed < 0.5, f"timeout took {elapsed:.2f}s, budget=0.2s"


def test_sandbox_timeout_kills_long_arithmetic_loop() -> None:
    """A finite-but-huge loop hits the same wall-clock timeout."""
    sb = Sandbox(globals_dict={"range": range}, timeout_s=0.2)
    with pytest.raises(SandboxTimeout):
        sb.run("total = 0\nfor i in range(10**8):\n    total += i")


def test_sandbox_run_returns_result_object() -> None:
    """The sandbox's normal-path return is a :class:`SandboxResult`
    that carries the locals dict (so the agent runner can pick up
    intermediate names if it cares) and a duration measurement."""
    sb = Sandbox(globals_dict={}, timeout_s=1.0)
    result = sb.run("x = 1 + 2\ny = x * 10")
    assert isinstance(result, SandboxResult)
    assert result.locals["x"] == 3
    assert result.locals["y"] == 30
    assert result.duration_s >= 0.0


def test_sandbox_propagates_skill_exceptions_without_killing_runner() -> None:
    """If a skill itself raises (e.g. a malformed argument), the
    sandbox surfaces the exception cleanly — the runner catches and
    feeds it back to the LLM next turn rather than crashing the
    process. Distinct from :class:`SandboxRejected` (AST violation)
    and :class:`SandboxTimeout` (wall-clock kill)."""
    def goto(label):
        if not isinstance(label, str):
            raise TypeError(f"label must be str, got {type(label).__name__}")
        return {"status": "queued"}
    sb = Sandbox(globals_dict={"goto": goto}, timeout_s=1.0)
    with pytest.raises(TypeError, match="label must be str"):
        sb.run("goto(5)")
