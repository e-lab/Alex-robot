"""AST whitelist + wall-clock timeout sandbox for LLM-emitted Python (D1).

The agent runner (LA-5) hands each LLM response through this sandbox
before executing. The contract:

1. **AST pre-check** runs *before* ``exec``. Rejects:
   * ``import`` / ``from ... import``
   * attribute access on a dunder name (``__class__``, ``__builtins__``,
     ``__getattribute__``, ...) — the well-known Python sandbox-escape
     surface
   * calls to identifiers that aren't in the whitelisted globals dict
     (catches LLM hallucination — calling ``walk_to`` when the skill
     is ``goto_xy``)
2. **Wall-clock timeout** runs the ``exec`` on a daemon thread; a
   periodic trace function checks elapsed time and raises
   :class:`SandboxTimeout` once the budget expires. ``signal.SIGALRM``
   would be simpler but doesn't work off the main thread, and the
   agent loop deliberately runs on its own thread (D3).
3. **No ``__builtins__``**. The exec namespace contains only what the
   caller explicitly puts in ``globals_dict`` — typically the skill
   functions plus a tiny stdlib subset (``range``, ``len``, ``min``,
   ``max``, ``print``).

Returns a :class:`SandboxResult` on success; raises
:class:`SandboxRejected` (AST violation) or :class:`SandboxTimeout`
(wall-clock kill). Other exceptions from skill code propagate
unchanged — the runner catches them and feeds them back to the LLM as
``last action: error: ...``.
"""
from __future__ import annotations

import ast
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class SandboxRejected(Exception):
    """Raised when the AST pre-check finds a forbidden construct.

    The message names the violation and (when possible) the source
    line so the agent runner can render a helpful feedback string
    next turn.
    """


class SandboxTimeout(Exception):
    """Raised when ``exec`` runs past the wall-clock budget.

    Note: because the exec runs on a child thread, this exception is
    raised *inside* that thread and the main thread re-raises it
    after joining. The child thread is left to wind down (the trace
    hook stops further opcode execution), which is the standard
    Python pattern for cooperative thread cancellation.
    """


@dataclass
class SandboxResult:
    """Successful sandbox run.

    ``locals`` lets the caller inspect intermediate names (mostly
    useful for tests; the agent runner ignores it). ``duration_s`` is
    informational — the runner logs it via the existing ``Timings``
    accumulator.
    """

    locals: Dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0


# ── AST checker ─────────────────────────────────────────────────────────────
class _AstChecker(ast.NodeVisitor):
    """Walks the parsed AST and raises :class:`SandboxRejected` on the
    first violation."""

    def __init__(self, allowed_names: set[str]) -> None:
        self._allowed = allowed_names

    # ``import`` / ``from ... import``
    def visit_Import(self, node: ast.Import) -> None:
        raise SandboxRejected(f"import is forbidden (line {node.lineno})")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        raise SandboxRejected(f"import is forbidden (line {node.lineno})")

    # Dunder attribute access — well-known sandbox-escape surface.
    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.attr, str) and (
            node.attr.startswith("__") and node.attr.endswith("__")
        ):
            raise SandboxRejected(
                f"dunder attribute access is forbidden: "
                f"'{node.attr}' (line {node.lineno})"
            )
        self.generic_visit(node)

    # Unknown calls. Only Name-target calls are checked here; calls
    # against method attributes (``some_list.append(x)``) are allowed
    # — the Attribute visitor already handles dunder-attribute escapes.
    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name):
            if func.id not in self._allowed:
                raise SandboxRejected(
                    f"unknown call: '{func.id}' (line {node.lineno})"
                )
        self.generic_visit(node)


def _ast_check(source: str, allowed_names: set[str]) -> ast.Module:
    """Parse ``source`` and run the AST whitelist. Raises
    :class:`SandboxRejected` on the first violation."""
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as e:
        raise SandboxRejected(f"syntax error: {e}") from e
    _AstChecker(allowed_names).visit(tree)
    return tree


# ── Wall-clock timeout via sys.settrace ─────────────────────────────────────
class _Killer:
    """Trace function that raises :class:`SandboxTimeout` once the
    deadline passes.

    The trace hook fires on each Python bytecode "call" event. Checking
    ``time.monotonic()`` once per call is cheap and gives us
    sub-millisecond responsiveness on tight loops.
    """

    def __init__(self, deadline: float) -> None:
        self._deadline = deadline

    def __call__(self, frame, event, arg):  # noqa: ANN001
        if time.monotonic() >= self._deadline:
            raise SandboxTimeout(
                f"sandbox exceeded {self._deadline_relative_str()}"
            )
        return self

    def _deadline_relative_str(self) -> str:
        # Best-effort: the caller knows the budget; the message is for
        # the agent runner's feedback line.
        return "wall-clock budget"


# ── Sandbox ─────────────────────────────────────────────────────────────────
class Sandbox:
    """Per-tick sandbox. Construct once with the skill globals; call
    :meth:`run` for each LLM response.

    The globals dict carries everything the LLM can name. Common
    stdlib helpers (``range``, ``len``, ``min``, ``max``, ``print``)
    are *not* added automatically — callers opt in. The agent runner
    composes them in :class:`loco_x.skills.registry.SkillRegistry`.
    """

    def __init__(
        self,
        globals_dict: Dict[str, Any],
        *,
        timeout_s: float = 5.0,
    ) -> None:
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        self._globals = globals_dict
        self._timeout_s = float(timeout_s)

    def run(self, source: str) -> SandboxResult:
        """Compile, AST-check, and execute ``source``.

        Returns :class:`SandboxResult` on normal completion; raises
        :class:`SandboxRejected` for AST violations and
        :class:`SandboxTimeout` for wall-clock kills. Any other
        exception (e.g. a skill raising ``TypeError``) propagates
        unchanged — the runner catches it and feeds back the message.
        """
        allowed = set(self._globals.keys())
        _ast_check(source, allowed)

        # Run exec on a daemon thread so the trace-based timeout can
        # interrupt without blocking the calling thread.
        result_holder: Dict[str, Any] = {"locals": None, "exc": None}
        t0 = time.monotonic()
        deadline = t0 + self._timeout_s
        # Globals must not include the magic ``__builtins__`` entry
        # (Python inserts ``builtins`` by default when exec'd globals
        # has no ``__builtins__`` key — explicitly set to a tiny dict
        # so dunder-attribute access can't reach the real builtins).
        exec_globals = dict(self._globals)
        exec_globals.setdefault("__builtins__", {})
        exec_locals: Dict[str, Any] = {}

        killer = _Killer(deadline)

        def _target() -> None:
            sys.settrace(killer)
            try:
                # Compile once outside the killer's reach to avoid
                # spurious timeouts on tiny snippets.
                code = compile(source, "<sandbox>", "exec")
                exec(code, exec_globals, exec_locals)  # noqa: S102
            except SandboxTimeout as e:
                result_holder["exc"] = e
            except BaseException as e:   # propagate skill / runtime errors
                result_holder["exc"] = e
            finally:
                sys.settrace(None)
                result_holder["locals"] = exec_locals

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        thread.join(timeout=self._timeout_s + 0.5)  # extra grace for tearing down
        duration = time.monotonic() - t0

        if thread.is_alive():
            # The trace function couldn't stop the thread (e.g. a
            # blocking C call that doesn't yield to the bytecode
            # interpreter). Surface the timeout so the runner reports
            # it; the daemon thread will eventually exit at process end.
            raise SandboxTimeout(
                f"sandbox exceeded {self._timeout_s}s "
                f"(daemon thread did not unwind)"
            )

        exc = result_holder["exc"]
        if isinstance(exc, SandboxTimeout):
            raise exc
        if isinstance(exc, BaseException):
            raise exc
        return SandboxResult(
            locals=result_holder["locals"] or {},
            duration_s=duration,
        )


__all__ = [
    "Sandbox",
    "SandboxRejected",
    "SandboxResult",
    "SandboxTimeout",
]
