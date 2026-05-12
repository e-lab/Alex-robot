"""Meta skills: finish, fail (D11).

Both set ``bundle["agent_should_stop"] = True`` and record the task
result. The agent runner picks this up after the sandbox returns, then
unwinds per D11:

* prints a one-line summary ([agent] FAILED: ... after N turns ...),
* forces ``_cmd`` to the safe-stop (0, 0, 0, 1),
* preserves FSM mode for postmortem (does NOT reset),
* exits the agent thread cleanly so the sim keeps stepping.

LA-1 only handles the *call*; the runner's unwinding logic lands in
LA-5.
"""
from __future__ import annotations

from typing import Any, Callable, Dict

from ._errors import ok_dict


def make_meta_skills(bundle: dict) -> Dict[str, Callable[..., Dict[str, Any]]]:
    """Build the two meta-skill closures for one tick."""

    def finish(message: str = "") -> Dict[str, Any]:
        """Mark the agent's task as succeeded and stop the loop.

        The autonomy loop will continue running (so the sim doesn't
        freeze and atexit hooks fire), but the agent thread exits.
        """
        bundle["agent_should_stop"] = True
        bundle["task_result_status"] = "succeeded"
        bundle["task_result_reason"] = str(message)
        return ok_dict(value=None, message=str(message))

    def fail(reason: str = "") -> Dict[str, Any]:
        """Mark the agent's task as failed and stop the loop.

        Per D11: ``_cmd`` is forced to the safe-stop in the autonomy
        unwinding step (LA-5); FSM mode is preserved; the sim keeps
        running. Multi-target chains skip remaining legs silently —
        the LLM is free to encode fallbacks explicitly in its code
        block, but the runner never auto-retargets.
        """
        bundle["agent_should_stop"] = True
        bundle["task_result_status"] = "failed"
        bundle["task_result_reason"] = str(reason)
        return ok_dict(value=None, reason=str(reason))

    return {
        "finish": finish,
        "fail": fail,
    }


__all__ = ["make_meta_skills"]
