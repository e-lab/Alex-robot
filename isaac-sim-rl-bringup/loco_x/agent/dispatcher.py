"""Task-queue dispatcher (LA-6).

The bridge between LA-1 skills (which only push to
``bundle["task_queue"]``) and the existing Phase 1-4 control surface
(``goal_label``, ``goal_lock_xyz``, ``face_yaw_rad``,
``safe_stop_requested``, ``head_yaw_request``, ``head_sweep_queue``).

The autonomy loop calls ``TaskDispatcher().drain(bundle)`` once per
tick. The dispatcher translates each task to a single bundle write
and clears the queue. The Phase 1-4 autonomy loop already knows what
to do with ``goal_label`` / ``goal_lock_xyz`` / etc.; LA-6 doesn't
change that logic — it just provides the bridge.

The dispatcher is a pure function over the bundle: no Isaac, no
threading, no clock. Tests assert the bundle writes directly.
"""
from __future__ import annotations

from typing import Any, Callable, Dict


class UnknownTaskKind(ValueError):
    """A task with a ``kind`` not registered in the dispatcher.

    The skill set and the dispatcher must stay in sync; a drift here
    is a programmer bug (skill author added a kind without wiring
    dispatch) — not an LLM hallucination. The runner's sandbox layer
    already rejects unknown identifiers, so by the time a task lands
    on the queue its ``kind`` is known to the codebase.
    """


# A handler is a function (bundle, task) -> None. Handlers mutate the
# bundle in place; the dispatcher records the task in task_history
# and pops it from the queue.
Handler = Callable[[Dict[str, Any], Dict[str, Any]], None]


def _h_goto(bundle: Dict[str, Any], task: Dict[str, Any]) -> None:
    bundle["goal_label"] = task.get("label")
    xy = task.get("world_xy")
    if xy is not None:
        bundle["goal_lock_xyz"] = tuple(xy)


def _h_goto_xy(bundle: Dict[str, Any], task: Dict[str, Any]) -> None:
    bundle["goal_label"] = None
    xy = task.get("xy")
    if xy is not None:
        bundle["goal_lock_xyz"] = tuple(xy)


def _h_face(bundle: Dict[str, Any], task: Dict[str, Any]) -> None:
    bundle["face_yaw_rad"] = float(task.get("yaw_rad", 0.0))


def _h_stop(bundle: Dict[str, Any], task: Dict[str, Any]) -> None:
    bundle["safe_stop_requested"] = True


def _h_peek(bundle: Dict[str, Any], task: Dict[str, Any]) -> None:
    bundle["head_yaw_request"] = task.get("direction")


def _h_survey(bundle: Dict[str, Any], task: Dict[str, Any]) -> None:
    angles = task.get("angles_deg") or []
    bundle["head_sweep_queue"] = list(angles)


_DEFAULT_HANDLERS: Dict[str, Handler] = {
    "goto": _h_goto,
    "goto_xy": _h_goto_xy,
    "face": _h_face,
    "stop": _h_stop,
    "peek": _h_peek,
    "survey": _h_survey,
}


class TaskDispatcher:
    """Drains ``bundle["task_queue"]`` into the autonomy bundle.

    Construct with an optional ``handlers`` map to swap behaviour for
    one task kind without subclassing — useful when LA-7 sim-
    acceptance tweaks the goto path to seed a different planner.
    """

    def __init__(self, handlers: Dict[str, Handler] | None = None) -> None:
        self._handlers = dict(_DEFAULT_HANDLERS)
        if handlers:
            self._handlers.update(handlers)

    def drain(self, bundle: Dict[str, Any]) -> int:
        """Apply every queued task in FIFO order; return the number
        of tasks dispatched. The queue is emptied; ``task_history``
        (if present in the bundle) gets every task appended for
        postmortem.
        """
        queue = bundle.get("task_queue") or []
        n = 0
        for task in list(queue):                 # snapshot
            kind = task.get("kind")
            handler = self._handlers.get(kind)
            if handler is None:
                # Don't leave a half-drained queue if we hit an
                # unknown kind mid-list; clear what we've done so the
                # caller sees a clean state, then raise.
                bundle["task_queue"] = []
                raise UnknownTaskKind(
                    f"no handler for task kind: {kind!r}"
                )
            handler(bundle, task)
            history = bundle.get("task_history")
            if history is not None:
                history.append(dict(task))
            n += 1
        bundle["task_queue"] = []
        return n


__all__ = ["TaskDispatcher", "UnknownTaskKind"]
