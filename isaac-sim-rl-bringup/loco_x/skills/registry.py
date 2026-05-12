"""Skill registry — single source of truth for the LLM action space (D4).

``make_skills(bundle)`` builds the namespace the sandbox executes
against. Every name the LLM can call is here; anything not in this
dict gets rejected at AST-check time before exec runs.

The agent runner does roughly::

    ns = make_skills(bundle)
    ns.update({"range": range, "len": len, "min": min, "max": max, "print": print})
    sandbox = Sandbox(ns, timeout_s=cfg.agent.exec_timeout_s)
    sandbox.run(llm_code_block)

Skills are closures over ``bundle`` — every call mutates ``bundle``
fields (typically pushing to ``task_queue`` or flipping
``agent_should_stop``) and returns a dict in the D4 spatial-error-
context shape.

The runner *never* indexes into ``bundle`` directly; it goes through
skills. That keeps the surface auditable and makes the prompt-decision
guide (D12) authoritative.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

from .exploration import make_exploration_skills
from .locomotion import make_locomotion_skills
from .meta import make_meta_skills
from .perception import make_perception_skills


SkillRegistry = Dict[str, Callable[..., Dict[str, Any]]]


# Small stdlib subset the LLM can reach for control flow + simple
# data manipulation. Deliberately *not* including ``map``, ``filter``,
# ``open``, ``__import__`` — those either rarely come up in code-as-
# policies or carry too much escape-hatch surface.
_STDLIB_HELPERS = {
    # Control flow
    "range": range,
    "len": len,
    "min": min,
    "max": max,
    "enumerate": enumerate,
    "zip": zip,
    # I/O
    "print": print,
    # Math
    "abs": abs,
    "round": round,
    "sum": sum,
    # Data
    "sorted": sorted,
    "set": set,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    # Type checks / conversions
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "isinstance": isinstance,
}


def make_skills(
    bundle: dict, *, include_stdlib: bool = True
) -> SkillRegistry:
    """Compose the skill namespace for one agent tick.

    Each sub-module's ``make_*_skills`` returns its slice of the
    namespace; we union them. Order doesn't matter — names are
    globally unique by convention (and tested in
    ``test_registry_exposes_all_skills_in_namespace``).

    ``include_stdlib=True`` (default) adds a minimal stdlib subset
    (range / len / min / max / print / abs / round / enumerate / zip)
    so the LLM can write idiomatic loops without the runner having to
    remember to splice them in. Set ``include_stdlib=False`` for unit
    tests that want to assert the skill-only surface.
    """
    namespace: SkillRegistry = {}
    namespace.update(make_locomotion_skills(bundle))
    namespace.update(make_perception_skills(bundle))
    namespace.update(make_exploration_skills(bundle))
    namespace.update(make_meta_skills(bundle))
    if include_stdlib:
        namespace.update(_STDLIB_HELPERS)
    return namespace


__all__ = ["SkillRegistry", "make_skills"]
