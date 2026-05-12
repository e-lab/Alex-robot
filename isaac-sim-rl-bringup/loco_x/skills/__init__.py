"""Loco-X skills — the LLM-callable action space (D4 + D12).

See :mod:`loco_x.skills.registry` for ``make_skills(bundle)`` — the
single entry point the agent runner uses to compose the namespace
the sandbox executes against.
"""
from .registry import SkillRegistry, make_skills

__all__ = ["SkillRegistry", "make_skills"]
