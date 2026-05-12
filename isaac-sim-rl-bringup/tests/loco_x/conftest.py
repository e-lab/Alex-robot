"""Make ``loco_x`` and ``autonomy`` importable from Loco-X tests.

Loco-X lives at ``<bringup>/loco_x``; the Phase 1-4 primitives live at
``<bringup>/scripts/alex_room_explore/autonomy``. Both must be reachable
on ``sys.path`` so the provider can call into the planner and so the
tests can use the Phase 1-4 rasteriser as a golden source.
"""
from __future__ import annotations

import pathlib
import sys

_BRINGUP = pathlib.Path(__file__).resolve().parents[2]
_PATHS = (
    _BRINGUP,                                      # for `import loco_x`
    _BRINGUP / "scripts" / "alex_room_explore",    # for `from autonomy import ...`
)
for p in _PATHS:
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
