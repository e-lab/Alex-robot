"""pytest bootstrap for the scene_graph test suite.

This file runs before any test is collected. It adds `isaac-sim-rl-bringup/`
to `sys.path` so `import scene_graph` works regardless of where pytest is
invoked from.

Run unit tests with (IsaacLab's bundled Kit python):
    cd ~/alex/repository-group/IsaacLab
    ./isaaclab.sh -p -m pytest /pathtoFolder/Alex-robot/isaac-sim-rl-bringup/tests/unit -q

Or, if you have numpy + pytest in a plain venv, from the bringup directory:
    python -m pytest tests/unit -q

The `replay` and `scenario` test folders need more setup (a recorded
hallway fixture and a running Isaac Sim respectively); see their READMEs.
"""

from __future__ import annotations
import sys
import pathlib

_BRINGUP_ROOT = pathlib.Path(__file__).resolve().parents[1]  # .../isaac-sim-rl-bringup/
if str(_BRINGUP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BRINGUP_ROOT))
