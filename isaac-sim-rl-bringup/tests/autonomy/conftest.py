"""Make the alex_room_explore/autonomy package importable from tests."""
import pathlib
import sys

# Resolve: tests/autonomy/ -> tests/ -> isaac-sim-rl-bringup/ -> ...
_BRINGUP = pathlib.Path(__file__).resolve().parents[2]
_AUTONOMY_PARENT = _BRINGUP / "scripts" / "alex_room_explore"

if str(_AUTONOMY_PARENT) not in sys.path:
    sys.path.insert(0, str(_AUTONOMY_PARENT))
