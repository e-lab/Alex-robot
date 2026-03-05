"""Alex macro-action definitions for planner/controller integration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AlexMacroAction(str, Enum):
  STOP = "stop"
  WALK_STRAIGHT = "walk_straight"
  WALK_BACKWARD = "walk_backward"
  TURN_LEFT = "turn_left"
  TURN_RIGHT = "turn_right"
  STRAFE_LEFT = "strafe_left"
  STRAFE_RIGHT = "strafe_right"
  TURN_HEAD_LEFT = "turn_head_left"
  TURN_HEAD_RIGHT = "turn_head_right"


@dataclass(frozen=True)
class MacroCommand:
  action: AlexMacroAction
  lin_x: float = 0.0
  lin_y: float = 0.0
  yaw: float = 0.0
  head_yaw_deg: float = 0.0
  description: str = ""


def build_default_action_set(
  lin_speed: float = 0.6,
  yaw_speed: float = 0.8,
  head_yaw_deg: float = 20.0,
) -> dict[AlexMacroAction, MacroCommand]:
  return {
    AlexMacroAction.STOP: MacroCommand(
      action=AlexMacroAction.STOP,
      description="Stop all locomotion commands.",
    ),
    AlexMacroAction.WALK_STRAIGHT: MacroCommand(
      action=AlexMacroAction.WALK_STRAIGHT,
      lin_x=lin_speed,
      description="Walk forward.",
    ),
    AlexMacroAction.WALK_BACKWARD: MacroCommand(
      action=AlexMacroAction.WALK_BACKWARD,
      lin_x=-lin_speed,
      description="Walk backward.",
    ),
    AlexMacroAction.TURN_LEFT: MacroCommand(
      action=AlexMacroAction.TURN_LEFT,
      yaw=yaw_speed,
      description="Turn left in place.",
    ),
    AlexMacroAction.TURN_RIGHT: MacroCommand(
      action=AlexMacroAction.TURN_RIGHT,
      yaw=-yaw_speed,
      description="Turn right in place.",
    ),
    AlexMacroAction.STRAFE_LEFT: MacroCommand(
      action=AlexMacroAction.STRAFE_LEFT,
      lin_y=lin_speed,
      description="Strafe left.",
    ),
    AlexMacroAction.STRAFE_RIGHT: MacroCommand(
      action=AlexMacroAction.STRAFE_RIGHT,
      lin_y=-lin_speed,
      description="Strafe right.",
    ),
    AlexMacroAction.TURN_HEAD_LEFT: MacroCommand(
      action=AlexMacroAction.TURN_HEAD_LEFT,
      head_yaw_deg=head_yaw_deg,
      description="Rotate head left (planner-level macro).",
    ),
    AlexMacroAction.TURN_HEAD_RIGHT: MacroCommand(
      action=AlexMacroAction.TURN_HEAD_RIGHT,
      head_yaw_deg=-head_yaw_deg,
      description="Rotate head right (planner-level macro).",
    ),
  }


DEFAULT_KEY_BINDINGS: dict[int, AlexMacroAction] = {
  48: AlexMacroAction.STOP,  # 0
  49: AlexMacroAction.WALK_STRAIGHT,  # 1
  50: AlexMacroAction.WALK_BACKWARD,  # 2
  51: AlexMacroAction.TURN_LEFT,  # 3
  52: AlexMacroAction.TURN_RIGHT,  # 4
  53: AlexMacroAction.STRAFE_LEFT,  # 5
  54: AlexMacroAction.STRAFE_RIGHT,  # 6
  55: AlexMacroAction.TURN_HEAD_LEFT,  # 7
  56: AlexMacroAction.TURN_HEAD_RIGHT,  # 8
}


ALIASES: dict[str, AlexMacroAction] = {
  "walk straight": AlexMacroAction.WALK_STRAIGHT,
  "walk backward": AlexMacroAction.WALK_BACKWARD,
  "turn left": AlexMacroAction.TURN_LEFT,
  "turn right": AlexMacroAction.TURN_RIGHT,
  "strafe left": AlexMacroAction.STRAFE_LEFT,
  "strafe right": AlexMacroAction.STRAFE_RIGHT,
  "strife left": AlexMacroAction.STRAFE_LEFT,
  "strife right": AlexMacroAction.STRAFE_RIGHT,
  "turn head left": AlexMacroAction.TURN_HEAD_LEFT,
  "turn head right": AlexMacroAction.TURN_HEAD_RIGHT,
  "stop": AlexMacroAction.STOP,
}


def resolve_action_name(name: str) -> AlexMacroAction:
  key = name.strip().lower()
  if key in ALIASES:
    return ALIASES[key]
  return AlexMacroAction(key)
