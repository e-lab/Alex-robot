"""LLM brain controller for visual macro-action planning."""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

try:
  from openai import OpenAI
except Exception:  # pragma: no cover - import availability is runtime-dependent
  OpenAI = None


@dataclass(frozen=True)
class BrainDecision:
  action: str
  done: bool
  summary: str = ""


def _extract_json_object(text: str) -> dict:
  text = text.strip()
  if text.startswith("{") and text.endswith("}"):
    return json.loads(text)
  start = text.find("{")
  end = text.rfind("}")
  if start < 0 or end <= start:
    raise ValueError("No JSON object found in model output.")
  return json.loads(text[start : end + 1])


class LLMBrainController:
  """Uses an LLM to pick macro-actions from visual observations."""

  def __init__(
    self,
    goal_prompt: str,
    action_descriptions: dict[str, str],
    capture_rgb_bgr_fn: Callable[[], np.ndarray | None],
    execute_action_fn: Callable[[str], bool],
    model: str = "gpt-4.1-mini",
    max_steps: int = 30,
    step_interval_s: float = 1.5,
    api_key_env: str = "OPENAI_API_KEY",
    verbose: bool = False,
    logger: Callable[[str], None] | None = None,
  ):
    if OpenAI is None:
      raise RuntimeError(
        "openai package is not available. Install it to use llm_brain_controller."
      )
    self.goal_prompt = goal_prompt
    self.action_descriptions = action_descriptions
    self.capture_rgb_bgr_fn = capture_rgb_bgr_fn
    self.execute_action_fn = execute_action_fn
    self.model = model
    self.max_steps = max_steps
    self.step_interval_s = step_interval_s
    self.verbose = verbose
    self.logger = logger

    api_key = os.getenv(api_key_env)
    if not api_key:
      raise RuntimeError(f"Environment variable {api_key_env} is required.")
    self.client = OpenAI(api_key=api_key)

    self._step_count = 0
    self._last_step_t = 0.0
    self._active = True
    self._satisfied = False
    self._history: list[dict[str, str]] = []

  @property
  def is_active(self) -> bool:
    return self._active

  @property
  def is_satisfied(self) -> bool:
    return self._satisfied

  def _log(self, msg: str) -> None:
    if self.logger is not None:
      self.logger(msg)

  def _build_system_prompt(self) -> str:
    actions = "\n".join(
      f"- {name}: {desc}" for name, desc in sorted(self.action_descriptions.items())
    )
    return (
      "You are a robot planning controller for Alex.\n"
      "Given the robot goal and latest head RGB image, select exactly one next atomic action.\n"
      "Only choose an action from the allowed list.\n"
      "If the goal is already satisfied, set done=true and action='stop'.\n"
      "Return strict JSON only:\n"
      '{"action":"<action_name>","done":<true|false>,"summary":"<short reason>"}\n'
      f"Allowed actions:\n{actions}\n"
    )

  def _build_user_prompt(self) -> str:
    history_lines = []
    for i, item in enumerate(self._history[-8:], start=max(1, len(self._history) - 7)):
      history_lines.append(
        f"{i}. action={item['action']} done={item['done']} summary={item['summary']}"
      )
    history = "\n".join(history_lines) if history_lines else "none"
    return (
      f"Goal: {self.goal_prompt}\n"
      f"Step: {self._step_count + 1}/{self.max_steps}\n"
      f"Recent decisions:\n{history}\n"
      "Decide the next action now."
    )

  def _plan_from_image(self, rgb_bgr: np.ndarray) -> BrainDecision:
    ok, encoded = cv2.imencode(".jpg", rgb_bgr)
    if not ok:
      raise RuntimeError("Failed to encode RGB frame for LLM input.")
    b64_image = base64.b64encode(encoded.tobytes()).decode("ascii")

    system_prompt = self._build_system_prompt()
    user_prompt = self._build_user_prompt()
    if self.verbose:
      self._log("[brain][llm][request][system]")
      self._log(system_prompt)
      self._log("[brain][llm][request][user]")
      self._log(user_prompt)
      self._log(
        f"[brain][llm][request][image] jpeg_bytes={len(encoded)} base64_chars={len(b64_image)}"
      )

    response = self.client.responses.create(
      model=self.model,
      input=[
        {
          "role": "system",
          "content": [{"type": "input_text", "text": system_prompt}],
        },
        {
          "role": "user",
          "content": [
            {"type": "input_text", "text": user_prompt},
            {
              "type": "input_image",
              "image_url": f"data:image/jpeg;base64,{b64_image}",
            },
          ],
        },
      ],
    )

    text = getattr(response, "output_text", "") or ""
    if not text and hasattr(response, "output"):
      chunks = []
      for out_item in response.output:
        for c in getattr(out_item, "content", []):
          maybe_text = getattr(c, "text", None)
          if maybe_text:
            chunks.append(maybe_text)
      text = "\n".join(chunks)
    if self.verbose:
      self._log("[brain][llm][response][raw]")
      self._log(text if text else "<empty>")
    payload = _extract_json_object(text)

    action = str(payload.get("action", "stop")).strip().lower()
    done = bool(payload.get("done", False))
    summary = str(payload.get("summary", "")).strip()
    if action not in self.action_descriptions:
      action = "stop"
      done = False
      if summary:
        summary = f"{summary} | invalid action from model, defaulting to stop"
      else:
        summary = "invalid action from model, defaulting to stop"
    return BrainDecision(action=action, done=done, summary=summary)

  def tick(self, now_s: float | None = None) -> None:
    if not self._active:
      return
    t = now_s if now_s is not None else time.time()
    if (t - self._last_step_t) < self.step_interval_s:
      return
    if self._step_count >= self.max_steps:
      self._log("[brain] max steps reached, stopping.")
      self.execute_action_fn("stop")
      self._active = False
      self._satisfied = False
      return

    frame = self.capture_rgb_bgr_fn()
    if frame is None:
      self._log("[brain] no camera frame available, skipping tick.")
      return

    decision = self._plan_from_image(frame)
    self._last_step_t = t
    self._step_count += 1
    self._history.append(
      {
        "action": decision.action,
        "done": str(decision.done).lower(),
        "summary": decision.summary,
      }
    )
    self._log(
      f"[brain] step={self._step_count} action={decision.action} done={decision.done} summary={decision.summary}"
    )

    if decision.done:
      self.execute_action_fn("stop")
      self._active = False
      self._satisfied = True
      return
    self.execute_action_fn(decision.action)
