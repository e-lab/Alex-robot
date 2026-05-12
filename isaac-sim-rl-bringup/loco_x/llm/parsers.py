"""Parse LLM responses into (code, signal) pairs.

The agent runner gets one raw string per turn from whichever backend
is in use. This module's job is to pull a Python code block out of
the message and detect the two optional control signals used by
CaP-Agent0's synthesis flow.

Conventions match :mod:`capx.llm.client`:

* Fenced Python block (triple-backtick ``python`` ... triple-backtick) — the most common
  case. We accept both the ``python``-tagged and bare-fence forms.
* ``FINISH`` on its own line, outside any code block — the LLM
  declares the task complete. Returned as ``signal="finish"``,
  ``code=""``.
* ``REGENERATE`` on its own line, followed by a fenced block — the
  LLM revises a prior decision. Returned as ``signal="regenerate"``
  with the new code.
* No code and no signal → :class:`LLMParseError`. The runner echoes
  a feedback line and asks the LLM to retry on the next turn.

The parser does **not** interpret the code — that's the sandbox's
job (LA-1). It also doesn't try to repair malformed responses; if
the format drifts, we'd rather hear about it loudly than guess.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional

# Matches both ```python\n...\n``` and ```\n...\n``` (bare-fence).
# DOTALL so ``.`` matches newlines inside the block.
_FENCED_BLOCK_RE = re.compile(
    r"```(?:python)?\s*\n(?P<code>.*?)```",
    re.DOTALL,
)

# Stand-alone signal tokens. Anchored to their own line so prose
# mentions ("we should regenerate") don't trigger.
_FINISH_LINE_RE = re.compile(r"(?m)^\s*FINISH\s*$")
_REGENERATE_LINE_RE = re.compile(r"(?m)^\s*REGENERATE\s*$")


Signal = Optional[Literal["finish", "regenerate"]]


@dataclass(frozen=True)
class LLMResponse:
    """Parsed LLM turn.

    ``code`` is the Python source to run (may be empty when
    ``signal=='finish'``). ``signal`` is ``"finish"``, ``"regenerate"``,
    or ``None`` for an ordinary code response.
    """

    code: str
    signal: Signal


class LLMParseError(ValueError):
    """Raised when an LLM response has neither a fenced code block
    nor a recognized signal — usable as a feedback line directly."""


def _strip_code_blocks(text: str) -> str:
    """Return ``text`` with all fenced blocks replaced by ``__BLOCK__``
    placeholders so signal-matching only sees prose."""
    return _FENCED_BLOCK_RE.sub("__BLOCK__", text)


def parse_response(text: str) -> LLMResponse:
    """Pull (code, signal) out of an LLM response string."""
    if not text or not text.strip():
        raise LLMParseError("empty response")

    blocks = list(_FENCED_BLOCK_RE.finditer(text))
    outside_blocks = _strip_code_blocks(text)

    has_finish = bool(_FINISH_LINE_RE.search(outside_blocks))
    has_regenerate = bool(_REGENERATE_LINE_RE.search(outside_blocks))

    # FINISH alone — task complete, no code expected.
    if has_finish and not blocks:
        return LLMResponse(code="", signal="finish")

    # REGENERATE followed by a fenced block.
    if has_regenerate and blocks:
        return LLMResponse(
            code=blocks[0].group("code").strip(),
            signal="regenerate",
        )

    # FINISH together with a fenced block — uncommon but well-defined:
    # the LLM wants to execute one last block then stop. Treat as
    # signal=finish so the runner can run the block and unwind.
    if has_finish and blocks:
        return LLMResponse(
            code=blocks[0].group("code").strip(),
            signal="finish",
        )

    # Plain code response (the most common case).
    if blocks:
        return LLMResponse(code=blocks[0].group("code").strip(), signal=None)

    raise LLMParseError(
        "response has neither a fenced code block nor a recognized "
        "FINISH / REGENERATE signal"
    )


__all__ = ["LLMParseError", "LLMResponse", "Signal", "parse_response"]
