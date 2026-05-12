"""Tests for LLM-response parsing (LA-4).

The agent runner sees one string per turn from whichever LLM backend
is in use; the parser turns that into ``(code_str | None, signal)``
where ``signal`` is one of ``"finish"``, ``"regenerate"``, or ``None``
(plain code response).

Format conventions match :mod:`capx.llm.client` from cap-x:

* fenced Python: ``\\`\\`\\`python\\n<code>\\n\\`\\`\\``` — the most common case
* synthesis flow: literal ``FINISH`` on its own line means the LLM
  declares the task complete
* synthesis flow: literal ``REGENERATE`` followed by another fenced
  block means "try again with this revised code"
* no code, no signal → parser raises so the runner can feed the
  garbled response back as a feedback line
"""
from __future__ import annotations

import pytest

from loco_x.llm.parsers import (
    LLMResponse,
    LLMParseError,
    parse_response,
)


# ── Canonical fenced-block cases ────────────────────────────────────────────
def test_parse_python_fenced_block_returns_code() -> None:
    """The single-fenced-block case the runner sees every turn."""
    text = "Here's my next step:\n\n```python\ngoto('stove')\n```\n"
    r = parse_response(text)
    assert r.code.strip() == "goto('stove')"
    assert r.signal is None


def test_parse_handles_bare_backtick_fence() -> None:
    """LLMs sometimes emit \\`\\`\\` without the ``python`` tag. Same
    treatment — extract the inner code."""
    text = "```\nfor label in ['a', 'b']:\n    goto(label)\n```"
    r = parse_response(text)
    assert "goto(label)" in r.code
    assert r.signal is None


def test_parse_takes_first_block_when_multiple_present() -> None:
    """A well-behaved LLM emits exactly one fenced block per turn,
    but some emit two with the second being explanatory pseudo-code.
    Take the first; the runner echoes a warning in the next
    observation."""
    text = (
        "```python\ngoto('stove')\n```\n"
        "Plus, for reference:\n"
        "```python\n# this is just an example\n```\n"
    )
    r = parse_response(text)
    assert r.code.strip() == "goto('stove')"


# ── FINISH / REGENERATE signals ────────────────────────────────────────────
def test_parse_finish_signal_alone() -> None:
    """``FINISH`` on its own line, no code: signal=='finish'."""
    text = "Task complete.\n\nFINISH\n"
    r = parse_response(text)
    assert r.signal == "finish"
    assert r.code == ""


def test_parse_regenerate_with_code() -> None:
    """``REGENERATE`` followed by a fenced block: signal=='regenerate',
    code is the new block."""
    text = (
        "Reconsidering — the previous goto missed because the label "
        "was misspelled.\n\n"
        "REGENERATE\n"
        "```python\ngoto('microwave')\n```\n"
    )
    r = parse_response(text)
    assert r.signal == "regenerate"
    assert r.code.strip() == "goto('microwave')"


def test_parse_regenerate_lookalike_in_prose_is_not_a_signal() -> None:
    """``regenerate`` mentioned in lowercase as part of a sentence
    must not trigger the signal. Match the literal uppercase token on
    its own line."""
    text = (
        "I'd suggest we regenerate the path next turn.\n\n"
        "```python\nstop()\n```\n"
    )
    r = parse_response(text)
    assert r.signal is None
    assert r.code.strip() == "stop()"


# ── Error path ─────────────────────────────────────────────────────────────
def test_parse_rejects_response_with_no_code_block_and_no_signal() -> None:
    """A response with neither a code block nor a recognized signal
    is unusable. Raise :class:`LLMParseError` so the runner echoes a
    feedback line and asks the LLM to retry."""
    text = "I'm thinking about what to do next."
    with pytest.raises(LLMParseError):
        parse_response(text)


def test_parse_empty_string_rejected() -> None:
    """Empty response → parse error. Defensive — happens when an
    LLM call times out without a body."""
    with pytest.raises(LLMParseError):
        parse_response("")


# ── Code with FINISH inside the fenced block is still code ──────────────────
def test_parse_finish_inside_code_block_is_not_signal() -> None:
    """If ``FINISH`` appears inside a fenced ```python``` block it's
    just code text (probably ``# FINISH`` comment). Don't trigger
    the signal — only bare-line FINISH outside any block matters."""
    text = "```python\nfinish('done')\n# REGENERATE later\n```\n"
    r = parse_response(text)
    assert r.signal is None
    assert "finish('done')" in r.code
