"""Tests for the LLM client backends (LA-4).

Three concrete clients implement the same Protocol:

* :class:`ScriptedClient` — canned responses, used by LA-5 runner
  tests and as the offline-demo workhorse,
* :class:`StdinClient` — reads from a file/stream so a human (or a
  pipe in CI) can drive the agent,
* :class:`AnthropicClient` — live network. Tested with a mocked
  ``httpx`` client so unit tests don't burn credits or require a key.

The :meth:`query` method takes a list of messages and returns
``(code, signal)``; :meth:`query_multimodal` adds an image and
returns a plain caption (D12 ``describe_view``).
"""
from __future__ import annotations

import io
import os
from typing import Any, Dict, List, Optional

import pytest

from loco_x.llm.client import (
    AnthropicClient,
    LLMClient,
    Message,
    ScriptedClient,
    StdinClient,
)


# ── Scripted backend ───────────────────────────────────────────────────────
def test_scripted_client_returns_canned_blocks() -> None:
    """The :class:`ScriptedClient` plays back a queue of canned
    responses — what the runner tests in LA-5 will lean on."""
    canned = [
        "```python\ngoto('stove')\n```",
        "```python\ngoto('sink')\n```",
        "FINISH\n",
    ]
    client = ScriptedClient(responses=canned)
    r1 = client.query([Message(role="user", content="turn 1")])
    assert r1.code.strip() == "goto('stove')"
    assert r1.signal is None

    r2 = client.query([Message(role="user", content="turn 2")])
    assert "goto('sink')" in r2.code

    r3 = client.query([Message(role="user", content="turn 3")])
    assert r3.signal == "finish"


def test_scripted_client_raises_when_queue_exhausted() -> None:
    """If the test feeds N responses and the runner asks N+1 times,
    surface a clear error rather than silently returning empty."""
    client = ScriptedClient(responses=["FINISH"])
    client.query([Message(role="user", content="turn 1")])
    with pytest.raises(IndexError, match="scripted"):
        client.query([Message(role="user", content="turn 2")])


def test_scripted_multimodal_returns_canned_caption() -> None:
    """For LA-1's ``describe_view`` stub the runner will be replaced
    in LA-5 by a real VLM call. The scripted client supports the
    multimodal path with its own queue."""
    client = ScriptedClient(
        responses=[],
        multimodal_responses=["a kitchen scene with a stove and a fridge"],
    )
    caption = client.query_multimodal(
        messages=[Message(role="user", content="describe")],
        image_bytes=b"\x00\x01\x02",
    )
    assert "kitchen" in caption


# ── Stdin backend ──────────────────────────────────────────────────────────
def test_stdin_client_reads_one_response_from_file_handle() -> None:
    """``StdinClient(stream=...)`` reads one response per ``query()``.

    Boundary marker is a single-line ``EOF`` (matching cap-x's
    convention for the offline mode) so multi-line code blocks fit
    in one read.
    """
    sample = (
        "Here we go:\n"
        "```python\n"
        "goto('stove')\n"
        "```\n"
        "EOF\n"
    )
    stream = io.StringIO(sample)
    client = StdinClient(stream=stream)
    r = client.query([Message(role="user", content="turn")])
    assert r.code.strip() == "goto('stove')"
    assert r.signal is None


def test_stdin_client_multimodal_prompts_and_reads_caption() -> None:
    """The multimodal path in stdin mode just asks the human to type
    a caption. Useful for offline tests of ``describe_view`` flow
    without a VLM."""
    sample = "a kitchen with a stove and a fridge\nEOF\n"
    stream = io.StringIO(sample)
    client = StdinClient(stream=stream)
    caption = client.query_multimodal(
        messages=[Message(role="user", content="describe")],
        image_bytes=b"\x00\x01\x02",
    )
    assert "stove" in caption


# ── Anthropic backend (mocked) ─────────────────────────────────────────────
def test_anthropic_client_sends_system_prompt_and_messages(monkeypatch) -> None:
    """``AnthropicClient`` posts to the Messages API with the system
    prompt and message list. We mock ``httpx.post`` so the test
    doesn't hit the network."""
    captured: Dict[str, Any] = {}

    class _FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return {
                "content": [{
                    "type": "text",
                    "text": "```python\ngoto('stove')\n```",
                }],
                "stop_reason": "end_turn",
            }

        def raise_for_status(self):
            pass

    def _fake_post(url, *, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return _FakeResponse()

    import loco_x.llm.client as client_mod
    monkeypatch.setattr(client_mod.httpx, "post", _fake_post)

    client = AnthropicClient(
        api_key="test-key",
        model="claude-opus-4-7",
        system_prompt="You are a helpful robot.",
    )
    r = client.query([Message(role="user", content="please walk to stove")])
    assert r.code.strip() == "goto('stove')"
    # Verified: the request carried system + messages.
    assert captured["json"]["system"] == "You are a helpful robot."
    assert captured["json"]["model"] == "claude-opus-4-7"
    assert captured["json"]["messages"][0]["role"] == "user"
    assert "x-api-key" in captured["headers"]


def test_anthropic_multimodal_query_sends_image_block(monkeypatch) -> None:
    """For ``query_multimodal``, the request must carry an image
    content block alongside the text. The Anthropic Messages API
    accepts ``{"type": "image", "source": {...}}`` parts."""
    captured: Dict[str, Any] = {}

    class _FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"content": [{"type": "text", "text": "a kitchen"}]}

        def raise_for_status(self):
            pass

    def _fake_post(url, *, json, headers, timeout):
        captured["json"] = json
        return _FakeResponse()

    import loco_x.llm.client as client_mod
    monkeypatch.setattr(client_mod.httpx, "post", _fake_post)

    client = AnthropicClient(
        api_key="test-key",
        model="claude-sonnet-4-6",
        system_prompt=None,
        multimodal_model="claude-sonnet-4-6",
    )
    caption = client.query_multimodal(
        messages=[Message(role="user", content="describe this")],
        image_bytes=b"\x89PNG\x0d\x0a\x1a\x0a",
    )
    assert "kitchen" in caption
    # The message must contain BOTH a text and an image content block.
    parts = captured["json"]["messages"][0]["content"]
    types = {p["type"] for p in parts}
    assert types == {"text", "image"}, f"got {types}"


def test_anthropic_client_propagates_http_error(monkeypatch) -> None:
    """A non-2xx response surfaces as an exception so the runner
    catches and falls back to ``error[llm_failed]`` in the next
    observation."""
    class _FakeError(Exception):
        pass

    class _FakeResponse:
        status_code = 500

        @staticmethod
        def json():
            return {"error": {"message": "server error"}}

        def raise_for_status(self):
            raise _FakeError("500 server error")

    import loco_x.llm.client as client_mod
    monkeypatch.setattr(
        client_mod.httpx, "post",
        lambda *a, **k: _FakeResponse(),
    )

    client = AnthropicClient(
        api_key="test-key", model="m", system_prompt=None,
    )
    with pytest.raises(_FakeError):
        client.query([Message(role="user", content="hi")])


# ── Protocol satisfaction ──────────────────────────────────────────────────
def test_all_backends_satisfy_llmclient_protocol() -> None:
    """:class:`LLMClient` is a Protocol; runtime check confirms each
    backend exposes the same surface so the agent runner is
    backend-agnostic."""
    sc = ScriptedClient(responses=["FINISH"])
    st = StdinClient(stream=io.StringIO("FINISH\nEOF\n"))
    an = AnthropicClient(api_key="x", model="m", system_prompt=None)
    for c in (sc, st, an):
        assert isinstance(c, LLMClient)
