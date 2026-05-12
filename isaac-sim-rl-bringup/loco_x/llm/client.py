"""LLM client backends — text + multimodal (D6, D12).

Three concrete clients implementing :class:`LLMClient` Protocol:

* :class:`ScriptedClient` — plays back a queue of canned responses.
  The LA-5 runner tests will lean on it; also serves as a no-network
  baseline.
* :class:`StdinClient` — reads a multi-line response from a file
  handle (defaults to ``sys.stdin``), bounded by a single-line
  ``EOF`` sentinel. Used for offline demos and CI integration tests.
* :class:`AnthropicClient` — live network via ``httpx.post``. The
  multimodal path attaches an image content block (PNG bytes) so
  ``describe_view`` returns a Claude caption.

The runner uses the same ``query() / query_multimodal()`` surface
regardless of backend; swapping clients is a Hydra-config flip
(``agent=stdin | anthropic | openrouter``).
"""
from __future__ import annotations

import base64
import io
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import httpx

from .parsers import LLMResponse, parse_response


# ── Message dataclass ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class Message:
    """One turn in an LLM conversation. ``role`` follows the standard
    Anthropic / OpenAI convention (``user``, ``assistant``, ``system``);
    ``content`` is plain text."""

    role: str
    content: str


# ── Protocol ───────────────────────────────────────────────────────────────
@runtime_checkable
class LLMClient(Protocol):
    """The surface every backend exposes.

    ``query`` returns the parsed :class:`LLMResponse`. ``query_multimodal``
    returns a plain caption string — used only by the D12
    ``describe_view`` skill, which doesn't need code-block parsing.
    """

    def query(self, messages: List[Message]) -> LLMResponse: ...

    def query_multimodal(
        self,
        *,
        messages: List[Message],
        image_bytes: bytes,
    ) -> str: ...


# ── ScriptedClient (test double + offline) ─────────────────────────────────
class ScriptedClient:
    """Plays back a queue of canned responses.

    Two queues — ``responses`` for :meth:`query` and
    ``multimodal_responses`` for :meth:`query_multimodal` — so a
    single client can drive a mixed-modality test. Asking for an
    item past the end of the queue raises ``IndexError`` so tests
    fail loudly when they're under-fed.
    """

    def __init__(
        self,
        responses: List[str],
        *,
        multimodal_responses: Optional[List[str]] = None,
    ) -> None:
        self._responses = list(responses)
        self._multimodal = list(multimodal_responses or [])
        self._idx = 0
        self._mm_idx = 0

    def query(self, messages: List[Message]) -> LLMResponse:
        if self._idx >= len(self._responses):
            raise IndexError(
                f"scripted client exhausted at turn {self._idx + 1}; "
                f"only {len(self._responses)} canned responses"
            )
        raw = self._responses[self._idx]
        self._idx += 1
        return parse_response(raw)

    def query_multimodal(
        self, *, messages: List[Message], image_bytes: bytes,
    ) -> str:
        if self._mm_idx >= len(self._multimodal):
            raise IndexError(
                f"scripted multimodal exhausted at turn {self._mm_idx + 1}; "
                f"only {len(self._multimodal)} canned captions"
            )
        caption = self._multimodal[self._mm_idx]
        self._mm_idx += 1
        return caption


# ── StdinClient ────────────────────────────────────────────────────────────
class StdinClient:
    """Reads responses from a file handle.

    Each ``query()`` reads lines until it sees a bare ``EOF`` on its
    own line; the lines before that are the response. Lets a human
    drive the agent interactively, or a CI script pipe in canned
    responses via shell redirect.
    """

    def __init__(
        self,
        *,
        stream=None,                          # noqa: ANN001
        prompt_label: str = "[llm-stdin]",
    ) -> None:
        self._stream = stream if stream is not None else sys.stdin
        self._prompt = prompt_label

    def _read_until_eof(self) -> str:
        lines: List[str] = []
        for line in self._stream:
            if line.rstrip() == "EOF":
                break
            lines.append(line)
        return "".join(lines)

    def query(self, messages: List[Message]) -> LLMResponse:
        raw = self._read_until_eof()
        return parse_response(raw)

    def query_multimodal(
        self, *, messages: List[Message], image_bytes: bytes,
    ) -> str:
        # No image — just prompt the human for a caption.
        return self._read_until_eof().strip()


# ── AnthropicClient (live network, mockable) ───────────────────────────────
class AnthropicClient:
    """Anthropic Messages API client.

    The API expects ``messages: [{role, content}]`` for plain text and
    ``content: [{type: text|image, ...}]`` lists for multimodal. We
    keep both paths in one class so the agent runner doesn't have to
    pick between text- and image-aware clients.
    """

    API_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        multimodal_model: Optional[str] = None,
        timeout_s: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._system = system_prompt
        self._max_tokens = int(max_tokens)
        # D12: VLM captioning often uses a cheaper / faster model
        # (Sonnet) than the planning LLM (Opus). Allow distinct
        # selection; default to the same model.
        self._multimodal_model = multimodal_model or model
        self._timeout_s = float(timeout_s)

    def _headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": self.API_VERSION,
            "content-type": "application/json",
        }

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        resp = httpx.post(
            self.API_URL,
            json=payload,
            headers=self._headers(),
            timeout=self._timeout_s,
        )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _text_from_response(body: Dict[str, Any]) -> str:
        for part in body.get("content", []):
            if part.get("type") == "text":
                return part.get("text", "")
        return ""

    def query(self, messages: List[Message]) -> LLMResponse:
        payload: Dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if self._system:
            payload["system"] = self._system
        body = self._post(payload)
        text = self._text_from_response(body)
        return parse_response(text)

    def query_multimodal(
        self,
        *,
        messages: List[Message],
        image_bytes: bytes,
    ) -> str:
        # The Messages API takes a list of content blocks for image
        # inputs; we put one image block + one text block per the
        # supplied message. Anthropic accepts base64-encoded PNGs.
        b64 = base64.b64encode(image_bytes).decode("ascii")
        # Use the latest message's text as the user prompt; we don't
        # currently support multi-image multi-turn (rare for VLM
        # captioning).
        prompt = messages[-1].content if messages else "describe this view"
        payload: Dict[str, Any] = {
            "model": self._multimodal_model,
            "max_tokens": self._max_tokens,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        }
        if self._system:
            payload["system"] = self._system
        body = self._post(payload)
        return self._text_from_response(body).strip()


__all__ = [
    "AnthropicClient",
    "LLMClient",
    "Message",
    "ScriptedClient",
    "StdinClient",
]
