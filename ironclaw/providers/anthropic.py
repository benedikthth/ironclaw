"""Anthropic provider adapter.

Translates the institute's neutral message/tool shape (providers/base.py) to and
from the Claude Messages API. Because reliability lives in the harness, this
adapter is deliberately thin: one non-streaming Messages call per agent turn, no
sampling parameters (removed on current models), no thinking config (omitted so
the same adapter works across Haiku/Sonnet/Opus — a cheap PhD model and an
expensive PI model use identical code).

Requires the ``anthropic`` package and an API key (``ANTHROPIC_API_KEY`` or the
``api_key`` argument).
"""

from __future__ import annotations

from typing import Any

from .base import AssistantTurn, LLMProvider, ToolCall, ToolSpec


class AnthropicProvider(LLMProvider):
    def __init__(
        self,
        model: str = "claude-opus-4-8",
        *,
        api_key: str | None = None,
        max_tokens: int = 4096,
        client: Any = None,
    ) -> None:
        self.name = f"anthropic:{model}"
        self.model = model
        self.max_tokens = max_tokens
        if client is not None:
            self._client = client
        else:
            import anthropic  # imported lazily so the core stays dependency-free

            self._client = anthropic.Anthropic(api_key=api_key)

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
    ) -> AssistantTurn:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[_to_anthropic(m) for m in messages],
            tools=[_tool_to_anthropic(t) for t in tools],
        )
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
        usage = {
            "input_tokens": getattr(resp.usage, "input_tokens", 0),
            "output_tokens": getattr(resp.usage, "output_tokens", 0),
            "model": self.model,
        }
        return AssistantTurn(text="".join(text_parts), tool_calls=calls, usage=usage)


def _tool_to_anthropic(spec: ToolSpec) -> dict[str, Any]:
    return {"name": spec.name, "description": spec.description, "input_schema": spec.input_schema}


def _to_anthropic(msg: dict[str, Any]) -> dict[str, Any]:
    """Neutral message -> Anthropic message. The only shape difference is the
    tool_result block: our neutral form carries ``output``, the API wants
    ``content``."""
    content = []
    for block in msg["content"]:
        if block.get("type") == "tool_result":
            content.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block["tool_use_id"],
                    "content": block.get("output", ""),
                    "is_error": block.get("is_error", False),
                }
            )
        else:
            content.append(block)
    return {"role": msg["role"], "content": content}
