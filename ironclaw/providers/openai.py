"""OpenAI-compatible provider — covers OpenAI, OpenRouter, and local servers.

OpenAI, OpenRouter, and every common self-hosted runtime (Ollama, vLLM, LM
Studio, llama.cpp's server) speak the same Chat Completions API, differing only
by ``base_url`` and key. So one adapter, pointed at different endpoints, covers
all three "providers" — the registry supplies the endpoint.

Translates the institute's neutral message/tool shape (providers/base.py) to and
from Chat Completions, whose tool protocol differs from Anthropic's: tool results
are their own ``role: "tool"`` messages rather than blocks inside a user turn.
The client is constructed lazily, so importing this module (and routing through
the registry) never requires the ``openai`` package until a real call is made.
"""

from __future__ import annotations

import json
from typing import Any

from .base import AssistantTurn, LLMProvider, ToolCall, ToolSpec


class OpenAICompatibleProvider(LLMProvider):
    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 4096,
        default_headers: dict | None = None,
        name: str | None = None,
        client: Any = None,
    ) -> None:
        self.name = name or "openai"
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self._api_key = api_key
        self._default_headers = default_headers
        self._client = client  # injected in tests; lazily built otherwise

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI  # lazy: no dependency until a real call

            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self._api_key or "not-needed",  # local servers ignore it
                default_headers=self._default_headers,
            )
        return self._client

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
    ) -> AssistantTurn:
        resp = self._get_client().chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=_to_openai_messages(system, messages),
            tools=[_tool_to_openai(t) for t in tools] or None,
        )
        msg = resp.choices[0].message
        calls: list[ToolCall] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            args = tc.function.arguments or "{}"
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=json.loads(args)))
        usage = None
        if getattr(resp, "usage", None) is not None:
            usage = {
                "input_tokens": getattr(resp.usage, "prompt_tokens", 0),
                "output_tokens": getattr(resp.usage, "completion_tokens", 0),
                "model": self.model,
            }
        return AssistantTurn(text=msg.content or "", tool_calls=calls, usage=usage)


def _tool_to_openai(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": spec.name, "description": spec.description, "parameters": spec.input_schema},
    }


def _to_openai_messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Neutral messages -> Chat Completions messages. The key shape difference:
    a neutral ``tool_result`` block (which we carry inside a user turn) becomes a
    standalone ``role: "tool"`` message keyed by the tool call id."""
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in m["content"]:
            t = block.get("type")
            if t == "text":
                text_parts.append(block["text"])
            elif t == "tool_use":
                tool_calls.append(
                    {
                        "id": block["id"],
                        "type": "function",
                        "function": {"name": block["name"], "arguments": json.dumps(block.get("input", {}))},
                    }
                )
            elif t == "tool_result":
                out.append(
                    {"role": "tool", "tool_call_id": block["tool_use_id"], "content": block.get("output", "")}
                )
        if m["role"] == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts) or None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            out.append(msg)
        elif m["role"] == "user" and text_parts:
            out.append({"role": "user", "content": "".join(text_parts)})
        elif m["role"] == "system" and text_parts:  # mid-conversation operator note
            out.append({"role": "system", "content": "".join(text_parts)})
    return out
