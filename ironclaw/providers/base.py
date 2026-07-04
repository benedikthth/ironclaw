"""Provider-neutral LLM interface.

The whole institute must run against Anthropic, OpenAI, OpenRouter, or a
self-hosted endpoint interchangeably, and the harness must carry reliability so
a *cheap* model can drive a PhD. So the runtime speaks this neutral shape and
each provider adapter translates to/from its wire format.

Messages are kept in a neutral list of dicts to stay JSON-serializable for
durable state (a paused PhD waiting on a slurm job must be reloadable).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]  # JSON Schema for the tool's arguments


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class AssistantTurn:
    """One model response: free text plus zero or more tool calls."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Optional per-turn token usage for the observability/cost stream, e.g.
    # {"input_tokens": N, "output_tokens": M, "model": "..."}.
    usage: dict[str, Any] | None = None


# Neutral message helpers. A message is {"role", "content"} where content is a
# list of blocks: {"type": "text"|"tool_use"|"tool_result", ...}.


def user_text(text: str) -> dict[str, Any]:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def tool_result(call_id: str, output: str, is_error: bool = False) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": call_id,
                "output": output,
                "is_error": is_error,
            }
        ],
    }


def assistant_message(turn: AssistantTurn) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if turn.text:
        content.append({"type": "text", "text": turn.text})
    for tc in turn.tool_calls:
        content.append(
            {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
        )
    return {"role": "assistant", "content": content}


class LLMProvider(Protocol):
    """Adapters implement this. ``complete`` is one turn of the agent loop."""

    name: str

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
    ) -> AssistantTurn: ...
