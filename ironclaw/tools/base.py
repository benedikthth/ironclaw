"""Tool protocol and execution context.

Tools are the PhD's hands. Keeping them as small, strongly-specified units (clear
JSON schema, structured string result) is a big part of how the harness lets a
cheap model act reliably: the model picks a tool and fills a schema rather than
free-forming an action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..providers.base import ToolSpec


@dataclass
class ToolContext:
    """Shared state handed to every tool invocation."""

    workspace: str
    skills: Any = None  # SkillRegistry; late-bound to avoid an import cycle
    scratch: dict[str, Any] = field(default_factory=dict)


class Tool(Protocol):
    name: str
    description: str
    input_schema: dict[str, Any]

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str: ...


def spec_of(tool: Tool) -> ToolSpec:
    return ToolSpec(name=tool.name, description=tool.description, input_schema=tool.input_schema)


class ToolError(Exception):
    """Raised by a tool for an expected failure; the loop reports it as a tool error
    result rather than crashing the PhD."""
