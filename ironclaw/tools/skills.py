"""The invoke_skill tool: pull a skill's full procedure on demand."""

from __future__ import annotations

from typing import Any

from .base import Tool, ToolContext, ToolError


class InvokeSkill(Tool):
    name = "invoke_skill"
    description = (
        "Fetch the full procedure for a registered skill by name. Consult the "
        "skills catalog in your instructions first; use this before hand-rolling "
        "any interaction with shared infrastructure."
    )
    input_schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        if ctx.skills is None:
            raise ToolError("no skill registry available")
        skill = ctx.skills.get(args["name"])
        if skill is None:
            available = ", ".join(s.name for s in ctx.skills.list()) or "(none)"
            raise ToolError(f"unknown skill {args['name']!r}. available: {available}")
        return skill.body
