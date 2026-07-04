"""File tools, sandboxed to the task workspace."""

from __future__ import annotations

import os
from typing import Any

from ..contracts import workspace_path
from .base import Tool, ToolContext, ToolError


class WriteFile(Tool):
    name = "write_file"
    description = "Create or overwrite a UTF-8 text file at a workspace-relative path."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative path."},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        try:
            full = workspace_path(ctx.workspace, args["path"])
        except ValueError as e:
            raise ToolError(str(e)) from e
        os.makedirs(os.path.dirname(full) or ctx.workspace, exist_ok=True)
        with open(full, "w", encoding="utf-8") as fh:
            fh.write(args["content"])
        return f"wrote {len(args['content'])} bytes to {args['path']}"


class ReadFile(Tool):
    name = "read_file"
    description = "Read a UTF-8 text file at a workspace-relative path."
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        try:
            full = workspace_path(ctx.workspace, args["path"])
        except ValueError as e:
            raise ToolError(str(e)) from e
        if not os.path.exists(full):
            raise ToolError(f"no such file: {args['path']}")
        with open(full, "r", encoding="utf-8") as fh:
            return fh.read()
