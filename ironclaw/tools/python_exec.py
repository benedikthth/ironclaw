"""Run Python in the workspace.

This is the PhD's general-purpose actuator for the self-contained cases
(transform data, run a simulation, compute a result). Long-running/queued work
(slurm, training) goes through the durable job tools in slice 2, not here.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

from .base import Tool, ToolContext


class PythonExec(Tool):
    name = "python_exec"
    description = (
        "Run a short Python script in the workspace directory and return its "
        "combined stdout/stderr. Use for self-contained computation and data "
        "transforms. Not for long-running or queued jobs."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python source to execute."},
            "timeout_s": {"type": "integer", "default": 60},
        },
        "required": ["code"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        timeout = int(args.get("timeout_s", 60))
        try:
            proc = subprocess.run(
                [sys.executable, "-c", args["code"]],
                cwd=ctx.workspace,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return f"[timeout after {timeout}s]"
        out = (proc.stdout or "") + (proc.stderr or "")
        return f"[exit {proc.returncode}]\n{out}".rstrip()
