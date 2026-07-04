"""A general shell — the PhD's hands for interacting with anything.

This is deliberately unspecialized: a PhD uses it to ssh into a host, submit and
poll a cluster job (`sbatch`/`squeue`), curl an API, clone a repo, run a CLI —
whatever a real researcher would type. The institute has *no* per-system feature
(no "slurm mode"); interacting with an external system is just running the right
commands, guided by a skill the Infrastructure Manager packaged. Long-running or
queued work goes through the durable job tools; use this for synchronous steps.
"""

from __future__ import annotations

import subprocess
from typing import Any

from .base import Tool, ToolContext


class ShellExec(Tool):
    name = "shell"
    description = (
        "Run a shell command from your workspace and return its combined "
        "stdout/stderr and exit code. Use for any synchronous interaction with a "
        "system — ssh, cluster commands (sbatch/squeue), curl, git, package "
        "managers, CLIs. For long-running or queued work that you must wait on, "
        "use start_job/await_job instead."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to run."},
            "timeout_s": {"type": "integer", "default": 120},
        },
        "required": ["command"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> str:
        timeout = int(args.get("timeout_s", 120))
        try:
            proc = subprocess.run(
                args["command"],
                shell=True,
                cwd=ctx.workspace,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return f"[timeout after {timeout}s]"
        out = (proc.stdout or "") + (proc.stderr or "")
        return f"[exit {proc.returncode}]\n{out}".rstrip()
