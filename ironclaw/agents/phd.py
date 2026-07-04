"""The PhD agent: owns exactly one task, does the work, self-verifies.

~90% of the institute's real work happens here. The PhD assembles its own
context (task contract + skills catalog), then runs the shared agent loop with a
standard toolset. Everything above it (Senior, PI) is management wrapped around
this executor.
"""

from __future__ import annotations

import os
from typing import Any

from ..contracts import Task, TaskResult
from ..providers.base import LLMProvider
from ..skills import SkillRegistry
from ..tools.base import Tool, ToolContext
from ..tools.files import ReadFile, WriteFile
from ..tools.python_exec import PythonExec
from ..tools.skills import InvokeSkill
from ..tools.submit import SubmitResult
from ..runtime.loop import LoopConfig, LoopOutcome, run_agent_loop

SYSTEM_TEMPLATE = """You are a PhD researcher at the Ironclaw institute. You own a \
single task and must deliver artifacts that satisfy its acceptance criteria.

Operating rules:
- Prefer a registered skill over improvising any interaction with shared \
infrastructure. Skills encode procedures that already work.
- Produce durable artifacts (files in your workspace): reports as Markdown, code \
as source files, collected data in a queryable format.
- Test your own work before submitting. You are graded only on the acceptance \
criteria, checked mechanically.
- When done, call submit_result with every artifact you produced.

## Available skills
{skills_catalog}
"""


def default_tools() -> list[Tool]:
    return [WriteFile(), ReadFile(), PythonExec(), InvokeSkill(), SubmitResult()]


def run_phd(
    *,
    task: Task,
    provider: LLMProvider,
    workspace: str,
    skills_root: str | None = None,
    tools: list[Tool] | None = None,
    config: LoopConfig | None = None,
) -> LoopOutcome:
    os.makedirs(workspace, exist_ok=True)
    registry = SkillRegistry(skills_root) if skills_root else None
    ctx = ToolContext(workspace=workspace, skills=registry)
    system = SYSTEM_TEMPLATE.format(
        skills_catalog=registry.catalog() if registry else "(no skills registered)"
    )
    return run_agent_loop(
        provider=provider,
        system=system,
        task=task,
        tools=tools or default_tools(),
        ctx=ctx,
        config=config,
    )
