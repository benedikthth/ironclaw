"""The PhD agent: owns exactly one task, does the work, self-verifies.

~90% of the institute's real work happens here. The PhD assembles its own
context (task contract + skills catalog), then runs the shared agent loop with a
standard toolset. It can suspend on a durable background job and be resumed later
(see ``resume_phd``). Everything above it (Senior, PI) is management wrapped
around this executor.
"""

from __future__ import annotations

import os
from typing import Any

from ..contracts import Task
from ..jobs import JobBackend, JobStatus, LocalProcessBackend
from ..providers.base import LLMProvider
from ..skills import SkillRegistry
from ..tools.base import Tool, ToolContext
from ..tools.files import ReadFile, WriteFile
from ..tools.jobs import AwaitJob, StartJob
from ..tools.python_exec import PythonExec
from ..tools.shell import ShellExec
from ..tools.skills import InvokeSkill
from ..tools.submit import SubmitResult
from ..runtime.loop import LoopConfig, LoopOutcome, resume_loop, run_loop
from ..runtime.state import PersistedTask, load_state

SYSTEM_TEMPLATE = """You are a PhD researcher at the Ironclaw institute. You own a \
single task and must deliver artifacts that satisfy its acceptance criteria.

Operating rules:
- You have a general shell: to interact with any external system (ssh into a \
host, submit/poll a cluster job, call an API, run a CLI) just run the commands, \
exactly as a researcher would. There is no special mode for any system.
- Prefer a registered skill over improvising an interaction with shared \
infrastructure. Skills encode procedures that already work; consult the catalog.
- For long-running or queued work you must wait on, use start_job then \
await_job; never block inline.
- Produce durable artifacts (files in your workspace): reports as Markdown, code \
as source files, collected data in a queryable format.
- Test your own work before submitting. You are graded only on the acceptance \
criteria, checked mechanically.
- When done, call submit_result with every artifact you produced.

## Available skills
{skills_catalog}
"""


def default_tools() -> list[Tool]:
    return [
        WriteFile(),
        ReadFile(),
        PythonExec(),
        ShellExec(),
        InvokeSkill(),
        StartJob(),
        AwaitJob(),
        SubmitResult(),
    ]


def _system_for(registry: SkillRegistry | None) -> str:
    return SYSTEM_TEMPLATE.format(
        skills_catalog=registry.catalog() if registry else "(no skills registered)"
    )


def run_phd(
    *,
    task: Task,
    provider: LLMProvider,
    workspace: str,
    skills_root: str | None = None,
    jobs: JobBackend | None = None,
    tools: list[Tool] | None = None,
    config: LoopConfig | None = None,
    state_path: str | None = None,
) -> LoopOutcome:
    os.makedirs(workspace, exist_ok=True)
    registry = SkillRegistry(skills_root) if skills_root else None
    backend = jobs or LocalProcessBackend(workspace)
    ctx = ToolContext(workspace=workspace, skills=registry, jobs=backend)
    return run_loop(
        provider=provider,
        system=_system_for(registry),
        task=task,
        tools=tools or default_tools(),
        ctx=ctx,
        config=config,
        state_path=state_path,
    )


def resume_phd(
    *,
    state_path: str,
    provider: LLMProvider,
    jobs: JobBackend | None = None,
    tools: list[Tool] | None = None,
) -> LoopOutcome:
    """Resume a PhD suspended on a background job. Returns a fresh WAITING outcome
    if the job still is not done, otherwise drives the task to completion.
    """
    state = load_state(state_path)
    registry = SkillRegistry(state.skills_root) if state.skills_root else None
    backend = jobs or LocalProcessBackend(state.workspace)
    ctx = ToolContext(workspace=state.workspace, skills=registry, jobs=backend)

    awaiting = state.awaiting or {}
    status, output = backend.poll(awaiting["job_id"])
    if status is JobStatus.RUNNING:
        # Still not ready: hand back a WAITING outcome unchanged.
        from ..contracts import TaskResult, TaskStatus

        return LoopOutcome(
            TaskResult(
                task_id=state.task.id,
                status=TaskStatus.WAITING,
                summary=f"Still waiting on job {awaiting['job_id']}.",
                iterations=state.iterations,
            ),
            list(state.messages),
            state_path=state_path,
            waiting_on=awaiting["job_id"],
        )

    return resume_loop(
        provider=provider,
        task=state.task,
        tools=tools or default_tools(),
        ctx=ctx,
        state=state,
        job_output=output,
        state_path=state_path,
    )
