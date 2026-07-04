"""The generic agent loop: model <-> tools until submission, verify, iterate.

This is the reliability core. A cheap model gets: a bounded loop, structured
tools, a forced verification gate on submission, and a hard iteration budget that
converts "tirespinning" into a definite ESCALATED/FAILED outcome instead of an
infinite spend. Every superior role in the institute ultimately relies on this
loop terminating honestly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..contracts import (
    Artifact,
    Task,
    TaskResult,
    TaskStatus,
)
from ..providers.base import (
    AssistantTurn,
    LLMProvider,
    assistant_message,
    tool_result,
    user_text,
)
from ..tools.base import Tool, ToolContext, ToolError, spec_of
from ..verify import all_passed, verify


@dataclass
class LoopConfig:
    max_iterations: int = 12


@dataclass
class LoopOutcome:
    result: TaskResult
    messages: list[dict[str, Any]] = field(default_factory=list)


def run_agent_loop(
    *,
    provider: LLMProvider,
    system: str,
    task: Task,
    tools: list[Tool],
    ctx: ToolContext,
    config: LoopConfig | None = None,
) -> LoopOutcome:
    config = config or LoopConfig()
    by_name = {t.name: t for t in tools}
    specs = [spec_of(t) for t in tools]
    messages: list[dict[str, Any]] = [user_text(_task_prompt(task))]

    iterations = 0
    while iterations < config.max_iterations:
        iterations += 1
        turn: AssistantTurn = provider.complete(system, messages, specs)
        messages.append(assistant_message(turn))

        if not turn.tool_calls:
            messages.append(
                user_text(
                    "You produced no tool call. Use a tool to make progress, or "
                    "call submit_result when the acceptance criteria are met."
                )
            )
            continue

        for call in turn.tool_calls:
            tool = by_name.get(call.name)
            if tool is None:
                messages.append(tool_result(call.id, f"unknown tool {call.name!r}", is_error=True))
                continue
            try:
                output = tool.run(call.arguments, ctx)
                messages.append(tool_result(call.id, output))
            except ToolError as e:
                messages.append(tool_result(call.id, f"error: {e}", is_error=True))

        if "submission" in ctx.scratch:
            checks = verify(task, ctx.workspace)
            if all_passed(checks):
                sub = ctx.scratch.pop("submission")
                return LoopOutcome(
                    TaskResult(
                        task_id=task.id,
                        status=TaskStatus.PASSED,
                        artifacts=sub["artifacts"],
                        checks=checks,
                        summary=sub["summary"],
                        iterations=iterations,
                    ),
                    messages,
                )
            # Failed verification: report the failing checks and let the PhD fix.
            ctx.scratch.pop("submission")
            failing = "\n".join(
                f"- FAIL [{c.criterion.description}]: {c.detail}" for c in checks if not c.passed
            )
            messages.append(
                user_text(
                    "Verification failed. These acceptance criteria are not yet "
                    f"met:\n{failing}\nAddress them and submit again."
                )
            )

    # Budget exhausted: escalate rather than silently fail.
    checks = verify(task, ctx.workspace)
    return LoopOutcome(
        TaskResult(
            task_id=task.id,
            status=TaskStatus.ESCALATED,
            artifacts=[],
            checks=checks,
            summary=(
                f"Iteration budget ({config.max_iterations}) exhausted without "
                "passing verification. Escalating to the Senior Researcher."
            ),
            iterations=iterations,
        ),
        messages,
    )


def _task_prompt(task: Task) -> str:
    lines = [f"# Task {task.id}", "", "## Goal", task.goal, "", "## Acceptance criteria"]
    for i, c in enumerate(task.acceptance, 1):
        lines.append(f"{i}. {c.description}")
    if task.inputs:
        lines += ["", "## Inputs", *(f"- {k}: {v}" for k, v in task.inputs.items())]
    lines += [
        "",
        "Work the task, then call submit_result. You are graded only on the "
        "acceptance criteria above.",
    ]
    return "\n".join(lines)
