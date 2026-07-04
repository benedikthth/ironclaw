"""The generic agent loop: model <-> tools until submission, verify, iterate —
now with durable suspend/resume across background jobs.

This is the reliability core. A cheap model gets: a bounded loop, structured
tools, a forced verification gate on submission, and a hard iteration budget that
converts "tirespinning" into a definite ESCALATED outcome instead of an infinite
spend. For long-running work it can *suspend* on a background job (state written
to disk) and be *resumed* when the job finishes — surviving process restarts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..contracts import (
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
from ..observability import active_recorder
from ..tools.base import Tool, ToolContext, ToolError, spec_of
from ..verify import all_passed, verify
from .state import PersistedTask, save_state


@dataclass
class LoopConfig:
    # Cheap models explore more before closing out, so give a little headroom.
    # Budget exhaustion is still a hard stop, but see the exhaustion path in
    # _drive: work that satisfies the contract passes even without submit_result.
    max_iterations: int = 16


@dataclass
class LoopOutcome:
    result: TaskResult
    messages: list[dict[str, Any]] = field(default_factory=list)
    # Set when the PhD suspended on a background job. The caller (or a scheduler)
    # resumes via resume_loop(state_path, ...) once the job completes.
    state_path: str | None = None
    waiting_on: str | None = None  # job_id


def run_loop(
    *,
    provider: LLMProvider,
    system: str,
    task: Task,
    tools: list[Tool],
    ctx: ToolContext,
    config: LoopConfig | None = None,
    state_path: str | None = None,
) -> LoopOutcome:
    """Start a task fresh."""
    config = config or LoopConfig()
    messages: list[dict[str, Any]] = [user_text(_task_prompt(task))]
    return _drive(
        provider=provider,
        system=system,
        task=task,
        tools=tools,
        ctx=ctx,
        messages=messages,
        iterations=0,
        max_iterations=config.max_iterations,
        state_path=state_path,
    )


def resume_loop(
    *,
    provider: LLMProvider,
    task: Task,
    tools: list[Tool],
    ctx: ToolContext,
    state: PersistedTask,
    job_output: str,
    state_path: str,
) -> LoopOutcome:
    """Resume a suspended task: fill in the awaited job's tool result, continue."""
    messages = list(state.messages)
    awaiting = state.awaiting or {}
    messages.append(tool_result(awaiting["call_id"], job_output))
    return _drive(
        provider=provider,
        system=state.system,
        task=task,
        tools=tools,
        ctx=ctx,
        messages=messages,
        iterations=state.iterations,
        max_iterations=state.max_iterations,
        state_path=state_path,
    )


def _drive(
    *,
    provider: LLMProvider,
    system: str,
    task: Task,
    tools: list[Tool],
    ctx: ToolContext,
    messages: list[dict[str, Any]],
    iterations: int,
    max_iterations: int,
    state_path: str | None,
) -> LoopOutcome:
    by_name = {t.name: t for t in tools}
    specs = [spec_of(t) for t in tools]
    rec = active_recorder()
    if iterations == 0:
        rec.emit("task.start", role="phd", task_id=task.id, message=task.goal)

    while iterations < max_iterations:
        iterations += 1
        turn: AssistantTurn = provider.complete(system, messages, specs)
        messages.append(assistant_message(turn))
        if turn.usage:
            rec.emit("usage", role="phd", task_id=task.id, data=dict(turn.usage))

        if not turn.tool_calls:
            messages.append(
                user_text(
                    "You produced no tool call. Use a tool to make progress, or "
                    "call submit_result when the acceptance criteria are met."
                )
            )
            continue

        for call in turn.tool_calls:
            # await_job is intercepted here: finished -> feed result inline;
            # pending -> suspend the whole PhD to disk and hand back control.
            if call.name == "await_job":
                suspended = _handle_await(
                    call=call,
                    ctx=ctx,
                    messages=messages,
                    task=task,
                    system=system,
                    iterations=iterations,
                    max_iterations=max_iterations,
                    state_path=state_path,
                )
                if suspended is not None:
                    return suspended
                continue

            tool = by_name.get(call.name)
            if tool is None:
                messages.append(tool_result(call.id, f"unknown tool {call.name!r}", is_error=True))
                continue
            rec.emit("tool.call", role="phd", task_id=task.id, message=call.name)
            try:
                messages.append(tool_result(call.id, tool.run(call.arguments, ctx)))
            except ToolError as e:
                rec.emit("tool.error", role="phd", task_id=task.id, message=call.name, data={"error": str(e)})
                messages.append(tool_result(call.id, f"error: {e}", is_error=True))

        if "submission" in ctx.scratch:
            outcome = _finalize_submission(task, ctx, messages, iterations)
            if outcome is not None:
                return outcome

    # Budget exhausted. Mechanical verification — not the model's submit call — is
    # the source of truth: if the acceptance contract is satisfied, the task is
    # done even though the PhD never formally submitted (common with cheap models
    # that do the work but forget the closing step). Only escalate if it truly
    # isn't done.
    checks = verify(task, ctx.workspace)
    if all_passed(checks):
        rec.emit("task.result", role="phd", task_id=task.id, message="passed (no submit)",
                 data={"status": "passed", "iterations": iterations})
        return LoopOutcome(
            TaskResult(
                task_id=task.id,
                status=TaskStatus.PASSED,
                checks=checks,
                summary=(
                    "Acceptance criteria satisfied at budget exhaustion (the PhD "
                    "completed the work but did not call submit_result)."
                ),
                iterations=iterations,
            ),
            messages,
        )
    rec.emit("task.result", role="phd", task_id=task.id, message="escalated",
             data={"status": "escalated", "iterations": iterations})
    return LoopOutcome(
        TaskResult(
            task_id=task.id,
            status=TaskStatus.ESCALATED,
            checks=checks,
            summary=(
                f"Iteration budget ({max_iterations}) exhausted without passing "
                "verification. Escalating to the Senior Researcher."
            ),
            iterations=iterations,
        ),
        messages,
    )


def _handle_await(
    *,
    call: Any,
    ctx: ToolContext,
    messages: list[dict[str, Any]],
    task: Task,
    system: str,
    iterations: int,
    max_iterations: int,
    state_path: str | None,
) -> LoopOutcome | None:
    """Return a WAITING LoopOutcome if the job is pending, else None (result fed
    inline and the loop continues)."""
    from ..jobs import JobStatus  # local import to avoid a cycle

    if ctx.jobs is None:
        messages.append(tool_result(call.id, "error: no job backend", is_error=True))
        return None
    job_id = call.arguments["job_id"]
    status, output = ctx.jobs.poll(job_id)
    if status is not JobStatus.RUNNING:
        messages.append(tool_result(call.id, output))
        return None

    # Pending: persist and suspend. Note we deliberately do NOT append a
    # tool_result for this call — resume_loop fills it in with the job output.
    if state_path is None:
        state_path = f"{ctx.workspace}/.ironclaw_state.json"
    save_state(
        state_path,
        PersistedTask(
            task=task,
            system=system,
            messages=messages,
            iterations=iterations,
            workspace=ctx.workspace,
            max_iterations=max_iterations,
            skills_root=getattr(ctx.skills, "root", None),
            awaiting={"job_id": job_id, "call_id": call.id},
        ),
    )
    active_recorder().emit("job.await", role="phd", task_id=task.id, message=job_id)
    return LoopOutcome(
        TaskResult(
            task_id=task.id,
            status=TaskStatus.WAITING,
            summary=f"Suspended waiting on job {job_id}.",
            iterations=iterations,
        ),
        messages,
        state_path=state_path,
        waiting_on=job_id,
    )


def _finalize_submission(
    task: Task,
    ctx: ToolContext,
    messages: list[dict[str, Any]],
    iterations: int,
) -> LoopOutcome | None:
    rec = active_recorder()
    checks = verify(task, ctx.workspace)
    if all_passed(checks):
        sub = ctx.scratch.pop("submission")
        rec.emit("verify", role="phd", task_id=task.id, data={"passed": True})
        rec.emit("task.result", role="phd", task_id=task.id, message="passed",
                 data={"status": "passed", "iterations": iterations})
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
    rec.emit("verify", role="phd", task_id=task.id, data={"passed": False})
    failing = "\n".join(
        f"- FAIL [{c.criterion.description}]: {c.detail}" for c in checks if not c.passed
    )
    messages.append(
        user_text(
            "Verification failed. These acceptance criteria are not yet met:\n"
            f"{failing}\nAddress them and submit again."
        )
    )
    return None


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
