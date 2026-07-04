"""The Senior Researcher: supervises a task to a terminal state.

The Senior owns task *disposition* (the control model's middle gate). It runs a
PhD, and on failure it consumes the structured Escalation and acts on the
suggested Disposition:

- RELAUNCH_STRONGER -> re-assign the PhD to the next agent up its whitelist,
  with a larger iteration budget.
- REFORMULATE / RETRY_SAME / SPLIT -> (mechanical fallback) grant more budget and
  re-run. True reformulate/split need an LLM Senior to author new task content;
  that is the next enhancement and slots in here. Either way the loop stays
  bounded.
- ESCALATE_TO_PI -> hand the failure up to the PI. This is the only non-pass exit.

Termination is guaranteed: every non-pass attempt increments the relaunch count,
and the baseline policy (control.suggest_disposition) returns ESCALATE_TO_PI once
the count hits ``max_relaunches`` — so a Senior cannot tirespin, it escalates.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol

from ..contracts import Task, TaskResult, TaskStatus
from ..control import (
    AgentCatalog,
    AgentSpec,
    Disposition,
    Escalation,
    Role,
    escalate,
)
from ..observability import active_recorder


class PhDRunner(Protocol):
    """Runs one task attempt with a chosen agent + budget, returns the terminal
    TaskResult. The default implementation drives a real Claude-backed PhD; tests
    inject a deterministic stand-in."""

    def __call__(
        self, task: Task, agent: AgentSpec, budget_iterations: int, workspace: str
    ) -> TaskResult: ...


@dataclass
class Attempt:
    agent_id: str
    status: TaskStatus
    budget: int
    disposition: Disposition | None = None  # what the Senior decided after this


@dataclass
class SupervisionResult:
    task_id: str
    status: TaskStatus  # PASSED, or ESCALATED (handed to the PI)
    result: TaskResult  # the terminal PhD result
    attempts: list[Attempt] = field(default_factory=list)
    escalation: Escalation | None = None  # set only when handed up to the PI


def supervise(
    *,
    task: Task,
    catalog: AgentCatalog,
    runner: PhDRunner,
    workspace: str,
    phd_agent_id: str,
    max_relaunches: int = 2,
    base_budget: int = 16,
) -> SupervisionResult:
    agent_id = phd_agent_id
    budget = base_budget
    relaunches = 0
    attempts: list[Attempt] = []

    rec = active_recorder()
    while True:
        agent = catalog.get(agent_id)
        ws = os.path.join(workspace, f"attempt_{len(attempts)}")
        rec.emit("attempt", role="senior", task_id=task.id, agent_id=agent_id,
                 data={"budget": budget, "n": len(attempts)})
        result = runner(task, agent, budget, ws)

        if result.status is TaskStatus.PASSED:
            attempts.append(Attempt(agent_id, TaskStatus.PASSED, budget))
            return SupervisionResult(task.id, TaskStatus.PASSED, result, attempts)

        # Non-pass: consume the escalation and read the suggested disposition.
        esc = escalate(
            result=result,
            from_role=Role.PHD,
            to_role=Role.SENIOR,
            catalog=catalog,
            agent_id=agent_id,
            prior_relaunches=relaunches,
            max_relaunches=max_relaunches,
        )
        disposition = esc.diagnosis.suggested_disposition
        attempts.append(Attempt(agent_id, result.status, budget, disposition))
        rec.emit("disposition", role="senior", task_id=task.id, agent_id=agent_id,
                 message=disposition.value,
                 data={"failure_kind": esc.diagnosis.failure_kind.value})

        if disposition is Disposition.ESCALATE_TO_PI:
            to_pi = escalate(
                result=result,
                from_role=Role.SENIOR,
                to_role=Role.PI,
                catalog=catalog,
                agent_id=agent_id,
                prior_relaunches=relaunches,
                max_relaunches=max_relaunches,
            )
            rec.emit("escalation", role="senior", task_id=task.id, message="to_pi",
                     data={"notes": to_pi.diagnosis.notes})
            return SupervisionResult(task.id, TaskStatus.ESCALATED, result, attempts, to_pi)

        if disposition is Disposition.RELAUNCH_STRONGER:
            stronger = catalog.next_stronger(Role.PHD, agent_id)
            # The policy only suggests RELAUNCH_STRONGER when one exists.
            agent_id = stronger.id  # type: ignore[union-attr]

        # RELAUNCH_STRONGER, REFORMULATE, RETRY_SAME, SPLIT all re-run with more
        # budget; each bumps the relaunch count, so the policy will escalate to the
        # PI once the bound is hit.
        budget = int(budget * 1.5)
        relaunches += 1


def anthropic_phd_runner(skills_root: str | None = None, *, api_key: str | None = None) -> PhDRunner:
    """Default runner: a real Claude-backed PhD on the assigned agent's model."""

    def run(task: Task, agent: AgentSpec, budget_iterations: int, workspace: str) -> TaskResult:
        from ..providers.anthropic import AnthropicProvider
        from ..runtime.loop import LoopConfig
        from .phd import run_phd

        provider = AnthropicProvider(model=agent.model, api_key=api_key)
        outcome = run_phd(
            task=task,
            provider=provider,
            workspace=workspace,
            skills_root=skills_root,
            config=LoopConfig(max_iterations=budget_iterations),
        )
        return outcome.result

    return run
