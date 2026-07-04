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


class TaskAuthorLike(Protocol):
    """Structural type for the Senior's authoring seam (see authoring.TaskAuthor).
    Kept local to avoid an import cycle with the authoring module."""

    def reformulate(self, task: Task, diagnosis) -> Task: ...
    def split(self, task: Task, diagnosis) -> list[Task]: ...


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
    subtasks: list["SupervisionResult"] = field(default_factory=list)  # set when SPLIT


def supervise(
    *,
    task: Task,
    catalog: AgentCatalog,
    runner: PhDRunner,
    workspace: str,
    phd_agent_id: str,
    max_relaunches: int = 2,
    base_budget: int = 16,
    author: "TaskAuthorLike | None" = None,
    split_depth: int = 0,
    max_split_depth: int = 1,
) -> SupervisionResult:
    agent_id = phd_agent_id
    budget = base_budget
    relaunches = 0
    attempts: list[Attempt] = []

    rec = active_recorder()
    while True:
        agent = catalog.get(agent_id)
        # All attempts share the task's workspace (which is the shared *project*
        # workspace): a relaunch on a stronger agent builds on what the weaker one
        # produced instead of starting from zero.
        ws = workspace
        rec.emit("attempt", role="senior", task_id=task.id, project_id=task.project_id,
                 agent_id=agent_id, data={"budget": budget, "n": len(attempts)})
        result = runner(task, agent, budget, ws)

        if result.status is TaskStatus.PASSED:
            attempts.append(Attempt(agent_id, TaskStatus.PASSED, budget))
            rec.emit("task.done", role="senior", task_id=task.id, project_id=task.project_id,
                     message="passed", data={"status": "passed"})
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
        rec.emit("disposition", role="senior", task_id=task.id, project_id=task.project_id,
                 agent_id=agent_id, message=disposition.value,
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
            rec.emit("escalation", role="senior", task_id=task.id, project_id=task.project_id,
                     message="to_pi", data={"notes": to_pi.diagnosis.notes})
            rec.emit("task.done", role="senior", task_id=task.id, project_id=task.project_id,
                     message="escalated", data={"status": "escalated"})
            return SupervisionResult(task.id, TaskStatus.ESCALATED, result, attempts, to_pi)

        if disposition is Disposition.RELAUNCH_STRONGER:
            stronger = catalog.next_stronger(Role.PHD, agent_id)
            # The policy only suggests RELAUNCH_STRONGER when one exists.
            agent_id = stronger.id  # type: ignore[union-attr]

        # Content-authoring dispositions: if a Senior author is wired in, actually
        # rewrite or split the work instead of just re-running with more budget.
        elif author is not None and disposition is Disposition.REFORMULATE:
            task = author.reformulate(task, esc.diagnosis)
            rec.emit("reformulate", role="senior", task_id=task.id, message=task.goal[:60])
        elif author is not None and disposition is Disposition.SPLIT and split_depth < max_split_depth:
            return _split(
                task=task, diagnosis=esc.diagnosis, catalog=catalog, runner=runner,
                workspace=workspace, phd_agent_id=phd_agent_id, max_relaunches=max_relaunches,
                base_budget=base_budget, author=author, split_depth=split_depth,
                max_split_depth=max_split_depth, attempts=attempts, last=result, rec=rec,
            )

        # RELAUNCH_STRONGER / REFORMULATE / RETRY_SAME / (un-authored) SPLIT re-run
        # with more budget; each bumps the relaunch count, so the policy escalates
        # to the PI once the bound is hit.
        budget = int(budget * 1.5)
        relaunches += 1


def _split(
    *, task, diagnosis, catalog, runner, workspace, phd_agent_id, max_relaunches,
    base_budget, author, split_depth, max_split_depth, attempts, last, rec,
) -> SupervisionResult:
    """Author subtasks and supervise each. The parent passes iff every subtask
    passes; otherwise it escalates to the PI. Bounded: a subtask can't itself
    split (``split_depth`` guard in the caller)."""
    subtasks = author.split(task, diagnosis)
    rec.emit("split", role="senior", task_id=task.id,
             data={"subtasks": [s.id for s in subtasks]})
    sub_results: list[SupervisionResult] = []
    for st in subtasks:
        sub_results.append(
            supervise(
                task=st, catalog=catalog, runner=runner,
                workspace=os.path.join(workspace, f"split_{st.id}"),
                phd_agent_id=phd_agent_id, max_relaunches=max_relaunches,
                base_budget=base_budget, author=author,
                split_depth=split_depth + 1, max_split_depth=max_split_depth,
            )
        )
    if all(s.status is TaskStatus.PASSED for s in sub_results):
        rec.emit("task.done", role="senior", task_id=task.id, project_id=task.project_id,
                 message="passed", data={"status": "passed"})
        return SupervisionResult(task.id, TaskStatus.PASSED, last, attempts, None, sub_results)
    to_pi = escalate(
        result=last, from_role=Role.SENIOR, to_role=Role.PI, catalog=catalog,
        agent_id=phd_agent_id, prior_relaunches=max_relaunches, max_relaunches=max_relaunches,
    )
    rec.emit("escalation", role="senior", task_id=task.id, project_id=task.project_id,
             message="to_pi", data={"notes": "split subtasks did not all pass"})
    rec.emit("task.done", role="senior", task_id=task.id, project_id=task.project_id,
             message="escalated", data={"status": "escalated"})
    return SupervisionResult(task.id, TaskStatus.ESCALATED, last, attempts, to_pi, sub_results)


def provider_runner(registry=None, *, skills_root: str | None = None) -> PhDRunner:
    """Real-PhD runner that routes each agent to its own provider/model via the
    registry. The assigned ``AgentSpec.provider`` picks the endpoint, so one lab
    can freely mix Anthropic / OpenAI / OpenRouter / local models."""
    from ..providers.registry import default_registry

    registry = registry or default_registry()

    def run(task: Task, agent: AgentSpec, budget_iterations: int, workspace: str) -> TaskResult:
        from ..runtime.loop import LoopConfig
        from .phd import run_phd

        provider = registry.build(agent.provider, agent.model)
        outcome = run_phd(
            task=task, provider=provider, workspace=workspace, skills_root=skills_root,
            config=LoopConfig(max_iterations=budget_iterations),
        )
        return outcome.result

    return run


def anthropic_phd_runner(skills_root: str | None = None, *, api_key: str | None = None) -> PhDRunner:
    """Convenience: a runner backed by Anthropic only (for `provider="anthropic"`
    agents). Prefer ``provider_runner`` with a registry for multi-provider labs."""
    from ..providers.registry import ProviderConfig, ProviderRegistry

    reg = ProviderRegistry([ProviderConfig("anthropic", "anthropic", api_key=api_key)])
    return provider_runner(reg, skills_root=skills_root)
