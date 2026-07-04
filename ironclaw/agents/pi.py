"""The Principal Investigator: owns a lab (one hard problem).

The PI decomposes a problem into projects, hands each project's tasks to the
Senior layer, and aggregates results upward. It owns the top gate — project
disposition: a task a Senior escalates all the way up arrives here, and the PI
decides the project's fate.

This slice runs the lifecycle end-to-end with a deterministic decomposition (a
static list of projects/tasks). Authoring projects from a raw problem statement,
and reshaping/spawning projects in response to a PI-level escalation, is the
LLM-PI seam — the orchestration and the aggregation are in place; the
`Decomposer` is pluggable, exactly like the PhD/Postdoc runners below it.

The full stack composes: PI -> Senior -> Postdoc -> PhD. The PI hands the Senior
a `runner`; in production that runner is the Postdoc-gated real-PhD runner, so
one call drives the whole institute.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Protocol

from ..contracts import Task, TaskResult, TaskStatus
from ..control import AgentCatalog
from ..observability import Recorder, active_recorder, using
from .senior import PhDRunner, SupervisionResult, supervise


class PIDecision(str, Enum):
    """The PI's response when a Senior escalates a task all the way up."""

    ACCEPT = "accept"  # accept the failure, record it, carry on
    RETRY = "retry"  # give it another full run (fresh relaunch ladder)
    ABANDON = "abandon"  # abandon the whole project


class Overseer(Protocol):
    """The human-in-the-loop seam. The PI consults it on every task a Senior
    gives up on — this is where "the PI steps in if direction is drifting"
    actually happens. The default is non-interactive; a UI/CLI plugs in here."""

    def on_escalation(self, supervision: SupervisionResult) -> PIDecision: ...


class AutoOverseer:
    """Non-interactive default: accept whatever the Seniors couldn't solve, so a
    lab runs unattended and the failures surface in the PI action list."""

    def on_escalation(self, supervision: SupervisionResult) -> PIDecision:
        return PIDecision.ACCEPT


class ConsoleOverseer:
    """Minimal human interaction: prompt at the terminal on each escalation."""

    def on_escalation(self, supervision: SupervisionResult) -> PIDecision:
        esc = supervision.escalation
        note = esc.diagnosis.notes if esc else ""
        print(f"\n[PI] task {supervision.task_id} escalated: {note}")
        ans = input("    [a]ccept / [r]etry / a[b]andon project? ").strip().lower()
        return {"r": PIDecision.RETRY, "b": PIDecision.ABANDON}.get(ans, PIDecision.ACCEPT)


@dataclass
class ProjectSpec:
    id: str
    goal: str
    tasks: list[Task]


@dataclass
class Problem:
    statement: str
    projects: list[ProjectSpec]


@dataclass
class ProjectResult:
    project_id: str
    status: TaskStatus  # PASSED iff every task passed; else ESCALATED
    tasks: list[SupervisionResult] = field(default_factory=list)


@dataclass
class LabResult:
    statement: str
    status: TaskStatus  # PASSED iff every project passed
    projects: list[ProjectResult] = field(default_factory=list)

    def escalations(self) -> list[SupervisionResult]:
        """Tasks that reached the PI (a Senior gave up) — the PI's action list."""
        return [
            s
            for p in self.projects
            for s in p.tasks
            if s.status is TaskStatus.ESCALATED
        ]


# A decomposer turns a raw problem statement into projects. Default is LLM-driven
# (the seam); tests and demos pass a static Problem directly.
Decomposer = Callable[[str], list[ProjectSpec]]


def run_lab(
    *,
    problem: Problem,
    catalog: AgentCatalog,
    runner: PhDRunner,
    workspace: str,
    phd_agent_id: str,
    max_relaunches: int = 2,
    base_budget: int = 16,
    overseer: Overseer | None = None,
    recorder: Recorder | None = None,
    max_pi_retries: int = 3,
    author=None,
    memory=None,
    learner=None,
) -> LabResult:
    overseer = overseer or AutoOverseer()
    with using(recorder):
        rec = active_recorder()
        rec.emit("lab.start", role="pi", lab=problem.statement,
                 data={"projects": len(problem.projects)})
        projects: list[ProjectResult] = []

        for proj in problem.projects:
            rec.emit("project.start", role="pi", project_id=proj.id, message=proj.goal)
            # One shared workspace per project: a task's artifacts are simply
            # present when a later task in the same project runs.
            proj_ws = os.path.join(workspace, proj.id)
            os.makedirs(proj_ws, exist_ok=True)
            task_results: list[SupervisionResult] = []
            passed_ids: set[str] = set()
            abandoned = False

            for task in _dependency_order(proj.tasks):
                unmet = [d for d in task.depends_on if d not in passed_ids]
                if unmet:
                    # A dependency didn't pass — the inputs this task needs aren't
                    # there. Block rather than run it into a confusing failure.
                    rec.emit("task.blocked", role="pi", task_id=task.id, project_id=proj.id, data={"unmet": unmet})
                    rec.emit("task.done", role="pi", task_id=task.id, project_id=proj.id,
                             message="blocked", data={"status": "blocked"})
                    task_results.append(
                        SupervisionResult(
                            task.id, TaskStatus.ESCALATED,
                            TaskResult(task.id, TaskStatus.ESCALATED, summary=f"blocked: unmet dependencies {unmet}"),
                        )
                    )
                    continue

                sup = _run_task(task, proj, catalog, runner, proj_ws, phd_agent_id, max_relaunches, base_budget, author=author)
                # PI gate: a task the Senior gave up on comes here for a decision.
                retries = 0
                while sup.status is TaskStatus.ESCALATED and retries < max_pi_retries:
                    decision = overseer.on_escalation(sup)
                    rec.emit("pi.decision", role="pi", task_id=task.id, message=decision.value)
                    if decision is PIDecision.ACCEPT:
                        break
                    if decision is PIDecision.ABANDON:
                        abandoned = True
                        break
                    retries += 1  # RETRY: another full run in the shared workspace
                    sup = _run_task(task, proj, catalog, runner, proj_ws, phd_agent_id, max_relaunches, base_budget, author=author)
                task_results.append(sup)
                if sup.status is TaskStatus.PASSED:
                    passed_ids.add(task.id)
                    # Share the finding across labs, and learn a skill from it.
                    if memory is not None:
                        memory.add(
                            "finding", f"{task.goal}\n{sup.result.summary}",
                            ref={"task_id": task.id, "summary": sup.result.summary,
                                 "artifacts": [{"path": a.path, "kind": a.kind} for a in sup.result.artifacts]},
                        )
                    if learner is not None:
                        tool_calls = sum(
                            1 for e in rec.events if e.kind == "tool.call" and e.task_id == task.id
                        )
                        learner.consider(task, sup, proj_ws, tool_calls)
                if abandoned:
                    break

            proj_ok = not abandoned and all(s.status is TaskStatus.PASSED for s in task_results)
            status = TaskStatus.PASSED if proj_ok else TaskStatus.ESCALATED
            rec.emit("project.result", role="pi", project_id=proj.id, message=status.value,
                     data={"abandoned": abandoned})
            projects.append(ProjectResult(proj.id, status, task_results))

        lab_ok = all(p.status is TaskStatus.PASSED for p in projects)
        lab_status = TaskStatus.PASSED if lab_ok else TaskStatus.ESCALATED
        rec.emit("lab.result", role="pi", lab=problem.statement, message=lab_status.value)
        if memory is not None:
            memory.add(
                "lab", problem.statement, tags=[lab_status.value],
                ref={"statement": problem.statement, "status": lab_status.value,
                     "projects": [{"id": p.project_id, "status": p.status.value} for p in projects]},
            )
        return LabResult(problem.statement, lab_status, projects)


def _dependency_order(tasks: list[Task]) -> list[Task]:
    """Stable topological order by ``depends_on`` (declared order within each
    ready wave). Unknown deps are ignored; a dependency cycle falls back to
    declared order for the tasks left in it."""
    ids = {t.id for t in tasks}
    ordered: list[Task] = []
    seen: set[str] = set()
    progressed = True
    while progressed:
        progressed = False
        for t in tasks:
            if t.id in seen:
                continue
            if all(d in seen for d in t.depends_on if d in ids):
                ordered.append(t)
                seen.add(t.id)
                progressed = True
    ordered.extend(t for t in tasks if t.id not in seen)  # cycle leftovers
    return ordered


def _run_task(
    task, proj, catalog, runner, workspace, phd_agent_id, max_relaunches, base_budget,
    *, author=None,
) -> SupervisionResult:
    # workspace is the shared project workspace; the Senior runs all attempts in it.
    return supervise(
        task=task,
        catalog=catalog,
        runner=runner,
        workspace=workspace,
        phd_agent_id=phd_agent_id,
        max_relaunches=max_relaunches,
        base_budget=base_budget,
        author=author,
    )
