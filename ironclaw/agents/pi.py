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

from dataclasses import dataclass, field
from typing import Callable

from ..contracts import Task, TaskStatus
from ..control import AgentCatalog
from .senior import PhDRunner, SupervisionResult, supervise


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
) -> LabResult:
    projects: list[ProjectResult] = []
    for proj in problem.projects:
        task_results: list[SupervisionResult] = []
        for task in proj.tasks:
            ws = f"{workspace}/{proj.id}/{task.id}"
            sup = supervise(
                task=task,
                catalog=catalog,
                runner=runner,
                workspace=ws,
                phd_agent_id=phd_agent_id,
                max_relaunches=max_relaunches,
                base_budget=base_budget,
            )
            task_results.append(sup)
        proj_ok = all(s.status is TaskStatus.PASSED for s in task_results)
        projects.append(
            ProjectResult(
                proj.id,
                TaskStatus.PASSED if proj_ok else TaskStatus.ESCALATED,
                task_results,
            )
        )
    lab_ok = all(p.status is TaskStatus.PASSED for p in projects)
    return LabResult(
        problem.statement,
        TaskStatus.PASSED if lab_ok else TaskStatus.ESCALATED,
        projects,
    )
